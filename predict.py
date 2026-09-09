#!/usr/bin/env python3
"""
HiDRA -- the High-Dimensional Rodent Annotator.
Apply the per-lab-head mouse-behaviour classifier ensemble to a folder of pose tracking parquets.

QUICK START
  # 0. one-time: fetch the model weights (~660 MB) from the Hugging Face Hub into models/
  pip install huggingface_hub && python download_models.py
  # 1. see which (lab, action) classifiers exist:
  python predict.py --list-heads
  # 2a. run a few of them (zero-shot -- no labels of your own needed):
  python predict.py /path/to/parquet_folder --labs LyricalHare --actions attack,sniff \
      --out results/ --pix-per-cm 16 --fps 30
  # 2b. or select per-pair with a job sheet; with neither, EVERY head runs on every pair:
  python predict.py /path/to/parquet_folder --dump-jobs jobs.csv   # edit, then:
  python predict.py /path/to/parquet_folder --jobs jobs.csv --out results/

  See docs/zero-shot.md for choosing a lab, and docs/fine-tuning.md for adapting a head
  to your own annotations (finetune.py) and running it here with --weights/--thresholds.

INPUT
  A folder of tracking parquets (or .pkt), long format with columns:
     video_frame, mouse_id, bodypart, x, y
  Bodypart NAMES must be from the model schema; the 7 the model can use are:
     tail_base, ear_right, ear_left, nose, neck, body_center, tail_tip
  (extra bodyparts are ignored; missing ones are treated as unobserved.)

METADATA (REQUIRED)  -- a missing pixel scale silently zeroes every prediction, so this is mandatory:
  Supply pix_per_cm and fps per file via a `metadata.csv` in the folder:
     file,pix_per_cm,fps
     *,16.0,30.0                 <- a '*' row sets the default for ALL files
     mouseA_day1.parquet,18.3,30 <- per-file rows override the default
  ...or pass one value for all files with --pix-per-cm and --fps.

SELECTING CLASSIFIERS (the job sheet)
  Columns: run,lab,action,subject,target
    run     1 to include the row, 0 to skip.
    subject/target  a mouse id (mouse1..mouse4), 'self' (self-directed), or '*' for all pairs.
  With no --jobs, every classifier is applied to every pair (the default).

OUTPUT  (per input parquet, in --out)
  <stem>.bouts.csv     compact ethogram: subject,target,lab,action,start_frame,stop_frame,mean_prob
  <stem>.frames.parquet  per-frame prob + call for every kept (subject,target,lab,action)
  (control with --output {both,calls,probs})
"""
import os, sys, argparse, glob, subprocess, pickle, hashlib, shutil

import scratch
os.environ.setdefault("SNIFFALL", "1")                       # register sniffall (id 37) before solution import
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")  # 6bp configs hang without this
os.environ.setdefault("PREDICT_BATCH", "64")                 # long-video OOM guard
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # don't grab the whole GPU
PKG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PKG)
import numpy as np, pandas as pd

ALL_LABS = ["AdaptableSnail", "BoisterousParrot", "CautiousGiraffe", "DeliriousFly", "ElegantMink",
            "GroovyShrew", "InvincibleJellyfish", "JovialSwallow", "LyricalHare", "NiftyGoldfinch",
            "PleasantMeerkat", "ReflectiveManatee", "SparklingTapir", "TranquilPanther", "UppityFerret"]
ALL_CONFIGS = ["11fps_4bp", "15fps_5bp", "19fps_6bp", "23fps_7bp", "27fps_6bp"]
LAB_ID = "userdata"          # arbitrary tracking-folder tag; must NOT be 'MABe22_movies'
DEFAULT_THR = 0.30
THR_CSV = os.path.join(PKG, "derived_thresholds_train.csv")


def _solution():
    import solution
    return solution


def action_names():
    s = _solution()
    return {i: s.ACTIONS.decode(i) for i in range(len(s.ACTIONS))}


def bodypart_schema():
    s = _solution()
    allbp = set(s.BODYPARTS.decode(i) for i in range(len(s.BODYPARTS)))
    canon7 = [s.BODYPARTS.decode(i) for i in range(7)]
    return allbp, canon7


def load_get_threshold(thr_csv=THR_CSV, const=None):
    """(lab, action) -> threshold. `const` overrides every head with one value."""
    if const is not None:
        return lambda lab, action: float(const)
    d = pd.read_csv(thr_csv)
    thr = {str(k): float(v) for k, v in zip(d[d.columns[0]], d[d.columns[1]])}
    return lambda lab, action: thr.get(f"{lab}__{action}", thr.get(f"pooled__{action}", DEFAULT_THR))


def enum_heads(thr_csv=THR_CSV):
    """(lab, action) for every classifier head, from the threshold file (includes sniffall)."""
    d = pd.read_csv(thr_csv)
    heads = []
    for k in d[d.columns[0]]:
        if "__" not in str(k):
            continue
        lab, act = str(k).split("__", 1)
        if lab != "pooled":
            heads.append((lab, act))
    return sorted(set(heads))


def list_heads(thr_csv=THR_CSV):
    """Print every available (lab, action) classifier head, grouped by lab."""
    heads = enum_heads(thr_csv)
    by_lab = {}
    for lab, act in heads:
        by_lab.setdefault(lab, []).append(act)
    print(f"{len(heads)} classifier heads across {len(by_lab)} labs:\n")
    for lab in sorted(by_lab):
        print(f"  {lab:<21} {' '.join(sorted(by_lab[lab]))}")
    print("\nUse --labs / --actions to select a subset, or --dump-jobs for a per-pair job sheet.")


def dump_jobs(path, thr_csv=THR_CSV):
    rows = [dict(run=1, lab=lab, action=act, subject="*", target="*") for lab, act in enum_heads(thr_csv)]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote job-sheet template with {len(rows)} classifier heads -> {path}")
    print("Edit it: set run=0 to skip a row; set subject/target to a mouse id (mouse1..), 'self', or '*' (all pairs).")


def jobs_from_filters(labs=None, actions=None, subject="*", target="*", thr_csv=THR_CSV):
    """Build the same {(lab, action): [(subject, target)]} filter parse_jobs() returns, from
    --labs/--actions/--subject/--target instead of a job sheet. Unknown names are an error."""
    heads = enum_heads(thr_csv)
    known_labs = sorted({l for l, _ in heads})
    known_acts = sorted({a for _, a in heads})
    if labs:
        bad = [l for l in labs if l not in known_labs]
        if bad:
            sys.exit(f"ERROR: unknown lab(s) {bad}. Available: {known_labs}")
    if actions:
        bad = [a for a in actions if a not in known_acts]
        if bad:
            sys.exit(f"ERROR: unknown action(s) {bad}. Available: {known_acts}")
    sel = [(l, a) for l, a in heads
           if (not labs or l in labs) and (not actions or a in actions)]
    if not sel:
        # the usual cause is a lab/action pair that was never trained together (e.g. the 5
        # sniff-splitting labs have 'sniffall' and no plain 'sniff') -- say who does have it.
        detail = ""
        for a in (actions or []):
            who = sorted(l for l, act in heads if act == a)
            detail += f"\n  '{a}' has heads for: {who}"
        for l in (labs or []):
            detail += f"\n  {l} has heads for: {sorted(act for lab_, act in heads if lab_ == l)}"
        sys.exit(f"ERROR: no classifier head matches labs={labs} actions={actions}.{detail}")
    return {k: [(subject, target)] for k in sel}


def parse_jobs(path):
    if not path:
        return None
    d = pd.read_csv(path)
    if "run" in d.columns:
        d = d[d["run"].fillna(1).astype(int) == 1]
    jobs = {}
    def _pat(r, field):
        """A blank field means "any pair". pandas reads a blank CSV cell as NaN, and NaN is TRUTHY,
        so a plain `or "*"` leaves it as the string "nan" -- which matches no subject, so the run
        silently produces nothing: every head executes, the process exits 0, an empty bouts.csv is
        written and no frames parquet at all. Indistinguishable from a behaviour that never
        occurred. Treat NaN, None and empty alike."""
        v = getattr(r, field, None)
        if v is None or (isinstance(v, float) and v != v) or str(v).strip() in ("", "nan"):
            return "*"
        return str(v).strip()

    for r in d.itertuples(index=False):
        subj = _pat(r, "subject")
        tgt = _pat(r, "target")
        jobs.setdefault((str(r.lab), str(r.action)), []).append((subj, tgt))
    return jobs


def discover(folder):
    fs = sorted(glob.glob(os.path.join(folder, "*.parquet")) + glob.glob(os.path.join(folder, "*.pkt")))
    return [f for f in fs if os.path.basename(f) not in ("index.csv", "metadata.csv")]


def load_metadata(folder, parquets, cli_pix, cli_fps):
    default = [cli_pix, cli_fps]
    per = {}
    mp = os.path.join(folder, "metadata.csv")
    if os.path.isfile(mp):
        md = pd.read_csv(mp)
        low = {c.lower(): c for c in md.columns}
        fcol = low.get("file") or low.get("filename") or md.columns[0]
        pcol = low.get("pix_per_cm") or low.get("pix_per_cm_approx")
        scol = low.get("fps") or low.get("frames_per_second")
        for r in md.itertuples(index=False):
            f = str(getattr(r, fcol))
            p = getattr(r, pcol) if pcol else np.nan
            s = getattr(r, scol) if scol else np.nan
            if f in ("*", "all", "ALL"):
                if not pd.isna(p): default[0] = float(p)
                if not pd.isna(s): default[1] = float(s)
            else:
                per[f] = (None if pd.isna(p) else float(p), None if pd.isna(s) else float(s))
    out = {}
    for pq in parquets:
        b = os.path.basename(pq)
        p, s = per.get(b, (None, None))
        p = p if p is not None else default[0]
        s = s if s is not None else default[1]
        if p is None:
            sys.exit(f"ERROR: no pix_per_cm for {b}. Add a metadata.csv (file,pix_per_cm,fps) or pass --pix-per-cm. "
                     "A missing pixel scale makes ALL predictions zero.")
        if s is None:
            sys.exit(f"ERROR: no fps for {b}. Add it to metadata.csv or pass --fps.")
        out[pq] = (float(p), float(s))
    return out


def vid_of(path):
    return int(hashlib.md5(os.path.basename(path).encode()).hexdigest()[:12], 16) % 2_000_000_000


def runs(mask):
    m = np.asarray(mask, bool)
    d = np.diff(m.astype(np.int8))
    st = list(np.where(d == 1)[0] + 1); en = list(np.where(d == -1)[0] + 1)
    if m.size and m[0]: st = [0] + st
    if m.size and m[-1]: en = en + [len(m)]
    return list(zip(st, en))


def run(folder, jobs=None, labs=None, actions=None, subject="*", target="*",
        out="doom_predictions", pix_per_cm=None, fps=None, gpu="0", output="both",
        keep_work=False, weights=None, thresholds=THR_CSV, threshold=None, configs=None):
    """Run the ensemble over a folder of tracking parquets and write bouts/frames into `out`.

    jobs      job-sheet path, or the dict parse_jobs() returns, or None for every head
    labs      restrict to these classifier labs (alternative to a job sheet)
    actions   restrict to these behaviours
    subject   acting mouse for the filtered heads: 'mouse1'..'mouse4', 'self', or '*'
    target    recipient mouse, same values
    weights   per-lab checkpoint path template with a '{config}' placeholder, for
              fine-tuned weights (default: models/{config}_supervised_perlab_sniffall.pkl)
    thresholds  CSV of '<lab>__<action>,threshold' rows (default derived_thresholds_train.csv)
    threshold   one constant threshold for every head, overriding `thresholds`
    configs   run only these ensemble configs (default: all 5) -- faster, lower quality

    Returns the list of written output stems.
    """
    if isinstance(jobs, str):
        jobs = parse_jobs(jobs)
    if jobs is None and (labs or actions or subject != "*" or target != "*"):
        jobs = jobs_from_filters(labs, actions, subject, target, thresholds)
    if weights and "{config}" not in weights:
        sys.exit("ERROR: --weights must contain a '{config}' placeholder, e.g. "
                 "ft_models/{config}__myrig.pkl")
    if configs:
        bad = [c for c in configs if c not in ALL_CONFIGS]
        if bad:
            sys.exit(f"ERROR: unknown config(s) {bad}; available: {ALL_CONFIGS}")

    from download_models import require_weights
    require_weights()          # weights are hosted on Hugging Face, not in git

    parquets = discover(folder)
    if not parquets:
        sys.exit(f"no .parquet/.pkt files in {folder}")
    meta = load_metadata(folder, parquets, pix_per_cm, fps)
    run_labs = sorted({lab for (lab, _) in jobs}) if jobs else ALL_LABS
    run_labs = [l for l in run_labs if l in ALL_LABS]
    print(f"{len(parquets)} parquet(s); running {len(run_labs)} lab classifier set(s): {run_labs}")

    allbp, canon7 = bodypart_schema()
    os.makedirs(out, exist_ok=True)
    ds = os.path.abspath(os.path.join(out, "_work"))
    tdir = os.path.join(ds, "custom_tracking", LAB_ID)
    outstore = os.path.join(ds, "perlab_allbeh")
    os.makedirs(tdir, exist_ok=True); os.makedirs(outstore, exist_ok=True)

    rows = []
    vmap = {}
    for pq in parquets:
        vid = vid_of(pq); vmap[vid] = pq
        bps = set(pd.read_parquet(pq, columns=["bodypart"]).bodypart.unique())
        bad = bps - allbp
        if bad:
            sys.exit(f"ERROR: {os.path.basename(pq)} contains bodypart names not in the model schema: {sorted(bad)}\n"
                     f"Rename them to schema names (usable: {canon7}).")
        miss = [b for b in canon7 if b not in bps]
        if miss:
            print(f"  note: {os.path.basename(pq)} is missing {miss} -> configs needing them use fewer keypoints (still runs)")
        shutil.copyfile(pq, os.path.join(tdir, f"{vid}.parquet"))
        p, s = meta[pq]
        rows.append(dict(lab_id=LAB_ID, video_id=vid, frames_per_second=s, pix_per_cm_approx=p, behaviors_labeled="[]"))
    pd.DataFrame(rows).to_csv(os.path.join(ds, "manifest.csv"), index=False)

    env = dict(os.environ, SNIFFALL="1", PREDICT_BATCH="64", XLA_FLAGS="--xla_gpu_autotune_level=0",
               XLA_PYTHON_CLIENT_PREALLOCATE="false",
               CUDA_VISIBLE_DEVICES=str(gpu), CUSTOM_DIR=ds, CUSTOM_CSV="manifest.csv",
               CUSTOM_MODE="custom", CUSTOM_OUT=outstore,
               PERLAB_WORKDIR=scratch.path(f"doom_predict_{os.getpid()}"))
    if weights:
        env["HIDRA_PERLAB_CKPT"] = os.path.abspath(weights)   # the subprocess runs with cwd=PKG
        print(f"using fine-tuned per-lab weights: {weights}")
    if configs:
        env["HIDRA_CONFIGS"] = ",".join(configs)
        print(f"WARNING: running {len(configs)}/5 ensemble configs ({configs}) -- faster but "
              "lower quality than the full ensemble, and the bundled thresholds were calibrated "
              "on all 5. For iterating only.")
    shutil.rmtree(env["PERLAB_WORKDIR"], ignore_errors=True)
    for i, lab in enumerate(run_labs, 1):
        print(f"[{i}/{len(run_labs)}] inferring {lab} ...", flush=True)
        r = subprocess.run([sys.executable, os.path.join(PKG, "run_allbehaviors_perlab.py"),
                            "--dataset", "custom", "--embedding-lab", lab, "--epochs", "1"], env=env, cwd=PKG)
        if r.returncode != 0:
            print(f"  WARNING: {lab} inference exited {r.returncode}; skipping its outputs")
    shutil.rmtree(env["PERLAB_WORKDIR"], ignore_errors=True)

    getthr = load_get_threshold(thresholds, threshold); anames = action_names()

    def keep(lab, act, subj, tgt):
        if jobs is None:
            return True
        filt = jobs.get((lab, act))
        return filt is not None and any((fs in ("*", subj)) and (ft in ("*", tgt)) for fs, ft in filt)

    stores = {lab: pickle.load(open(os.path.join(outstore, f"{lab}.pkl"), "rb"))
              for lab in run_labs if os.path.isfile(os.path.join(outstore, f"{lab}.pkl"))}
    written = []
    for pq in parquets:
        vid = vid_of(pq); stem = os.path.splitext(os.path.basename(pq))[0]
        bouts, frames = [], []
        for lab, P in stores.items():
            for (v, ae, tn, aid), arr in P.items():
                if v != vid:
                    continue
                act = anames.get(aid, str(aid)); subj = f"mouse{ae + 1}"; tgt = ("self" if ae == tn else f"mouse{tn + 1}")
                if not keep(lab, act, subj, tgt):
                    continue
                pr = np.asarray(arr, np.float32); thr = getthr(lab, act); call = pr >= thr
                for s, e in runs(call):
                    bouts.append(dict(subject=subj, target=tgt, lab=lab, action=act,
                                      start_frame=int(s), stop_frame=int(e - 1),
                                      n_frames=int(e - s), mean_prob=round(float(pr[s:e].mean()), 4), threshold=round(thr, 4)))
                if output in ("both", "probs"):
                    idx = np.arange(len(pr))
                    frames.append(pd.DataFrame(dict(frame=idx, subject=subj, target=tgt, lab=lab, action=act,
                                                    prob=pr, call=call.astype(np.int8))))
        if output in ("both", "calls"):
            pd.DataFrame(bouts).to_csv(os.path.join(out, f"{stem}.bouts.csv"), index=False)
        if output in ("both", "probs") and frames:
            pd.concat(frames, ignore_index=True).to_parquet(os.path.join(out, f"{stem}.frames.parquet"))
        written.append(stem)
        print(f"  {stem}: {len(bouts)} bouts -> {out}/{stem}.*")

    if not keep_work:
        shutil.rmtree(ds, ignore_errors=True)
    print(f"done: {len(written)} parquet(s) -> {out}/")
    return written


def main():
    ap = argparse.ArgumentParser(prog="HiDRA",
                                 description="HiDRA (High-Dimensional Rodent Annotator): apply the per-lab-head behaviour classifier ensemble to pose parquets.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("folder", nargs="?", help="folder of tracking parquets (.parquet/.pkt)")
    ap.add_argument("--jobs", help="CSV job sheet selecting lab/action/subject/target (default: all)")
    ap.add_argument("--dump-jobs", metavar="FILE", help="write an editable job-sheet template and exit")
    ap.add_argument("--list-heads", action="store_true", help="print every (lab, action) classifier head and exit")
    ap.add_argument("--labs", help="comma-separated classifier labs to run (quick alternative to --jobs)")
    ap.add_argument("--actions", help="comma-separated behaviours to run (quick alternative to --jobs)")
    ap.add_argument("--subject", default="*", help="acting mouse for --labs/--actions: mouse1..mouse4, self, or * (default *)")
    ap.add_argument("--target", default="*", help="recipient mouse for --labs/--actions: mouse1..mouse4, self, or * (default *)")
    ap.add_argument("--out", default="doom_predictions", help="output folder (default: doom_predictions)")
    ap.add_argument("--pix-per-cm", type=float, help="pixels-per-cm for ALL files (overridden by metadata.csv rows)")
    ap.add_argument("--fps", type=float, help="frames-per-second for ALL files")
    ap.add_argument("--gpu", default="0", help="CUDA device index (default 0)")
    ap.add_argument("--output", choices=["both", "calls", "probs"], default="both",
                    help="what to write: both (default), calls-only (bouts.csv), or probs-only (frames.parquet)")
    ap.add_argument("--weights", metavar="TEMPLATE",
                    help="fine-tuned per-lab checkpoints, e.g. 'finetuned/{config}__MyLab.pkl' "
                         "('{config}' is filled with each of the 5 config names)")
    ap.add_argument("--thresholds", default=THR_CSV, metavar="FILE",
                    help="thresholds CSV ('<lab>__<action>,threshold' rows); default derived_thresholds_train.csv")
    ap.add_argument("--threshold", type=float, help="one constant threshold for every head (overrides --thresholds)")
    ap.add_argument("--configs", help=f"run only these ensemble configs (default all 5: "
                                      f"{','.join(ALL_CONFIGS)}); faster, lower quality, for iterating")
    ap.add_argument("--keep-work", action="store_true", help="keep the scratch inference dir")
    args = ap.parse_args()

    if args.list_heads:
        list_heads(args.thresholds); return
    if args.dump_jobs:
        dump_jobs(args.dump_jobs, args.thresholds); return
    if not args.folder:
        ap.error("provide a parquet folder (or use --dump-jobs FILE / --list-heads)")
    if args.jobs and (args.labs or args.actions):
        ap.error("--jobs and --labs/--actions are alternatives; use one or the other")

    run(args.folder, jobs=args.jobs,
        labs=[x.strip() for x in args.labs.split(",") if x.strip()] if args.labs else None,
        actions=[x.strip() for x in args.actions.split(",") if x.strip()] if args.actions else None,
        subject=args.subject, target=args.target, out=args.out, pix_per_cm=args.pix_per_cm,
        fps=args.fps, gpu=args.gpu, output=args.output, keep_work=args.keep_work,
        weights=args.weights, thresholds=args.thresholds, threshold=args.threshold,
        configs=[c.strip() for c in args.configs.split(",") if c.strip()] if args.configs else None)


if __name__ == "__main__":
    main()

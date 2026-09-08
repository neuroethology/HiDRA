#!/usr/bin/env python3
"""
HiDRA -- the High-Dimensional Rodent Annotator.
Apply the per-lab-head mouse-behaviour classifier ensemble to a folder of pose tracking parquets.

QUICK START
  # 0. one-time: fetch the model weights (~660 MB) from the Hugging Face Hub into models/
  pip install huggingface_hub && python download_models.py
  # 1. (recommended) generate a job sheet listing every available classifier, then edit it:
  python predict.py /path/to/parquet_folder --dump-jobs jobs.csv
  # 2. run it (default with no --jobs = ALL lab classifiers x ALL actions x ALL mouse pairs):
  python predict.py /path/to/parquet_folder --jobs jobs.csv --out results/

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


def load_get_threshold():
    d = pd.read_csv(THR_CSV)
    thr = {str(k): float(v) for k, v in zip(d[d.columns[0]], d[d.columns[1]])}
    return lambda lab, action: thr.get(f"{lab}__{action}", thr.get(f"pooled__{action}", DEFAULT_THR))


def enum_heads():
    """(lab, action) for every classifier head, from the threshold file (includes sniffall)."""
    d = pd.read_csv(THR_CSV)
    heads = []
    for k in d[d.columns[0]]:
        if "__" not in str(k):
            continue
        lab, act = str(k).split("__", 1)
        if lab != "pooled":
            heads.append((lab, act))
    return sorted(set(heads))


def dump_jobs(path):
    rows = [dict(run=1, lab=lab, action=act, subject="*", target="*") for lab, act in enum_heads()]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote job-sheet template with {len(rows)} classifier heads -> {path}")
    print("Edit it: set run=0 to skip a row; set subject/target to a mouse id (mouse1..), 'self', or '*' (all pairs).")


def parse_jobs(path):
    if not path:
        return None
    d = pd.read_csv(path)
    if "run" in d.columns:
        d = d[d["run"].fillna(1).astype(int) == 1]
    jobs = {}
    for r in d.itertuples(index=False):
        subj = str(getattr(r, "subject", "*") or "*")
        tgt = str(getattr(r, "target", "*") or "*")
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


def main():
    ap = argparse.ArgumentParser(prog="HiDRA",
                                 description="HiDRA (High-Dimensional Rodent Annotator): apply the per-lab-head behaviour classifier ensemble to pose parquets.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("folder", nargs="?", help="folder of tracking parquets (.parquet/.pkt)")
    ap.add_argument("--jobs", help="CSV job sheet selecting lab/action/subject/target (default: all)")
    ap.add_argument("--dump-jobs", metavar="FILE", help="write an editable job-sheet template and exit")
    ap.add_argument("--out", default="doom_predictions", help="output folder (default: doom_predictions)")
    ap.add_argument("--pix-per-cm", type=float, help="pixels-per-cm for ALL files (overridden by metadata.csv rows)")
    ap.add_argument("--fps", type=float, help="frames-per-second for ALL files")
    ap.add_argument("--gpu", default="0", help="CUDA device index (default 0)")
    ap.add_argument("--output", choices=["both", "calls", "probs"], default="both",
                    help="what to write: both (default), calls-only (bouts.csv), or probs-only (frames.parquet)")
    ap.add_argument("--keep-work", action="store_true", help="keep the scratch inference dir")
    args = ap.parse_args()

    if args.dump_jobs:
        dump_jobs(args.dump_jobs); return
    if not args.folder:
        ap.error("provide a parquet folder (or use --dump-jobs FILE)")

    from download_models import require_weights
    require_weights()          # weights are hosted on Hugging Face, not in git

    parquets = discover(args.folder)
    if not parquets:
        sys.exit(f"no .parquet/.pkt files in {args.folder}")
    meta = load_metadata(args.folder, parquets, args.pix_per_cm, args.fps)
    jobs = parse_jobs(args.jobs)
    labs = sorted({lab for (lab, _) in jobs}) if jobs else ALL_LABS
    labs = [l for l in labs if l in ALL_LABS]
    print(f"{len(parquets)} parquet(s); running {len(labs)} lab classifier set(s): {labs}")

    allbp, canon7 = bodypart_schema()
    os.makedirs(args.out, exist_ok=True)
    ds = os.path.abspath(os.path.join(args.out, "_work"))
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
               CUDA_VISIBLE_DEVICES=str(args.gpu), CUSTOM_DIR=ds, CUSTOM_CSV="manifest.csv",
               CUSTOM_MODE="custom", CUSTOM_OUT=outstore, PERLAB_WORKDIR=f"/dev/shm/doom_predict_{os.getpid()}")
    shutil.rmtree(env["PERLAB_WORKDIR"], ignore_errors=True)
    for i, lab in enumerate(labs, 1):
        print(f"[{i}/{len(labs)}] inferring {lab} ...", flush=True)
        r = subprocess.run([sys.executable, os.path.join(PKG, "run_allbehaviors_perlab.py"),
                            "--dataset", "custom", "--embedding-lab", lab, "--epochs", "1"], env=env, cwd=PKG)
        if r.returncode != 0:
            print(f"  WARNING: {lab} inference exited {r.returncode}; skipping its outputs")
    shutil.rmtree(env["PERLAB_WORKDIR"], ignore_errors=True)

    getthr = load_get_threshold(); anames = action_names()

    def keep(lab, act, subj, tgt):
        if jobs is None:
            return True
        filt = jobs.get((lab, act))
        return filt is not None and any((fs in ("*", subj)) and (ft in ("*", tgt)) for fs, ft in filt)

    stores = {lab: pickle.load(open(os.path.join(outstore, f"{lab}.pkl"), "rb"))
              for lab in labs if os.path.isfile(os.path.join(outstore, f"{lab}.pkl"))}
    n_out = 0
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
                if args.output in ("both", "probs"):
                    idx = np.arange(len(pr))
                    frames.append(pd.DataFrame(dict(frame=idx, subject=subj, target=tgt, lab=lab, action=act,
                                                    prob=pr, call=call.astype(np.int8))))
        if args.output in ("both", "calls"):
            pd.DataFrame(bouts).to_csv(os.path.join(args.out, f"{stem}.bouts.csv"), index=False)
        if args.output in ("both", "probs") and frames:
            pd.concat(frames, ignore_index=True).to_parquet(os.path.join(args.out, f"{stem}.frames.parquet"))
        n_out += 1
        print(f"  {stem}: {len(bouts)} bouts -> {args.out}/{stem}.*")

    if not args.keep_work:
        shutil.rmtree(ds, ignore_errors=True)
    print(f"done: {n_out} parquet(s) -> {args.out}/")


if __name__ == "__main__":
    main()

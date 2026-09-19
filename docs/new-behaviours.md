# Adding a behaviour

"Behaviour" in HiDRA means a **(lab, behaviour) head** — one column of the per-lab classifier
head, trained on one lab's definition of one behaviour. There are 82 of them across 15 labs, and
the names they use come from a fixed vocabulary of 37 actions plus the merged `sniffall`. Whether
you can get a classifier for *your* behaviour, and how much it costs, depends on which of the
cases below you are in. Work down the list; the first one that fits is the cheapest.

```bash
python predict.py --list-heads                 # every (lab, behaviour) head
python -c "import hidra; print(hidra.heads(action='mount'))"   # who has a given behaviour
```

## 1. A published head already matches → zero-shot

Several labs usually have "your" behaviour, defined slightly differently. Run each on a pilot
recording and pick the one whose calls match what you would have annotated
([zero-shot.md §2](zero-shot.md#2-pick-a-classifier)). If the calls are right but too many or too
few, fix the threshold first ([zero-shot.md §5](zero-shot.md#5-thresholds)) — `calibrate` needs a
few labelled videos and no GPU.

## 2. The head exists but is systematically off → fine-tune it

[fine-tuning.md](fine-tuning.md). In `head` mode this is what the research code calls the
*linear-only add of a behaviour on top of a lab's trained tail*: the tail that already knows
mouse behaviour is kept, and only the column for your behaviour is refit to your annotations.
Nothing else in the checkpoint moves, so the lab's other heads remain usable.

## 3. Your lab has no column for it, but another lab does → adopt the other lab

The head table is sparse: GroovyShrew has `rear` and `sniffgenital` but no `mount`; ElegantMink
has `mount`. The cheapest fix is to adopt ElegantMink for that behaviour and fine-tune *its* head on
your data (§4 is the alternative when you want the column under your own lab):

```bash
python finetune.py prepare --tracking parquets/ --annotations bouts.csv --lab ElegantMink --out ft_mount/
python finetune.py train --data ft_mount/ --lab ElegantMink --actions mount --out ft_models/ --tag mount
```

One fine-tune per adopted lab; predict each with its own `--labs`/`--weights`, and keep `--mode
head` if you want each checkpoint to stay valid for its lab's other behaviours. `prepare` tells
you which labs have a behaviour when the one you named does not.

## 4. The name is in the vocabulary, but your lab has no column for it → add a column

Any (lab, behaviour) pair not among the 82 — including the four names no lab has a head for at all
(`dominancemount`, `disengage`, `genitalgroom`; `none` is a placeholder) — can be given a column
with `finetune.py train --new-head Lab,action`. It is the same warm-start that added `sniffall`
to the published model, made general: the published columns are copied into a one-wider head,
the new column is trained on your annotations on top of the lab's frozen, already-trained
features, and nothing else moves. PyTorch backend, one GPU, `head` mode:

```bash
python finetune.py prepare --tracking parquets/ --annotations bouts.csv --lab GroovyShrew \
    --new-head GroovyShrew,attack --out ft_attack/ --pix-per-cm 16 --fps 30
python finetune.py train --data ft_attack/ --lab GroovyShrew --new-head GroovyShrew,attack \
    --seed-from LyricalHare,attack --out ft_models/ --tag attack
python predict.py held_out/ --labs GroovyShrew --actions attack --out ft_preds/ \
    --weights 'ft_models/{config}__attack.pkl' --pix-per-cm 16 --fps 30
python finetune.py calibrate --frames ft_preds/ --annotations bouts.csv --out ft_thresholds.csv
python predict.py videos/ --labs GroovyShrew --actions attack --out results/ \
    --weights 'ft_models/{config}__attack.pkl' --thresholds ft_thresholds.csv --pix-per-cm 16 --fps 30
```

- **`prepare --new-head`** admits bouts labelled with the new action, which it would otherwise
  reject as "no head for". Annotate it exhaustively in every staged video, like any other
  behaviour ([fine-tuning.md §1](fine-tuning.md#1-annotate)); for a social behaviour annotate
  both directed pairs.
- **`train --new-head`** builds, per config, an 83-column `out-proj-perlab`: the 82 published
  columns (weights `w`, gain `s`, bias `b`) are copied by name into their positions in the extended
  table, bit for bit; the layer's input-normalization statistics are per input feature and copy
  unchanged. The new column starts from **`--seed-from Lab,action`** — an existing head of a
  similar behaviour, typically another lab's version of the same one — or, without it, from a
  fresh init. Seeding is what made the sniffall heads converge: a linear readout on frozen
  features trained from scratch tends to plateau, while a related head's direction is already
  most of the way there. Supervision is masked to the new column (add other heads of the lab
  with `--actions new,existing` if you annotated them too), and only `--mode head` is allowed.
- **The checkpoint carries its column list**: `ft_models/{config}__attack.heads.json` next to
  each `.pkl`, and `hidra-convert-weights --one` writes the same list into the safetensors
  metadata. Both backends' loaders read it before falling back to the published table, so
  `predict.py --weights` runs the head without any edit to `models/thresholds.json` — and a
  widened checkpoint separated from its sidecar refuses to load rather than mis-route columns.
- **Seeing the head**: `predict.py --list-heads --weights 'ft_models/{config}__attack.pkl'`
  lists it (marked as declared by the weights), `--actions attack` is accepted whenever those
  weights are given, and `finetune.py calibrate` writes its `GroovyShrew__attack` row into the
  thresholds CSV like any other head. Until then the new head uses the fallback threshold
  (`pooled__attack`, else 0.30), so calibrate before drawing conclusions.
- **Several at once.** `--new-head` is repeatable: `--new-head GroovyShrew,attack --new-head
  GroovyShrew,mount` widens the head by both columns in one run and supervises both.
- **What it cannot do.** The action must be one of the vocabulary names (§5 for a new *name*),
  and the pair must not already exist (fine-tune it instead, §2). In `head` mode the tail is not
  retrained, so the column can only express what the lab's frozen features already separate; if
  that is not enough, use `--mode tail` (§4b) or the full route below. As with any fine-tune the
  ensemble is five configs, so predict only once all five are trained (`--configs` for
  iterating), and pass `--labs <lab>` when predicting with the result.

## 4b. You want the columns under *your* lab, not someone else's → adopt a head-free slot

The lab embedding has 21 rows; only 15 carry published head columns. Six more were in the
consortium's training data but have no scored heads, and five of those can be claimed: naming
one as `--lab` and giving it columns costs nothing published and makes it, in effect, your lab.

```bash
python predict.py --list-heads          # the 15 labs with heads
python -c "from hidra.head_table import head_free_slots; print(head_free_slots())"
```

```
['CRIM13', 'CalMS21_supplemental', 'CalMS21_task1', 'CalMS21_task2', 'MABe22_keypoints']
```

(`MABe22_movies` is the sixth row and is excluded: the data loader permutes that lab's
bodyparts, because its published keypoints were scrambled, so your pose would be scrambled too.)

```bash
python finetune.py prepare --tracking parquets/ --annotations bouts.csv --lab MABe22_keypoints \
    --new-head MABe22_keypoints,attack --new-head MABe22_keypoints,rear \
    --out ft_data/ --pix-per-cm 16 --fps 30
python finetune.py train --data ft_data/ --lab MABe22_keypoints \
    --new-head MABe22_keypoints,attack --new-head MABe22_keypoints,rear \
    --seed-from LyricalHare --mode tail --cache-features \
    --out ft_models/ --tag mylab
python predict.py held_out/ --labs MABe22_keypoints --actions attack,rear --out ft_preds/ \
    --weights 'ft_models/{config}__mylab.pkl' --pix-per-cm 16 --fps 30
```

Two things differ from §4:

- **`--seed-from <DonorLab>`** — a bare lab name rather than a `Lab,action` pair. Each new
  column starts from that lab's column of the *same action* (here LyricalHare's `attack` and
  `rear`), and the lab's **embedding row** is seeded from the donor's too. A head-free slot's
  embedding row was only ever trained through the shared 38-way head on that consortium dataset,
  so starting it at a lab whose recordings resemble yours puts your data where the tail's
  features already make sense. A column whose action the donor lacks falls back to a fresh init,
  and the run says so.
- **`--mode tail`** is the point of doing it this way. With your own lab row you can retrain the
  whole per-lab tail without making any *published* lab's head meaningless — the checkpoint
  speaks for your lab, which is what `--labs` selects anyway. This is the "new-lab adaptation"
  the leave-one-lab-out experiments used. `--mode head` still works and is cheaper; §3b of
  [fine-tuning.md](fine-tuning.md) has the trade.

The checkpoint carries its own column list, so `predict.py --weights` runs the slot as a lab like
any other and `--list-heads --weights` shows its behaviours. Without those weights the slot has
no head and `predict.py --labs <slot>` refuses, which is the intended behaviour: nothing published
claims to classify anything for it.

The full alternative is **stage-2 training** with an extended table
([training.md §3](training.md#3-stage-2-the-supervised-per-lab-tail-and-heads)): add a
`[lab, action, 0.3]` row to `thresholds.json` in a fresh `HIDRA_MODELS_DIR` and
`train_perlab_heads.py` builds the wider head from scratch, tail included, on the consortium data
plus yours. JAX-only, 50k steps per config, and inference then needs that `thresholds.json` next
to the checkpoints. Reach for it when the new column needs features the frozen tail does not
provide.

## 5. The name is not in the vocabulary at all

Adding a name to `schema.ACTIONS` changes the width of the shared 38-way head and the label space
every checkpoint was trained in, so it is a schema change followed by the full stage-2 retrain of
§4 — and the new name would exist only in checkpoints trained with it.

The practical alternative is to **repurpose a column**: a head column learns whatever its label
marks, and the name is just a key. Adopt a lab whose head for some vocabulary name you do not
otherwise need — `flinch`, `tussle`, `shepherd`, … — annotate your behaviour under that name, and
fine-tune that column in `head` mode (§2). Record the mapping in your own notes and in the
`--tag`, and never mix such a checkpoint with the published thresholds for that head: the column
no longer means what the lab meant by it.

## Where the head table lives

| what | defined by | used by |
|---|---|---|
| the checkpoint's columns (which (lab, action) has a head, and in what order) | keys of `models/thresholds.json` → `schema.lab_action_table()` (`train_perlab_heads.LAB_ACTION` on the JAX side); a checkpoint widened by `--new-head` carries its own list in `{config}__{tag}.heads.json` / safetensors metadata (`hidra.head_table`) | building the head at train and inference time; must match the checkpoint's width |
| which heads the CLI lists, accepts in `--actions`, and thresholds | `Lab__action,threshold` rows of `derived_thresholds_train.csv` (or the CSV given to `--thresholds`) | `predict.py`, `finetune.py prepare/train` validation, `hidra.heads()` |
| the `sniffall` label | synthesized as the OR of `sniff, sniffface, sniffbody, sniffgenital, reciprocalsniff` | the trainer's loss; `prepare` stages `sniffall` rows as `sniff` accordingly |
| the label space | `schema.ACTIONS` (37 names; `sniffall` appended when `SNIFFALL=1`) and `schema.LABS` (21 labs) | every checkpoint |

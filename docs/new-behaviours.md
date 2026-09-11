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
has `mount`. There is no way to give GroovyShrew a `mount` column through `finetune.py`, so adopt
ElegantMink for that behaviour and fine-tune *its* head on your data:

```bash
python finetune.py prepare --tracking parquets/ --annotations bouts.csv --lab ElegantMink --out ft_mount/
python finetune.py train --data ft_mount/ --lab ElegantMink --actions mount --out ft_models/ --tag mount
```

One fine-tune per adopted lab; predict each with its own `--labs`/`--weights`, and keep `--mode
head` if you want each checkpoint to stay valid for its lab's other behaviours. `prepare` tells
you which labs have a behaviour when the one you named does not.

## 4. The name is in the vocabulary, but no lab has a head for it → a new column

Four names have no head at all (`dominancemount`, `disengage`, `genitalgroom`, and the `none`
placeholder), and any (lab, behaviour) pair not in the 82 is in the same position. A new column
means **stage-2 training** ([training.md §3](training.md#3-stage-2-the-supervised-per-lab-tail-and-heads)),
because the column set is fixed when the per-lab head is built:

- The columns are the keys of `thresholds.json` in the models directory (one `[lab, action,
  threshold]` row per column; the threshold *values* are not used for training), minus
  PleasantMeerkat's attack/chase/escape, plus `sniffall` for the five sniff-splitting labs. Add a
  row, and `train_perlab_heads.py` builds an 83-column head.
- The action must be one of the vocabulary names in `schema.ACTIONS`; the lab must be one of the
  21 in `schema.LABS`.
- Annotate the behaviour under that name and stage it with `finetune.py prepare --thresholds` a
  CSV that lists the new head (`Lab__action,0.3`), since `prepare` validates labels against the
  head list it is given.
- Train the per-lab foundation from scratch with the extended table, on the consortium data plus
  yours. This is the expensive route: JAX-only, all of the merge/tail/heads relearn (the
  self-supervised trunk stays frozen), 50k steps per config.
- Inference then needs the same `thresholds.json` next to the checkpoints (`HIDRA_MODELS_DIR`) so
  the head table matches the checkpoint's width, and a thresholds CSV carrying the new
  `Lab__action` row so `predict.py --list-heads`/`--actions` know the head exists. Calibrate its
  threshold with `finetune.py calibrate` like any other.

The research code also contains the cheaper alternative that was used to add `sniffall` itself:
warm-start from the published checkpoint, copy the 82 existing columns into their positions in
the wider table, seed the new column from a related one, and train only the new column on frozen
features (`FROZEN_TRUNK` in `train_perlab_heads.py`). That remap is hard-wired to the sniffall
columns today — generalising it to an arbitrary new (lab, behaviour) is a code change, not a
flag, and `finetune.py` does not expose it.

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
| the checkpoint's columns (which (lab, action) has a head, and in what order) | keys of `models/thresholds.json` → `schema.lab_action_table()` (`train_perlab_heads.LAB_ACTION` on the JAX side) | building the head at train and inference time; must match the checkpoint's width |
| which heads the CLI lists, accepts in `--actions`, and thresholds | `Lab__action,threshold` rows of `derived_thresholds_train.csv` (or the CSV given to `--thresholds`) | `predict.py`, `finetune.py prepare/train` validation, `hidra.heads()` |
| the `sniffall` label | synthesized as the OR of `sniff, sniffface, sniffbody, sniffgenital, reciprocalsniff` | the trainer's loss; `prepare` stages `sniffall` rows as `sniff` accordingly |
| the label space | `schema.ACTIONS` (37 names; `sniffall` appended when `SNIFFALL=1`) and `schema.LABS` (21 labs) | every checkpoint |

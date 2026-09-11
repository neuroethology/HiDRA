"""`finetune.py`'s annotation handling -- pure pandas, no weights, no GPU.

The trap these guard against: the model has no `sniffall` label channel. The trainer
synthesizes it as the OR of the sniff-family channels (`MultiTaskPerLabModel._labels37`
and its torch port), overwriting label and mask alike, so a bout literally labelled
`sniffall` supervises nothing and the run logs a zero-weight loss. `prepare` therefore
stages such rows as plain `sniff`, and `train` resolves which head columns a set of
staged labels can actually supervise.
"""
import pandas as pd
import pytest

from hidra import finetune as ft

SNIFFALL_LAB = "GroovyShrew"      # sniff head is the merged `sniffall`
PLAIN_LAB = "LyricalHare"         # sniff head is plain `sniff`


def _annot(*rows):
    return pd.DataFrame(
        [dict(file="v.parquet", stem="v", agent="mouse1", target=t, action=a,
              start_frame=0, stop_frame=10) for a, t in rows])


def test_sniffall_rows_are_staged_as_sniff():
    annot = _annot(("rear", "self"), ("sniffall", "mouse2"))
    out = ft.check_actions(annot, SNIFFALL_LAB)
    assert sorted(out["action"].unique()) == ["rear", "sniff"]
    assert len(out) == 2, "no rows may be dropped by the translation"


def test_sniff_family_labels_are_accepted_for_a_sniffall_lab():
    for label in ft.SNIFF_FAMILY:
        out = ft.check_actions(_annot((label, "mouse2")), SNIFFALL_LAB)
        assert list(out["action"]) == [label]


def test_sniffall_is_rejected_for_a_plain_sniff_lab_and_names_who_has_it():
    with pytest.raises(SystemExit) as exc:
        ft.check_actions(_annot(("sniffall", "mouse2")), PLAIN_LAB)
    msg = str(exc.value)
    assert "no head for ['sniffall']" in msg
    assert SNIFFALL_LAB in msg, "the error should list the labs that do have the head"


def test_unknown_action_is_still_rejected():
    with pytest.raises(SystemExit):
        ft.check_actions(_annot(("tailrattle", "mouse2")), SNIFFALL_LAB)


def test_trainable_heads_maps_sniff_to_sniffall_only_for_sniffall_labs():
    hb = ft.heads_by_lab()
    assert ft.trainable_heads(SNIFFALL_LAB, {"rear", "sniff"}, hb) == ["rear", "sniffall"]
    # A subtype-only annotation set must NOT quietly redefine the merged head as that subtype.
    assert ft.trainable_heads(SNIFFALL_LAB, {"sniffgenital"}, hb) == ["sniffgenital"]
    # For a plain-sniff lab, `sniff` is its own head and there is no sniffall to add.
    assert ft.trainable_heads(PLAIN_LAB, {"sniff", "attack"}, hb) == ["attack", "sniff"]
    # Labels that are not heads of the lab train nothing.
    assert ft.trainable_heads(PLAIN_LAB, {"dig"}, hb) == []


def test_supervisable_follows_the_label_synthesis():
    assert ft.supervisable("rear", {"rear"})
    assert not ft.supervisable("rear", {"sniff"})
    # sniffall is fed by any family member -- an explicit --actions sniffall is allowed then...
    assert ft.supervisable("sniffall", {"sniffgenital"})
    assert ft.supervisable("sniffall", {"sniff"})
    # ...but a manifest carrying the literal `sniffall` (staged by an older prepare) gives it nothing.
    assert not ft.supervisable("sniffall", {"sniffall", "rear"})


def test_read_annotations_stop_inclusive(tmp_path):
    csv = tmp_path / "bouts.csv"
    pd.DataFrame([dict(file="a.parquet", agent="1", target="self", action="rear",
                       start_frame=10, stop_frame=20)]).to_csv(csv, index=False)
    excl = ft.read_annotations(csv)
    incl = ft.read_annotations(csv, stop_inclusive=True)
    assert excl.loc[0, "stop_frame"] == 20 and incl.loc[0, "stop_frame"] == 21
    assert excl.loc[0, "agent"] == "mouse1" and excl.loc[0, "stem"] == "a"

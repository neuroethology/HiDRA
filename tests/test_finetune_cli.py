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


def test_prepare_never_suggests_an_empty_actions_list():
    """Every label set `check_actions` accepts must yield a runnable `--actions`.

    `annotatable_actions` lets a sniff-splitting lab's CSV carry any sniff-family name, but
    `trainable_heads` maps only plain `sniff` onto the merged column -- so a `sniffface`-only
    set used to print `--actions  --out ft_models/`, which argparse rejects outright.
    """
    hb = ft.heads_by_lab()
    for lab in (SNIFFALL_LAB, PLAIN_LAB):
        for label in sorted(ft.annotatable_actions(lab, hb)):
            staged = set(ft.check_actions(_annot((label, "mouse2")), lab)["action"])
            heads, opt_in = ft.suggested_actions(lab, staged, hb)
            assert heads + opt_in, f"{lab}/{label} would print a bare `--actions` with no value"


def test_sniff_subtype_only_offers_sniffall_as_an_opt_in():
    hb = ft.heads_by_lab()
    # sniffface is not a GroovyShrew head, but it does feed the merged sniffall column --
    # offered, not assumed, because it would define sniffing as that subtype alone.
    assert ft.suggested_actions(SNIFFALL_LAB, {"sniffface"}, hb) == ([], ["sniffall"])
    # Plain `sniff` -- what prepare stages a `sniffall` row as -- trains it outright, no opt-in.
    assert ft.suggested_actions(SNIFFALL_LAB, {"rear", "sniff"}, hb) == (["rear", "sniffall"], [])
    # A lab's own subtype head is trainable, and still leaves sniffall on offer.
    assert ft.suggested_actions(SNIFFALL_LAB, {"sniffgenital"}, hb) == (["sniffgenital"], ["sniffall"])
    # Nothing sniff-related: no opt-in.
    assert ft.suggested_actions(SNIFFALL_LAB, {"rear"}, hb) == (["rear"], [])
    # A plain-sniff lab has no sniffall column to offer.
    assert ft.suggested_actions(PLAIN_LAB, {"sniff"}, hb) == (["sniff"], [])


# ------------------------------------------------------------------ the bundle's input forms

def test_also_scored_parses_translates_and_validates():
    """`--also-scored` marks combinations a video watched for but never saw. Its actions go
    through the same sniffall translation as bout rows, or they would score nothing."""
    allowed = ft.annotatable_actions(SNIFFALL_LAB, ft.heads_by_lab())
    assert ft.parse_also_scored("mouse1,mouse2,sniffgenital", allowed) == [
        ("mouse1", "mouse2", "sniffgenital")]
    assert ft.parse_also_scored("1,2,sniffall;mouse2,self,rear", allowed) == [
        ("mouse1", "mouse2", "sniff"), ("mouse2", "self", "rear")]
    assert ft.parse_also_scored(None, allowed) == [] and ft.parse_also_scored("", allowed) == []
    with pytest.raises(SystemExit, match="expected 'agent,target,action'"):
        ft.parse_also_scored("mouse1,mouse2", allowed)
    with pytest.raises(SystemExit, match="not one this lab can train"):
        ft.parse_also_scored("mouse1,mouse2,tailrattle", allowed)


def _bundle_annotation(path, rows):
    pd.DataFrame([dict(agent_id=a, target_id=t, action=act, start_frame=s, stop_frame=e)
                  for a, t, act, s, e in rows]).to_parquet(path)


def test_read_annotation_dir_matches_by_stem_and_by_video_id(tmp_path):
    """The bundle names its files by video_id and the repo by tracking stem; both resolve."""
    from hidra.cli import vid_of

    d = tmp_path / "annot"
    d.mkdir()
    by_stem = {"mouseA_day1": "/somewhere/mouseA_day1.parquet",
               "mouseB_day1": "/somewhere/mouseB_day1.parquet"}
    _bundle_annotation(d / "mouseA_day1.parquet", [("mouse1", "self", "rear", 10, 20)])
    vid_b = vid_of(by_stem["mouseB_day1"])
    _bundle_annotation(d / f"{vid_b}.parquet", [("mouse2", "mouse1", "sniff", 5, 9)])

    out = ft.read_annotation_dir(str(d), by_stem)
    assert sorted(out["stem"]) == ["mouseA_day1", "mouseB_day1"]
    assert set(out["action"]) == {"rear", "sniff"}
    assert out.loc[out["stem"] == "mouseA_day1", "target"].iloc[0] == "self"


def test_read_annotation_dir_reports_what_it_cannot_place(tmp_path):
    d = tmp_path / "annot"
    d.mkdir()
    _bundle_annotation(d / "nobody.parquet", [("mouse1", "self", "rear", 0, 5)])
    with pytest.raises(SystemExit, match="match no tracking file"):
        ft.read_annotation_dir(str(d), {"mouseA_day1": "/somewhere/mouseA_day1.parquet"})
    pd.DataFrame([dict(agent_id="mouse1", start_frame=0)]).to_parquet(d / "mouseA_day1.parquet")
    with pytest.raises(SystemExit, match="missing column"):
        ft.read_annotation_dir(str(d), {"mouseA_day1": "/somewhere/mouseA_day1.parquet",
                                        "nobody": "/somewhere/nobody.parquet"})


def test_read_annotations_dispatches_on_a_directory(tmp_path):
    d = tmp_path / "annot"
    d.mkdir()
    _bundle_annotation(d / "v.parquet", [("mouse1", "self", "rear", 10, 20)])
    by_stem = {"v": "/somewhere/v.parquet"}
    excl = ft.read_annotations(str(d), by_stem=by_stem)
    incl = ft.read_annotations(str(d), stop_inclusive=True, by_stem=by_stem)
    assert excl.loc[0, "stop_frame"] == 20 and incl.loc[0, "stop_frame"] == 21


def test_a_new_sniffall_column_accepts_the_sniff_family():
    """A head-free slot given a `sniffall` column has no published heads at all, so the
    family that feeds the merged channel has to be admitted from `--new-head` as well --
    otherwise the only label it would take is the one that supervises nothing."""
    slot = "CRIM13"
    for label in ["sniff", *ft.SNIFF_FAMILY]:
        out = ft.check_actions(_annot((label, "mouse2")), slot, extra_actions=["sniffall"])
        assert len(out) == 1
    # `sniffall` itself is still translated to the family member the trainer reads.
    assert list(ft.check_actions(_annot(("sniffall", "mouse2")), slot,
                                 extra_actions=["sniffall"])["action"]) == ["sniff"]
    # Without a sniffall column there is nothing to feed, so the family is not admitted.
    with pytest.raises(SystemExit, match="no head for"):
        ft.check_actions(_annot(("sniffbody", "mouse2")), slot, extra_actions=["rear"])

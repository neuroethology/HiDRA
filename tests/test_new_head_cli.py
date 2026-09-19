"""`finetune.py`'s `--new-head` handling -- pure pandas, no weights, no GPU.

The trap: a new column's action is, by definition, one the lab has no head for, which is
exactly what `prepare`/`train` otherwise reject. `--new-head` must admit precisely that one
action, refuse anything that is already a head or not in the vocabulary, and keep the
seed head honest.
"""
import pandas as pd
import pytest

from hidra import finetune as ft

LAB = "GroovyShrew"


def _annot(*rows):
    return pd.DataFrame(
        [dict(file="v.parquet", stem="v", agent="mouse1", target=t, action=a,
              start_frame=0, stop_frame=10) for a, t in rows])


def test_check_actions_admits_only_the_declared_new_action():
    annot = _annot(("rear", "self"), ("attack", "mouse2"))
    with pytest.raises(SystemExit, match="no head for \\['attack'\\]"):
        ft.check_actions(annot, LAB)
    out = ft.check_actions(annot, LAB, extra_actions=["attack"])
    assert sorted(out["action"].unique()) == ["attack", "rear"] and len(out) == 2
    with pytest.raises(SystemExit, match="no head for \\['mount'\\]"):
        ft.check_actions(_annot(("mount", "mouse2")), LAB, extra_actions=["attack"])


def test_resolve_new_heads_accepts_valid_pairs_and_seeds():
    hb = ft.heads_by_lab()
    pairs, seed = ft.resolve_new_heads(["GroovyShrew,attack"], LAB, hb, "LyricalHare,attack")
    assert pairs == [("GroovyShrew", "attack")] and seed == ("LyricalHare", "attack")
    assert ft.resolve_new_heads([], LAB, hb) == ([], None)
    # Several columns at once, and a donor lab rather than one donor column.
    pairs, seed = ft.resolve_new_heads(["GroovyShrew,attack", "GroovyShrew,mount"], LAB, hb,
                                       "LyricalHare")
    assert pairs == [("GroovyShrew", "attack"), ("GroovyShrew", "mount")]
    assert seed == ("LyricalHare", None)


@pytest.mark.parametrize("spec,seed,match", [
    ("LyricalHare,attack", None, "but --lab is"),                # lab must be the staged lab
    ("GroovyShrew,rear", None, "already has a head column"),      # exists -> plain fine-tune
    ("GroovyShrew,tailrattle", None, "not a behaviour in the vocabulary"),
    ("GroovyShrew", None, "expected 'Lab,action'"),
    ("GroovyShrew,attack", "GroovyShrew,attack", "not an existing head"),
    ("GroovyShrew,attack", "Nobody,attack", "not an existing head"),
    ("GroovyShrew,attack", "CRIM13", "not a lab with published heads"),
    ("GroovyShrew,attack", "GroovyShrew", "cannot donate to itself"),
])
def test_resolve_new_heads_refuses_with_a_reason(spec, seed, match):
    with pytest.raises(SystemExit, match=match):
        ft.resolve_new_heads([spec], LAB, ft.heads_by_lab(), seed)


def test_seed_from_without_new_head_is_an_error():
    with pytest.raises(SystemExit, match="--seed-from only makes sense"):
        ft.resolve_new_heads([], LAB, ft.heads_by_lab(), "LyricalHare,attack")


def test_a_head_free_slot_becomes_your_own_lab():
    """The zip's "reuse a head-free slot as your lab" workflow, validated here rather than
    silently producing a checkpoint with no column for it."""
    hb = ft.heads_by_lab()
    pairs, seed = ft.resolve_new_heads(["CRIM13,rear", "CRIM13,attack"], "CRIM13", hb,
                                       "GroovyShrew")
    assert pairs == [("CRIM13", "rear"), ("CRIM13", "attack")] and seed == ("GroovyShrew", None)
    # ... and its bouts stage, though the lab has no published head at all.
    annot = _annot(("rear", "self"), ("attack", "mouse2"))
    out = ft.check_actions(annot, "CRIM13", extra_actions=["rear", "attack"])
    assert sorted(out["action"].unique()) == ["attack", "rear"]
    with pytest.raises(SystemExit, match="no classifier heads for lab"):
        ft.check_actions(annot, "CRIM13")


def test_the_scrambled_lab_is_refused_as_a_slot():
    with pytest.raises(SystemExit, match="cannot take a head"):
        ft.resolve_new_heads(["MABe22_movies,rear"], "MABe22_movies", ft.heads_by_lab())


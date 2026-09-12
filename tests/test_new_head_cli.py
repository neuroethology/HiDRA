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


def test_resolve_new_head_accepts_a_valid_pair_and_seed():
    hb = ft.heads_by_lab()
    pair, seed = ft.resolve_new_head("GroovyShrew,attack", LAB, hb, "LyricalHare,attack")
    assert pair == ("GroovyShrew", "attack") and seed == ("LyricalHare", "attack")
    assert ft.resolve_new_head(None, LAB, hb) == (None, None)


@pytest.mark.parametrize("spec,seed,match", [
    ("LyricalHare,attack", None, "but --lab is"),                # lab must be the staged lab
    ("GroovyShrew,rear", None, "already has a head column"),      # exists -> plain fine-tune
    ("GroovyShrew,tailrattle", None, "not a behaviour in the vocabulary"),
    ("GroovyShrew", None, "expected 'Lab,action'"),
    ("GroovyShrew,attack", "GroovyShrew,attack", "not an existing head"),
    ("GroovyShrew,attack", "Nobody,attack", "not an existing head"),
])
def test_resolve_new_head_refuses_with_a_reason(spec, seed, match):
    with pytest.raises(SystemExit, match=match):
        ft.resolve_new_head(spec, LAB, ft.heads_by_lab(), seed)


def test_seed_from_without_new_head_is_an_error():
    with pytest.raises(SystemExit, match="--seed-from only makes sense"):
        ft.resolve_new_head(None, LAB, ft.heads_by_lab(), "LyricalHare,attack")


def test_new_head_for_a_lab_without_heads_is_refused():
    with pytest.raises(SystemExit, match="no published head"):
        ft.resolve_new_head("CRIM13,attack", "CRIM13", ft.heads_by_lab())

#!/usr/bin/env python3
"""Compatibility shim: `python predict.py` still works from a source checkout.

Everything moved into the installed `hidra` package (see pyproject.toml); the equivalent
entry point is `hidra-predict`. This file only exists so the commands in the README and in
docs/ keep working when the repo is used in place, without an install.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(os.path.join(_SRC, "hidra")):
    sys.path.insert(0, _SRC)
    # Subprocesses (`python -m hidra.run_allbehaviors_perlab`) need to find the package too.
    _pp = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = _SRC + (os.pathsep + _pp if _pp else "")

from hidra.cli import main  # noqa: E402

if __name__ == "__main__":
    main()

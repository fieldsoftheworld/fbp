"""Make ``pytest experiments/tests`` work from the repository root.

The stages normally run through ``experiments/run.sh``, which puts
``experiments/`` on ``PYTHONPATH`` and points ``FTW_PLANET_REPO`` at the
ftw-planet checkout. Tests are run directly, so do the same two things here.
"""

import os
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))
os.environ.setdefault("FTW_PLANET_REPO", str(Path.home() / "github" / "ftw-planet"))

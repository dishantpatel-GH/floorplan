"""Make the repo root and this folder importable when pytest is run from anywhere."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parents[1], HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

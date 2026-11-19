"""Runner tests. Run from runner/: .venv/bin/python -m unittest discover -s tests -v"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shared_path  # noqa: E402,F401  (puts shared/ on sys.path)

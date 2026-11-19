"""Puts the shared benchmark library (task.py, ...) on sys.path; import it before those modules.

The runner and every app use the same library: BENCHMARK_SHARED_DIR if set, else the repo's shared/ folder.
run_all.py passes SHARED_DIR on to each app it starts, so they always import the runner's copy.
"""

import os
import sys
from pathlib import Path

SHARED_DIR = Path(os.environ.get("BENCHMARK_SHARED_DIR") or Path(__file__).resolve().parent.parent / "shared").resolve()
if not (SHARED_DIR / "task.py").is_file():
    sys.exit(f"Shared benchmark library not found in {SHARED_DIR}; set BENCHMARK_SHARED_DIR to the repo's shared/ folder")
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

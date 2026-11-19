#!/usr/bin/env python3
"""Record one real prediction per classifier for the model pages (writes models/samples.json).

Each entry is run in its own folder with its own .venv, using the same constructor call that
the page shows. Entries that are paid, remote, need a server that is not running, or are not set
up are skipped and keep whatever sample they had before.

    python3 models/capture_samples.py              # every entry that can run locally
    python3 models/capture_samples.py nli kev      # only these entries
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))
from generate_pages import EXAMPLES, SAMPLE_TEXT, shortlist_example  # noqa: E402

sys.path.insert(0, str(ROOT / "runner"))
from apps import APPS  # noqa: E402

OUT = ROOT / "models" / "samples.json"
SKIP = {"djev": "remote GPU server; sends text off this machine"}
# Paid entries (jev: one billed request) run only when named on the command line.
PAID = {"jev"}

PROGRAM = """
import json, sys, time
from task import current_task
T = current_task()
labels = {s["id"]: s["description"] for s in T.labels}
instruction = T.instruction
from classifier import *
t0 = time.perf_counter()
%s
t1 = time.perf_counter()
p = clf.predict(%r)
t2 = time.perf_counter()
probs = {k: round(float(v), 4) for k, v in sorted(p.probabilities.items(), key=lambda kv: -kv[1])}
print("@@" + json.dumps({"label": p.label, "confidence": round(float(p.confidence), 4), "probabilities": probs,
                         "load_s": round(t1 - t0, 1), "predict_s": round(t2 - t1, 2)}))
"""


# Shortlist variants: the page's own two-stage example on Banking77 (it defines keep and p).
PROGRAM_SHORTLIST = """
import json, time
t0 = time.perf_counter()
%s
t1 = time.perf_counter()
probs = {k: round(float(v), 4) for k, v in sorted(p.probabilities.items(), key=lambda kv: -kv[1]) if v > 0}
print("@@" + json.dumps({"label": p.label, "confidence": round(float(p.confidence), 4), "probabilities": probs,
                         "shortlist": list(keep), "load_s": round(t1 - t0, 1), "predict_s": 0.0}))
"""


def main() -> None:
    wanted = set(sys.argv[1:])
    samples = json.loads(OUT.read_text()) if OUT.exists() else {}
    for app in APPS:
        if wanted and app.name not in wanted:
            continue
        if app.name in PAID and app.name not in wanted:
            print(f"skip {app.name}: paid API (name it explicitly to run one billed request)")
            continue
        if app.name in SKIP or (not app.shortlist and app.name not in EXAMPLES):
            print(f"skip {app.name}: {SKIP.get(app.name, 'no example')}")
            continue
        folder = app.path
        python = folder / ".venv" / "bin" / "python"
        if not python.exists():
            print(f"skip {app.name}: not set up")
            continue
        task = "banking77" if app.shortlist else "ag_news"
        program = (PROGRAM_SHORTLIST % shortlist_example(app) if app.shortlist
                   else PROGRAM % (EXAMPLES[app.name], SAMPLE_TEXT))
        env = {**os.environ, "PYTHONPATH": str(ROOT / "shared"), "BENCHMARK_TASK": task}
        start = time.time()
        try:
            r = subprocess.run([str(python), "-c", program], cwd=folder, env=env,
                               capture_output=True, text=True,
                               timeout=int(os.environ.get("CAPTURE_TIMEOUT", "1200")))
        except subprocess.TimeoutExpired:
            print(f"fail {app.name}: timed out")
            continue
        line = next((l for l in r.stdout.splitlines() if l.startswith("@@")), None)
        if line is None:
            print(f"fail {app.name}: {(r.stderr or r.stdout).strip().splitlines()[-1:]}")
            continue
        samples[app.name] = {**json.loads(line[2:]), "date": date.today().isoformat()}
        OUT.write_text(json.dumps(samples, indent=1) + "\n")
        print(f"ok   {app.name} ({time.time() - start:.0f}s): {samples[app.name]['label']}")


if __name__ == "__main__":
    main()

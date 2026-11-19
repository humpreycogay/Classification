"""Setup-time probe check for Winnow (run by setup.sh; exit 1 on failure). Never used in a timed run.

Starts server/serve.sh if nothing answers at WINNOW_URL (and stops it afterwards), then checks:
  - three fixed AG News probes return the expected label;
  - every answer's probabilities sum to 1 (|sum - 1| <= 1e-3);
  - the longest prompt of each task (its longest test text with every label, at most the server's 64) fits the
    server's context (POST /v1/winnow/inspect counts the positions without running the model).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from classifier import MAX_OPTIONS, MODEL, TASK, TASK_LABELS, WinnowClassifier
from probes import AG_NEWS_PROBES, check_probes
from task import load_split, load_task

APP_DIR = Path(__file__).resolve().parent


def ensure_server(clf: WinnowClassifier):
    """The server process this check started, or None if one was already running."""
    try:
        if httpx.get(f"{clf.url}/health", timeout=5).status_code == 200:
            return None
    except httpx.TransportError:
        pass
    print("starting the Winnow server (server/serve.sh) ...", flush=True)
    log = (APP_DIR / "results").resolve()
    log.mkdir(exist_ok=True)
    with open(log / "server-setup.log", "w") as f:
        proc = subprocess.Popen([str(APP_DIR / "server" / "serve.sh")], stdout=f, stderr=subprocess.STDOUT,
                                start_new_session=True)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            sys.exit(f"Winnow server exited early; see {log / 'server-setup.log'}")
        try:
            if httpx.get(f"{clf.url}/health", timeout=5).status_code == 200:
                return proc
        except httpx.TransportError:
            pass
        time.sleep(3)
    os.killpg(proc.pid, signal.SIGTERM)
    sys.exit("Winnow server not ready in 10 minutes")


def main() -> None:
    clf = WinnowClassifier(TASK_LABELS, TASK.instruction)
    proc = ensure_server(clf)
    try:
        check_probes(clf.predict_many, name="Winnow probe check")
        preds = clf.predict_many(list(AG_NEWS_PROBES))
        worst = max(abs(sum(p.probabilities.values()) - 1) for p in preds)
        print(f"probabilities sum to 1 within {worst:.1e}")
        if worst > 1e-3:
            sys.exit("Winnow probe check failed: probabilities do not sum to 1")
        for name in ("ag_news", "banking77"):
            task = load_task(name)
            criteria = {s["name"]: s["description"] for s in task.labels[:MAX_OPTIONS]}  # as the app sends them
            longest = max(load_split(task, "test")["text"], key=len)
            body = {"model": MODEL, "state": longest, "questions": {"label": {
                "type": "choice", "instructions": task.instruction, "criteria": criteria}}}
            r = httpx.post(f"{clf.url}/v1/winnow/inspect", json=body, timeout=120)
            r.raise_for_status()
            report = r.json()  # prefix (state) and per-question suffix token counts; runtime.context = n_ctx
            used, context = report["prefix_tokens"] + sum(report["suffix_tokens"]), report["runtime"]["context"]
            print(f"{name}: longest prompt ({len(criteria)} options) = {used} tokens; context {context}")
            if used >= context:
                sys.exit(f"Winnow check failed: {name}'s longest prompt does not fit the context (raise WINNOW_CONTEXT)")
        print("Winnow checks passed")
    finally:
        if proc is not None:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)


if __name__ == "__main__":
    main()

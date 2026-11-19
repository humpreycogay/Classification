"""The single entry point (./run at the repo root calls this): pick one classifier and run it, or all.

    ./run list                          every classifier: ready to run, or why not; its latest result
    ./run list --task banking77         the same, for another task
    ./run laya                          run one classifier (AG News, 200 items, seed 42)
    ./run llmqwen --task banking77 --n 500
    ./run tfidf_logreg_4k --dry-run     show the command without running it
    ./run embedsim -- --batch-size 64   anything after -- goes to the app's evaluate.py as is
    ./run setup laya                    run the classifier's setup.sh (virtualenv, model download)
    ./run status                        runs in progress: items done, %, time left (also detached runs)

    ./run all [run_all.py options]      every classifier + the comparison report (reports/<task>.html)
    ./run report [--task banking77]     rebuild the report from each classifier's latest results

Names are the runner registry's (runner/apps.py), so variants such as tfidf_logreg_4k or
verdict_shortlist run with their own arguments. Settings come from the root .env.

Unlike run_all.py, this writes only to the classifier's own results folder: no reports/runs/<timestamp>/
folder and no report. The results are validated like a runner run, and the next run_all.py report
(or ./run report) picks them up as that classifier's latest results.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from apps import APPS, APPS_BY_NAME, App
from run_all import (REPORTS_DIR, ROOT, app_command, estimated_seconds, preflight, result_files,
                     local_server, task_limits)
import shared_path  # noqa: F401  (puts shared/ on sys.path)
import task as tasks
from sampling import expected_sample
from validate import validate_result

NO_TIMEOUT = float("inf")


def find_app(name: str) -> App:
    if name in APPS_BY_NAME:
        return APPS_BY_NAME[name]
    lowered = {a.name.lower(): a for a in APPS}
    if name.lower() in lowered:
        return lowered[name.lower()]
    close = difflib.get_close_matches(name, list(APPS_BY_NAME), n=3, cutoff=0.5)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    sys.exit(f"Unknown classifier {name!r}.{hint} See: ./run list")


def load_task(name: str) -> tasks.Task:
    try:
        return tasks.load_task(name)
    except ValueError as e:
        sys.exit(str(e))


def latest_result(app: App, task: tasks.Task) -> str:
    """'<date> n=<N> acc <x>%' for the classifier's most recent result on this task, or '-'."""
    best = None
    for meta_path in app.results_dir.glob("run_*.json"):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("task", tasks.DEFAULT_TASK) != task.name or "n" not in meta:
            continue
        stamp = meta.get("finished_at") or ""
        if best is None or stamp > best[0]:
            best = (stamp, meta, meta_path)
    if best is None:
        return "-"
    stamp, meta, meta_path = best
    csv = meta_path.with_name(meta_path.name.replace("run_", "predictions_", 1).replace(".json", ".csv"))
    accuracy = ""
    with contextlib.suppress(Exception):
        df = pd.read_csv(csv, usecols=["label", "pred"])
        accuracy = f"  acc {(df['label'] == df['pred']).mean():.1%}"
    return f"{stamp[:10]}  n={meta['n']}{accuracy}"


def cmd_list(args: argparse.Namespace) -> None:
    task = load_task(args.task)
    print(f"Task: {task.name} ({len(task.label_ids)} labels).  Run one with: ./run <name>\n")
    rows = []
    for app in APPS:
        reason = preflight(app, 200, 1, NO_TIMEOUT, allow_paid=False, task=task)
        status = "ready" if reason is None else ("n/a" if task_limits(app, task) else "NOT READY")
        rows.append((app.name, status, latest_result(app, task), app.display_title, reason))
    header = ("NAME", "STATUS", "LATEST RESULT", "CLASSIFIER")
    widths = [max(len(r[i]) for r in [*rows, header]) for i in range(3)]
    line = lambda r: f"  {r[0]:<{widths[0]}}  {r[1]:<{widths[1]}}  {r[2]:<{widths[2]}}  {r[3]}"  # noqa: E731
    print(line(header))
    for row in rows:
        print(line(row))
        if row[1] == "NOT READY":
            print(f"  {'':<{widths[0]}}  └ {row[4]}")
    if any(r[1] == "n/a" for r in rows):
        print("\n  n/a = not for this task: more labels than the classifier takes (use its _shortlist variant),\n"
              "        a shortlist the task does not need, no train split, or text that must stay local.")


def cmd_setup(args: argparse.Namespace) -> None:
    app = find_app(args.name)
    script = app.path / "setup.sh"
    if not script.exists():
        sys.exit(f"{app.name} has no setup.sh ({app.path})")
    print(f"Setting up {app.name} ({app.path.name}/setup.sh) ...", flush=True)
    shared = str(shared_path.SHARED_DIR)  # setup scripts import classifier.py, which needs the shared library
    env = {**os.environ, "BENCHMARK_SHARED_DIR": shared,
           "PYTHONPATH": os.pathsep.join(p for p in (shared, os.environ.get("PYTHONPATH")) if p)}
    sys.exit(subprocess.call(["bash", str(script)], cwd=app.path, env=env))


def cmd_run(args: argparse.Namespace, passthrough: list[str]) -> None:
    app = find_app(args.name)
    task = load_task(args.task)
    try:
        expected = expected_sample(args.n, args.seed, task.name)
    except ValueError as e:
        sys.exit(f"Cannot run a sample of {args.n}: {e}")
    n = len(expected)
    extra = ("--latency-items", str(args.latency_items), *(("--sweep",) if args.sweep else ()), *passthrough)
    cmd, env = app_command(app, n, args.seed, task.name, extra)

    print(f"{app.name}: {app.display_title}  ({app.runs_on})")
    print(f"Task {task.name}, {n} items, seed {args.seed}.  Results: {app.results_dir.relative_to(ROOT)}/")
    if args.dry_run:
        shown = [c.replace(f"{app.path}/", "") for c in cmd]  # .venv/bin/python, relative to the app folder
        print(f"\ncd {app.path.relative_to(ROOT)} && {' '.join(shown)}")
        return
    latency_requests = args.latency_items + (4 * 128 + 3 * 32 if args.sweep else 0)
    if reason := preflight(app, n, 1, NO_TIMEOUT, args.allow_paid, latency_requests, task):
        hint = f"\nFix: ./run setup {app.name}" if reason.startswith("no .venv") else ""
        sys.exit(f"Cannot run {app.name}: {reason}{hint}")
    if estimate := estimated_seconds(app, n):
        print(f"Estimated time: ~{estimate / 60:.0f} min (from its last run)")
    print(flush=True)

    server_ctx = local_server(app, REPORTS_DIR) or contextlib.nullcontext()
    started = time.time()
    with server_ctx:
        code = subprocess.call(cmd, cwd=app.path, env=env)
    if code != 0:
        sys.exit(f"\n{app.name} failed (exit code {code}).")

    csv, meta = result_files(app, n, args.seed, task.name)
    problems = validate_result(csv, meta, expected=expected, seed=args.seed, confidence_rule=app.confidence_rule,
                               labels=task.label_ids, sum_tol=app.prob_sum_tol, not_before=started)
    elapsed = time.time() - started
    if problems:
        sys.exit(f"\n{app.name} finished in {elapsed:.0f} s but its results failed validation:\n  - "
                 + "\n  - ".join(problems[:8]))
    print(f"\n{app.name} done in {elapsed:.0f} s ({datetime.now():%H:%M}); results validated:"
          f"\n  {csv.relative_to(ROOT)}\n  {meta.relative_to(ROOT)}"
          f"\nCompare with the others: ./run report"
          + (f" --task {task.name}" if task.name != tasks.DEFAULT_TASK else ""))


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cmd_status(args: argparse.Namespace) -> None:
    """Every classifier run in progress, from the status files the harness keeps in reports/progress/; also
    runs that stopped without finishing (killed, crashed or timed out), which leave their file behind."""
    from harness import PROGRESS_DIR  # noqa: E402 (shared/ is on sys.path via run_all)
    from timing import duration

    rows = []
    for path in sorted(PROGRESS_DIR.glob("*.json")):
        try:
            s = json.loads(path.read_text())
        except (OSError, ValueError):
            continue  # being replaced right now
        rows.append((path, s, pid_alive(s["pid"])))
    if args.clean:
        for path, _, alive in rows:
            if not alive:
                path.unlink(missing_ok=True)
        rows = [r for r in rows if r[2]]
    if not rows:
        sys.exit("No classifier is running." + ("" if args.clean else "  (./run status --clean removes stopped runs)"))
    now = time.time()
    for _, s, alive in rows:
        where = f"{s['app']}  {s['task']} n={s['n']:,} seed {s['seed']}"
        if s.get("line"):
            what = s["line"].removeprefix("progress:").strip()
        else:
            what = f"{s['phase']} ({duration(now - s['started'])} since start)"
        age = duration(now - s["updated"])
        if alive:
            print(f"  RUNNING  {where}\n           {what}   [pid {s['pid']}, updated {age} ago]")
        else:
            print(f"  STOPPED  {where}\n           last seen: {what}   "
                  f"[{datetime.fromtimestamp(s['updated']):%Y-%m-%d %H:%M}, process gone: killed, crashed or timed out]")
    if any(not alive for *_, alive in rows):
        print("\n./run status --clean removes the stopped entries.")


def main() -> None:
    argv = sys.argv[1:]
    passthrough: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        sys.exit(__doc__)

    if argv[0] in ("all", "report"):  # the full benchmark: runner/run_all.py
        extra = ["--report-only"] if argv[0] == "report" else []
        run_all = Path(__file__).resolve().parent / "run_all.py"
        sys.exit(subprocess.call([sys.executable, str(run_all), *extra, *argv[1:], *passthrough]))
    if argv[0] == "list":
        p = argparse.ArgumentParser(prog="./run list")
        p.add_argument("--task", default=tasks.DEFAULT_TASK)
        cmd_list(p.parse_args(argv[1:]))
    elif argv[0] == "status":
        p = argparse.ArgumentParser(prog="./run status")
        p.add_argument("--clean", action="store_true", help="forget runs that stopped without finishing")
        cmd_status(p.parse_args(argv[1:]))
    elif argv[0] == "setup":
        p = argparse.ArgumentParser(prog="./run setup")
        p.add_argument("name")
        cmd_setup(p.parse_args(argv[1:]))
    else:
        p = argparse.ArgumentParser(prog="./run", description="Run one classifier. See ./run help.")
        p.add_argument("name", help="classifier name (./run list)")
        p.add_argument("--task", default=tasks.DEFAULT_TASK, help="task manifest in data/tasks/ (default ag_news)")
        p.add_argument("--n", type=int, default=200, help="sample size (0 = full test set; default 200)")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--latency-items", type=int, default=100, help="items timed one at a time (0 = skip)")
        p.add_argument("--sweep", action="store_true", help="also time batch sizes / concurrency levels")
        p.add_argument("--allow-paid", action="store_true", help="let a paid API run above the item cap")
        p.add_argument("--dry-run", action="store_true", help="print the command, do not run it")
        cmd_run(p.parse_args(argv), passthrough)


if __name__ == "__main__":
    main()

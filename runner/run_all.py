"""Master runner: evaluate every classifier app on the same task sample (AG News by default), then build one HTML report.

Apps run one at a time (they share the GPU and memory). Each app is checked first and
skipped with a reason if it cannot run here; a failing app is recorded and the run goes on.

Usage:
    python run_all.py                       # all apps, 200 articles (smoke tier), seed 42
    python run_all.py --only laya nli       # run a subset (other apps are reported from their latest results)
    python run_all.py --skip jev djev
    python run_all.py --tier common         # 1,000 articles; --tier full = all 7,600 (same as --n 0)
    python run_all.py --n 201               # any size from 1 to 7,600 is produced exactly
    python run_all.py --seeds 42 43 44      # one run per seed (a different sample each)
    python run_all.py --repeat 3            # three runs per seed on the same sample
    python run_all.py --report-only         # no runs: rebuild the report from each app's latest results
    python run_all.py --sweep               # also time batch sizes 1/8/16/32 or concurrency 1/2/4/8
    python run_all.py --require-ac          # refuse to time on battery power
    python run_all.py --task banking77 --only tfidf_cnb nli   # another task (data/tasks/<task>.json)

Every timed run records the machine's conditions (power source, Low Power Mode, load), each app
process's peak memory footprint and RSS (/usr/bin/time -l), Ollama's loaded-model memory, the
OpenJev server's peak RSS (polled) and, inside each app, per-call and interactive latency.

Results are identified by task, sample size and seed: <app>/results/predictions_<task>_n{N}_seed{S}.csv
and run_<task>_n{N}_seed{S}.json (older AG News names without the task, or without the seed, are
still read when their seed matches). Apps that cannot run a task (too many labels, no train split,
remote apps on a task that keeps its data local) are skipped with the reason.
The runner computes the expected sample before any app runs, rejects sizes it cannot produce, and
validates every result (validate.py); an invalid result is reported as failed and never ranked.

Output: reports/runs/<timestamp>/ (logs, copied results, summary.json, report.html = full report,
report_simple.html = plain-language summary); reports/<task>.html is always the latest summary report for that
task (reports/ag_news.html, reports/banking77.html) and reports/<task>_full.html the latest full raw report,
reached through a link on the summary. With several seeds or repeats every run is kept
under apps/<app>/seed{S}_rep{R}/; the report ranks the first seed's first valid run and adds
mean / median / SD / IQR and prediction stability over all runs. The majority-class and
stratified-random virtual baselines (baselines.py) are generated on the same sample in every run.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from apps import APPS, APPS_BY_NAME, PAID_MAX_N, App, missing_weights, setting  # settings: shell, else the root .env
from baselines import write_virtual_baselines
from report import build_report
import shared_path  # puts shared/ on sys.path
import task as tasks
from sampling import expected_sample, result_stem, sample_sha1, tier_name
from validate import validate_result

RUNNER_DIR = Path(__file__).resolve().parent
ROOT = RUNNER_DIR.parent
REPORTS_DIR = ROOT / "reports"  # generated: the latest report per task, and runs/ (one folder per run)
RUNS_DIR = REPORTS_DIR / "runs"
OPENJEV_READY_TIMEOUT = 20 * 60  # first start loads ~16 GB of weights
WINNOW_READY_TIMEOUT = 10 * 60  # loads 12.7 GB of Q8 weights into unified memory
SERVER_READY_TIMEOUT = 20 * 60  # other local servers: time allowed from start to ready


def http_get(url: str, timeout: float = 5) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, ""
    except (urllib.error.URLError, OSError):
        return 0, ""


def ollama_host(app: App | None = None) -> str:
    """The Ollama server: the app's own (app.ollama_host, e.g. LLM_HOST) when set, else OLLAMA_HOST."""
    default = setting("OLLAMA_HOST", "http://127.0.0.1:11434")
    host = (setting(app.ollama_host, default) if app and app.ollama_host else default).rstrip("/")
    return host if host.startswith("http") else f"http://{host}"


def ollama_models(host: str | None = None) -> dict[str, str] | None:
    """Pulled Ollama models (name -> digest), or None if Ollama is not running."""
    tags = ollama_tags(host)
    return None if tags is None else {name: m.get("digest", "") for name, m in tags.items()}


def ollama_tags(host: str | None = None) -> dict[str, dict] | None:
    """Pulled Ollama models (name -> /api/tags entry, with digest and size on disk), or None."""
    status, body = http_get(f"{host or ollama_host()}/api/tags")
    if status != 200:
        return None
    return {m["name"]: m for m in json.loads(body).get("models", [])}


def ollama_loaded(tag: str, host: str | None = None) -> dict | None:
    """Memory of a loaded Ollama model (/api/ps): total and GPU (VRAM) bytes."""
    status, body = http_get(f"{host or ollama_host()}/api/ps")
    if status != 200:
        return None
    for m in json.loads(body).get("models", []):
        if m.get("name") in (tag, f"{tag}:latest") or m.get("model") in (tag, f"{tag}:latest"):
            return {"kind": "ollama", "loaded_bytes": m.get("size"), "gpu_bytes": m.get("size_vram"),
                    "method": "Ollama /api/ps after the run"}
    return None


def ollama_model_tag(app: App) -> str:
    return app.model_tag


def ollama_stop(model: str, host: str | None = None) -> None:
    subprocess.run(["ollama", "stop", model], capture_output=True, timeout=60,
                   env={**os.environ, "OLLAMA_HOST": host or ollama_host()})


def openjev_url() -> str:
    return setting("OPENJEV_URL", "http://127.0.0.1:8080").rstrip("/")


def winnow_url() -> str:
    return setting("WINNOW_URL", "http://127.0.0.1:8091").rstrip("/")


def djev_url() -> str:
    return setting("DJEV_URL", "http://127.0.0.1:8000").rstrip("/")


def estimated_seconds(app: App, n: int) -> float | None:
    """One run's time for n items, from the app's most recent run metadata: classification extrapolated
    to n, plus load, warm-up and training (training counts even when it was cached last time)."""
    metas = sorted(app.results_dir.glob("run_n*.json"), key=lambda p: p.stat().st_mtime)
    for path in reversed(metas):
        try:
            meta = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if meta.get("seconds_per_item"):
            tr = meta.get("training") or {}
            setup = (meta.get("load_seconds") or 0) + (meta.get("warmup_seconds") or 0)
            if not tr.get("train_cached"):
                setup += tr.get("train_seconds") or 0
            if not tr.get("train_embed_cached"):
                setup += tr.get("train_embed_seconds") or 0
            return meta["seconds_per_item"] * n + setup
    return None


def task_limits(app: App, task: "tasks.Task") -> str | None:
    """Why the app cannot or may not run this task, or None."""
    if app.shortlist and len(task.label_ids) <= app.shortlist:
        return (f"task {task.name} has {len(task.label_ids)} labels, not more than the shortlist of {app.shortlist}: "
                "no shortlist needed (see the app without it)")
    if app.max_labels is not None and len(task.label_ids) > app.max_labels:
        return f"task {task.name} has {len(task.label_ids)} labels; this app supports at most {app.max_labels}"
    if app.remote and not task.manifest.get("remote_allowed", False):
        return f"task {task.name} does not allow sending its text to remote services"
    if app.supervision == "supervised" and "train" not in task.files:
        return f"task {task.name} has no train split for a supervised baseline"
    if reason := app.training_overlap(task.name):
        return reason
    return None


def preflight(app: App, n: int, runs: int, timeout: float, allow_paid: bool, extra_requests: int = 0,
              task: "tasks.Task | None" = None) -> str | None:
    """None if the app can run, else the reason it is skipped. runs = seeds x repeats of this job;
    extra_requests = interactive-latency and sweep requests per run, counted toward the paid cap."""
    folder = app.path
    if task is not None and (reason := task_limits(app, task)):
        return reason
    if not (folder / ".venv" / "bin" / "python").exists():
        return f"no .venv: run {folder.name}/setup.sh"
    requests = (n + extra_requests) * runs
    if app.cost == "paid" and requests > PAID_MAX_N and not allow_paid:
        return (f"paid API: ({n} items + {extra_requests} latency/sweep requests) x {runs} run(s) = {requests} "
                f"requests is above {PAID_MAX_N} (pass --allow-paid to run it)")
    estimate = estimated_seconds(app, n)
    if estimate and estimate > timeout:
        return (f"estimated {estimate / 60:.0f} min per run for {n} items (from its last run, including setup) "
                f"exceeds the {timeout / 60:.0f} min timeout (raise --timeout to run it)")
    if app.needs == "jev_key" and not (setting("jev_api_key", "") or setting("TYPESAFE_API_KEY", "")):
        return "jev_api_key missing from the root .env"
    if app.needs == "ollama" or app.extra_ollama:
        # The app's model may live on its own server (app.ollama_host); extra_ollama is on the shared one (also
        # for apps whose model is not an Ollama one, e.g. winnow_shortlist's bge-m3).
        for model, host in ((app.ollama_model, ollama_host(app)), (app.extra_ollama, ollama_host())):
            if model is None:
                continue
            models = ollama_models(host)
            if models is None:
                return f"Ollama not reachable at {host}"
            tag = setting(*model)
            if tag not in models and f"{tag}:latest" not in models:
                return f"Ollama model {tag} not pulled on {host} (OLLAMA_HOST={host} ollama pull {tag})"
    if app.needs == "openjev_server":
        if http_get(f"{openjev_url()}/v1/models")[0] != 200:
            if not (folder / "server" / ".venv").exists():
                return "OpenJev server not running and not set up (run OpenJev/server/setup.sh)"
            if problem := missing_weights(app):
                return f"{problem} (run OpenJev/server/setup.sh)"
    elif app.needs == "winnow_server":
        if http_get(f"{winnow_url()}/health")[0] != 200:
            if not (folder / "server" / "src" / ".build" / "bin" / "winnow-server").exists():
                return "Winnow server not running and not built (run winnow/server/setup.sh)"
            if problem := missing_weights(app):
                return f"{problem} (run winnow/server/setup.sh)"
    elif problem := missing_weights(app):
        return f"{problem} (run ./run setup {app.name})"
    if app.needs == "djev_server" and http_get(f"{djev_url()}/ready")[0] != 200:
        return f"djev server not reachable at {djev_url()} (needs an NVIDIA B200 host + infra/tunnel.sh)"
    return None


def conditions() -> dict:
    """Machine state that changes timings: power source and charge, Low Power Mode, load average."""
    batt = sh(["pmset", "-g", "batt"])
    source = re.search(r"drawing from '([^']+)'", batt)
    charge = re.search(r"(\d+)%", batt)
    low = re.search(r"lowpowermode\s+(\d)", sh(["pmset", "-g"]))
    return {
        "power": source.group(1) if source else None,
        "on_battery": bool(source and "Battery" in source.group(1)),
        "battery_percent": int(charge.group(1)) if charge else None,
        "low_power_mode": bool(low and low.group(1) == "1"),
        "load_1m": round(os.getloadavg()[0], 2),
    }


def group_rss_bytes(pgid: int) -> int:
    """Summed RSS of every process in a process group (0 if none)."""
    pids = sh(["pgrep", "-g", str(pgid)]).split()
    if not pids:
        return 0
    return sum(int(x) for x in sh(["ps", "-o", "rss=", "-p", ",".join(pids)]).split() if x.isdigit()) * 1024


def parse_time_l(path: Path) -> dict:
    """Peak RSS and peak memory footprint from macOS `/usr/bin/time -l -o path` output."""
    if not path.exists():
        return {}
    text = path.read_text()
    out = {}
    for key, label in (("peak_rss_bytes", "maximum resident set size"), ("peak_footprint_bytes", "peak memory footprint")):
        m = re.search(rf"(\d+)\s+{label}", text)
        if m:
            out[key] = int(m.group(1))
    return out


class LocalServer:
    """A local model server the runner starts if nothing is serving yet, and stops afterwards only if it
    started it: start script, health URL (HTTP 200 = ready), log path. While apps run, the server's memory
    is polled (sample()) and recorded with each run."""

    def __init__(self, name: str, start_script: Path, health_url: str, log_path: Path,
                 ready_timeout: float = SERVER_READY_TIMEOUT, kind: str | None = None) -> None:
        self.name, self.start_script, self.health_url = name, start_script, health_url
        self.log_path, self.ready_timeout = log_path, ready_timeout
        self.kind = kind or f"{name.lower()}_server"  # server_memory "kind" in the run metadata
        self.proc: subprocess.Popen | None = None
        self.start_seconds: float | None = None  # None when the server was already running
        self.peak_rss = 0  # peak RSS of the server's process group while apps ran (only if we started it)

    def ready(self) -> bool:
        return http_get(self.health_url)[0] == 200

    def sample(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.peak_rss = max(self.peak_rss, group_rss_bytes(self.proc.pid))

    def memory(self) -> dict | None:
        if not self.proc:
            return None
        return {"kind": self.kind, "peak_rss_bytes": self.peak_rss,
                "method": "RSS of the server process group, polled every 2 s (misses short peaks; excludes "
                          "GPU memory not counted in RSS)"}

    def __enter__(self) -> "LocalServer":
        if self.ready():
            print(f"  {self.name} server already running; leaving it as is")
            return self
        print(f"  starting {self.name} server ({self.start_script.parent.name}/{self.start_script.name}) ...", flush=True)
        with self.log_path.open("w") as log:  # the server keeps its own copy of the descriptor
            self.proc = subprocess.Popen(
                [str(self.start_script)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        t0 = time.monotonic()
        deadline = t0 + self.ready_timeout
        while not self.ready():
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.name} server exited early; see {self.log_path}")
            if time.monotonic() > deadline:
                raise RuntimeError(f"{self.name} server not ready in time")
            time.sleep(5)
        self.start_seconds = round(time.monotonic() - t0, 1)
        print(f"  {self.name} server ready in {self.start_seconds:.0f} s", flush=True)
        return self

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
            print(f"  {self.name} server stopped")


def OpenJevServer(log_path: Path) -> LocalServer:
    """OpenJev/server/serve.sh, ready when /v1/models answers."""
    return LocalServer("OpenJev", APPS_BY_NAME["OpenJev"].path / "server" / "serve.sh", f"{openjev_url()}/v1/models",
                       log_path, OPENJEV_READY_TIMEOUT, kind="openjev_server")


def WinnowServer(log_path: Path) -> LocalServer:
    """winnow/server/serve.sh (llama.cpp, Metal), ready when /health answers."""
    return LocalServer("Winnow", APPS_BY_NAME["winnow"].path / "server" / "serve.sh", f"{winnow_url()}/health",
                       log_path, WINNOW_READY_TIMEOUT, kind="winnow_server")


# Local servers the runner manages, by App.needs: log path -> LocalServer.
LOCAL_SERVERS = {"openjev_server": OpenJevServer, "winnow_server": WinnowServer}


def local_server(app: App, log_dir: Path) -> "LocalServer | None":
    """The server this app needs the runner to start and stop (None if it needs none)."""
    make = LOCAL_SERVERS.get(app.needs or "")
    return make(log_dir / f"{app.folder or app.name}-server.log") if make else None


def app_command(app: App, n: int, seed: int, task: str, extra_args: tuple[str, ...] = (),
                run_id: str | None = None) -> tuple[list[str], dict[str, str]]:
    """The evaluate.py command line (run in app.path) and environment for one run of a registry entry."""
    cmd = [str(app.path / ".venv" / "bin" / "python"), "evaluate.py",
           "--task", task, "--n", str(n), "--seed", str(seed), *app.args, *extra_args]
    if app.model_arg:  # one folder serving several models (llm/): tell it which one this entry runs, and where
        cmd += ["--model", ollama_model_tag(app), "--host", ollama_host(app)]  # results go to results/<model>/
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "BENCHMARK_TASK": task, "BENCHMARK_APP_NAME": app.name,
           "BENCHMARK_SHARED_DIR": str(shared_path.SHARED_DIR)}  # apps import the runner's shared library
    if run_id:
        env["BENCHMARK_RUN_ID"] = run_id
    return cmd, env


PROGRESS_ECHO_SECONDS = 60  # how often ./run all repeats an app's latest "progress:" line on the console


def latest_progress(log_path: Path) -> str | None:
    """The app's latest "progress:" line (harness/timing.py), from the end of its log; None if none yet."""
    try:
        with log_path.open("rb") as f:
            f.seek(max(0, f.seek(0, os.SEEK_END) - 8192))
            tail = f.read().decode(errors="replace")
    except OSError:
        return None
    lines = [line for line in tail.replace("\r", "\n").splitlines() if line.startswith("progress:")]
    return lines[-1].removeprefix("progress:").strip() if lines else None


def run_app(app: App, n: int, seed: int, timeout: float, log_path: Path, run_id: str,
            extra_args: tuple[str, ...] = (), server: "LocalServer | None" = None, task: str = tasks.DEFAULT_TASK) -> dict:
    """Run evaluate.py once under /usr/bin/time -l (peak memory of the app process); status ok or failed
    (the result files are validated separately). A server the runner started is polled for memory."""
    folder = app.path
    time_file = log_path.with_suffix(".time.txt")
    cmd, env = app_command(app, n, seed, task, extra_args, run_id)
    cmd = ["/usr/bin/time", "-l", "-o", str(time_file), *cmd]
    run_conditions = conditions()
    started = time.time()
    t0 = time.perf_counter()
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=folder, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        code = None
        echoed, last_echo = None, t0
        while True:
            try:
                code = proc.wait(timeout=2)
                break
            except subprocess.TimeoutExpired:
                if server:
                    server.sample()
                if time.perf_counter() - last_echo >= PROGRESS_ECHO_SECONDS:
                    last_echo = time.perf_counter()
                    if (line := latest_progress(log_path)) and line != echoed:
                        echoed = line
                        print(f"    {app.name}: {line}", flush=True)
                if time.perf_counter() - t0 > timeout:
                    os.killpg(proc.pid, signal.SIGKILL)  # the app, not only /usr/bin/time
                    proc.wait()
                    code = None
                    break
    wall = time.perf_counter() - t0
    result = {"wall_seconds": round(wall, 2), "started": started, "conditions": run_conditions,
              **parse_time_l(time_file)}
    if server and server.memory():
        result["server_memory"] = server.memory()
    if code is None:
        return {**result, "status": "failed", "reason": f"timed out after {timeout:.0f} s"}
    if code != 0:
        tail = log_path.read_text().strip().splitlines()[-1:] or ["(no output)"]
        return {**result, "status": "failed", "reason": f"exit code {code}: {tail[0][:200]}"}
    return {**result, "status": "ok"}


def result_files(app: App, n: int, seed: int, task: str = tasks.DEFAULT_TASK) -> tuple[Path, Path]:
    """(predictions CSV, run metadata) for (task, n, seed) in the app folder.

    For AG News, falls back to the names used before tasks existed: predictions_n{N}_seed{S}.csv, then
    predictions_n{N}.csv when its metadata records the same seed (or has no metadata; validation then
    checks the sample).
    """
    stem = result_stem(n, seed, task)
    csv, meta = app.results_dir / f"predictions_{stem}.csv", app.results_dir / f"run_{stem}.json"
    if csv.exists() or task != tasks.DEFAULT_TASK:
        return csv, meta
    seeded_csv, seeded_meta = app.results_dir / f"predictions_n{n}_seed{seed}.csv", app.results_dir / f"run_n{n}_seed{seed}.json"
    if seeded_csv.exists():
        return seeded_csv, seeded_meta
    legacy_csv, legacy_meta = app.results_dir / f"predictions_n{n}.csv", app.results_dir / f"run_n{n}.json"
    if legacy_csv.exists():
        try:
            legacy_seed = json.loads(legacy_meta.read_text()).get("seed") if legacy_meta.exists() else seed
        except json.JSONDecodeError:
            legacy_seed = None
        if legacy_seed == seed:
            return legacy_csv, legacy_meta
    return csv, meta


def record_runner_timings(meta_path: Path, entry: dict) -> None:
    """Keep the runner's own timings with the app's results, so later reports can show them."""
    meta = json.loads(meta_path.read_text())
    keys = ("wall_seconds", "server_start_seconds", "conditions", "peak_rss_bytes", "peak_footprint_bytes",
            "server_memory", "model_disk_bytes")
    meta["runner"] = {k: entry[k] for k in keys if entry.get(k) is not None}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def copy_result(csv: Path, meta: Path, dest: Path, run_dir: Path) -> dict:
    """Copy one result (CSV + metadata if present) into the run folder; return the relative paths."""
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csv, dest / csv.name)
    files = {"csv": str((dest / csv.name).relative_to(run_dir)), "run_json": None}
    if meta.exists():
        shutil.copy2(meta, dest / meta.name)
        files["run_json"] = str((dest / meta.name).relative_to(run_dir))
    return files


def take_result(app: App, run: dict, *, n: int, seed: int, expected, dest: Path, run_dir: Path,
                run_id: str | None = None, task: "tasks.Task | None" = None) -> dict:
    """Validate the app's result files for (n, seed) and copy them into the run folder.

    Fresh runs (run_id set) must have written both files during this run. A result that fails
    validation gets status "invalid" and is kept in the run folder for inspection only.
    """
    task = task or tasks.load_task()
    csv, meta = result_files(app, n, seed, task.name)
    if not csv.exists():
        if run["status"] == "existing":
            return {**run, "status": "skipped", "reason": f"no results for n={n}, seed {seed} in the app folder yet"}
        return {**run, "status": "invalid", "reason": "finished but wrote no predictions CSV"}
    problems = validate_result(csv, meta, expected=expected, seed=seed, confidence_rule=app.confidence_rule,
                               labels=task.label_ids, sum_tol=app.prob_sum_tol,
                               not_before=run.get("started") if run_id else None, run_id=run_id)
    if run_id and not problems:
        record_runner_timings(meta, run)
    run = {**run, **copy_result(csv, meta, dest, run_dir)}
    if problems:
        shown = "; ".join(problems[:4]) + (f"; +{len(problems) - 4} more" if len(problems) > 4 else "")
        run.update(status="invalid", reason=f"failed validation: {shown}")
    return run


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def environment() -> dict:
    status, body = http_get(f"{ollama_host()}/api/version")
    models = ollama_models() or {}
    semif_path = APPS_BY_NAME["SemIf"].path / "requirements.txt"
    semif_req = semif_path.read_text() if semif_path.exists() else ""
    djev_deploy = APPS_BY_NAME["djev"].path / "infra" / "deploy_gpu_host.sh"
    djev_commit = re.search(r"^COMMIT=(\w+)", djev_deploy.read_text(), re.M) if djev_deploy.exists() else None
    semif_commit = re.search(r"SemIf-OpenJev\.git@(\w+)", semif_req)
    mem = sh(["sysctl", "-n", "hw.memsize"])
    num_parallel = setting("OLLAMA_NUM_PARALLEL", "")
    return {
        "conditions_at_start": conditions(),
        "ollama_num_parallel": num_parallel or "not set (Ollama server default)",
        "machine": sh(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor(),
        "memory_gb": round(int(mem) / 2**30) if mem.isdigit() else None,
        "os": f"macOS {platform.mac_ver()[0]}" if platform.mac_ver()[0] else platform.platform(),
        "runner_python": platform.python_version(),
        "ollama_version": json.loads(body).get("version") if status == 200 else None,
        "ollama_models": {a.name: {"tag": ollama_model_tag(a), "host": ollama_host(a),
                                   "digest": (models if not a.ollama_host else ollama_models(ollama_host(a)) or {})
                                   .get(ollama_model_tag(a), "")[:12]}
                          for a in APPS if a.ollama_model},
        "pinned_commits": {
            "SemIf (semif-phase1)": semif_commit.group(1)[:7] if semif_commit else None,
            "OpenJev server": sh(["git", "-C", str(APPS_BY_NAME["OpenJev"].path / "server" / "src"), "rev-parse", "--short", "HEAD"]) or None,
            "djev-dev": djev_commit.group(1)[:7] if djev_commit else None,
            "Nimble": sh(["git", "-C", str(APPS_BY_NAME["nimble"].path / "src"), "rev-parse", "--short", "HEAD"]) or None,
        },
    }


def latest_report_path(task_name: str) -> Path:
    """reports/<task>.html: the latest summary report for the task (reports/ag_news.html, reports/banking77.html)."""
    return REPORTS_DIR / f"{task_name}.html"


def latest_full_report_path(task_name: str) -> Path:
    """reports/<task>_full.html: the latest full raw report, linked only from the summary report."""
    return REPORTS_DIR / f"{task_name}_full.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--n", type=int, help="sample size per app, 1-7600 (0 = full test set; default 200)")
    size.add_argument("--tier", choices=["smoke", "common", "full"],
                      help="named size from the task manifest (AG News: smoke = 200, common = 1000, full = 7600)")
    parser.add_argument("--task", default=tasks.DEFAULT_TASK, help="task manifest in data/tasks/ (default ag_news)")
    seed = parser.add_mutually_exclusive_group()
    seed.add_argument("--seed", type=int, default=42)
    seed.add_argument("--seeds", type=int, nargs="+", metavar="SEED", help="one run per seed (different samples)")
    parser.add_argument("--repeat", type=int, default=1, metavar="N", help="runs per seed on the same sample")
    parser.add_argument("--only", nargs="+", metavar="APP", help="run only these apps")
    parser.add_argument("--skip", nargs="+", metavar="APP", default=[], help="do not run these apps")
    parser.add_argument("--timeout", type=float, default=3600, help="seconds per app run")
    parser.add_argument("--allow-paid", action="store_true", help=f"run paid APIs above {PAID_MAX_N} items")
    parser.add_argument("--report-only", action="store_true", help="no runs: report on each app's latest results")
    parser.add_argument("--latency-items", type=int, default=100,
                        help="articles each app times one at a time for interactive latency (0 = skip)")
    parser.add_argument("--sweep", action="store_true", help="also time batch sizes / concurrency levels (128 articles)")
    parser.add_argument("--require-ac", action="store_true", help="refuse to time apps while on battery power")
    args = parser.parse_args()

    for name in (args.only or []) + args.skip:
        if name not in APPS_BY_NAME:
            sys.exit(f"Unknown app {name!r}. Apps: {', '.join(APPS_BY_NAME)}")
    seeds = args.seeds or [args.seed]
    if len(set(seeds)) != len(seeds):
        sys.exit("--seeds must not repeat a seed (use --repeat for runs on the same sample)")
    if args.repeat < 1:
        sys.exit("--repeat must be at least 1")
    try:
        task = tasks.load_task(args.task)
        task_tiers = tasks.tiers(task)
    except ValueError as e:
        sys.exit(str(e))
    if args.tier and args.tier not in task_tiers:
        sys.exit(f"task {task.name} has no {args.tier!r} tier ({', '.join(f'{k} = {v}' for k, v in task_tiers.items())})")
    n_requested = task_tiers[args.tier] if args.tier else (200 if args.n is None else args.n)
    # The exact sample every app must return, per seed; unsupported sizes stop here, before any app runs.
    try:
        samples = {s: expected_sample(n_requested, s, task.name) for s in seeds}
    except ValueError as e:
        sys.exit(f"Cannot run a sample of {n_requested}: {e}")
    n = len(samples[seeds[0]])
    if len(seeds) > 1 and len(set(sample_sha1(df) for df in samples.values())) == 1:
        sys.exit(f"Every seed gives the same {n} articles (the whole test set), so --seeds adds nothing; "
                 "use --repeat for repeated runs on it.")
    selected = [a for a in APPS if (not args.only or a.name in args.only) and a.name not in args.skip]
    start_conditions = conditions()
    if not args.report_only and start_conditions["on_battery"]:
        if args.require_ac:
            sys.exit("On battery power: timings would not be comparable. Plug in, or drop --require-ac.")
        print(f"WARNING: on battery ({start_conditions['battery_percent']}%): timings may be slower and are "
              "flagged in the report. Plug in, or use --require-ac to refuse.")
    extra_args = ("--latency-items", str(args.latency_items), *(("--sweep",) if args.sweep else ()))
    # Interactive requests, plus 4 sweep settings x 128 articles and 3 x 32 untimed agreement requests.
    latency_requests = args.latency_items + (4 * 128 + 3 * 32 if args.sweep else 0)
    repeat = 1 if args.report_only else args.repeat
    multi = len(seeds) * repeat > 1

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / stamp
    (run_dir / "logs").mkdir(parents=True)
    summary = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "report-only" if args.report_only else "run",
        "task": task.name,
        "task_info": {k: task.manifest.get(k) for k in ("version", "description", "source", "license", "sampling",
                                                          "primary_metric", "remote_allowed", "item_noun")},
        "labels": task.label_ids,
        # Exact text overlap of the sample with train / calibration (calibration is carved from train only).
        "leakage": tasks.overlap(task, samples[seeds[0]]["text"]),
        "n": n, "n_requested": n_requested, "tier": tier_name(n, task.name),
        "seed": seeds[0], "seeds": seeds, "repeat": repeat,
        "sample_sha1": sample_sha1(samples[seeds[0]]),
        "sample_sha1_by_seed": {str(s): sample_sha1(df) for s, df in samples.items()},
        "environment": environment(),
        "apps": {},
    }
    print(f"Run folder: {run_dir}\nTask: {task.name} ({len(task.label_ids)} labels). "
          f"Sample: {n} items ({tier_name(n, task.name)} tier), seed(s) {', '.join(map(str, seeds))}"
          + (f", {repeat} runs per seed" if repeat > 1 else ""))

    def dest(app: App, s: int, rep: int) -> Path:
        base = run_dir / "apps" / app.name
        return base / f"seed{s}_rep{rep}" if multi else base

    # Apps not selected still appear in the report, from their latest results in the app folder.
    for app in APPS:
        entry = {"title": app.display_title, "runs_on": app.runs_on, "confidence": app.confidence,
                 "supervision": app.supervision_on(task.name)}
        runs: list[dict] = []
        if (args.report_only or app not in selected) and (reason := task_limits(app, task)):
            entry.update(status="skipped", reason=reason)
        elif args.report_only or app not in selected:
            for s in seeds:
                run = {"seed": s, "rep": 1, "status": "existing",
                       "reason": "not run now: latest results from the app folder"}
                runs.append(take_result(app, run, n=n, seed=s, expected=samples[s], dest=dest(app, s, 1), run_dir=run_dir,
                                        task=task))
        elif reason := preflight(app, n, len(seeds) * repeat, args.timeout, args.allow_paid, latency_requests, task):
            print(f"- {app.name}: skipped ({reason})")
            entry.update(status="skipped", reason=reason)
        else:
            managed = local_server(app, run_dir / "logs")
            server_ctx = managed or contextlib.nullcontext()
            try:
                with server_ctx as server:
                    for s in seeds:
                        for rep in range(1, repeat + 1):
                            label = f" (seed {s}, run {rep}/{repeat})" if multi else ""
                            print(f"- {app.name}{label}: running ...", flush=True)
                            log = run_dir / "logs" / (f"{app.name}_seed{s}_rep{rep}.log" if multi else f"{app.name}.log")
                            run = {"seed": s, "rep": rep, "log": str(log.relative_to(run_dir)),
                                   **run_app(app, n, s, args.timeout, log, stamp, extra_args,
                                             server, task.name)}
                            if app.ollama_model:  # model memory lives in the Ollama server, not the app process
                                tag = ollama_model_tag(app)
                                run["server_memory"] = ollama_loaded(tag, ollama_host(app))
                                tags = ollama_tags(ollama_host(app)) or {}
                                run["model_disk_bytes"] = (tags.get(tag) or tags.get(f"{tag}:latest") or {}).get("size")
                            if server is not None and server.start_seconds is not None:
                                run["server_start_seconds"] = server.start_seconds
                            if run["status"] == "ok":
                                run = take_result(app, run, n=n, seed=s, expected=samples[s], dest=dest(app, s, rep),
                                                  run_dir=run_dir, run_id=stamp, task=task)
                            print(f"  {run['status']} in {run['wall_seconds']:.0f} s"
                                  + (f": {run['reason']}" if run.get("reason") else ""))
                            runs.append(run)
            except RuntimeError as e:
                entry.update(status="failed", reason=str(e))
            finally:
                if app.ollama_model:
                    ollama_stop(ollama_model_tag(app), ollama_host(app))  # free memory for the next app
                if app.extra_ollama:
                    ollama_stop(setting(*app.extra_ollama))
        for run in runs:
            run.pop("started", None)
        if runs:
            # The report ranks the first seed's first valid run; the others feed the repeated-run statistics.
            first_seed = [r for r in runs if r["seed"] == seeds[0]]
            primary = next((r for r in first_seed if r["status"] in ("ok", "existing")), first_seed[0])
            entry.update({k: v for k, v in primary.items() if k not in ("seed", "rep")})
            if multi:
                entry["runs"] = runs
        summary["apps"][app.name] = entry

    # Virtual baselines on the shared sample (not executed model apps).
    summary["apps"].update(write_virtual_baselines(run_dir, samples[seeds[0]], seeds[0], task))

    summary["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    latest, latest_full = latest_report_path(task.name), latest_full_report_path(task.name)
    report = build_report(run_dir, also=((latest, latest_full),))  # each copy gets its own relative links
    print(f"\nReport: {report} (summary: {report.with_name('report_simple.html')})"
          f"\nLatest: {latest} (full raw report: {latest_full})")


if __name__ == "__main__":
    main()

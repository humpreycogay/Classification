"""Timing, latency, request statistics and model-size helpers shared by every app.

Part of the shared benchmark library (shared/), imported by every app from its own .venv.

    timed_calls   run the main classification call by call, keeping per-call durations
    Progress      "progress: 1,200/7,600 (15.8%)" lines with elapsed time and an estimate of the time left
    unseen_texts  test articles outside the sample, so latency runs never repeat an article the
                  main run already sent (servers such as Ollama cache repeated prompts)
    interactive   one article per call: end-to-end latency as a single user would see it
    sweep         throughput / latency for several batch sizes or concurrency levels
    percentiles   p50 / p90 / p95 / p99, each only when there are enough samples
    RequestStats  per-request latency, retries, failures, timeouts and tokens for HTTP apps
    GpuProbe      in-process GPU memory: MLX peak, or PyTorch MPS sampled after each call
    hf_model_bytes  size of a Hugging Face model in the local cache (download size)
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

LATENCY_ITEMS = 100  # articles timed one at a time for interactive latency
SWEEP_ITEMS = 128  # articles per sweep point (fresh articles for every point)
AGREEMENT_ITEMS = 32  # shared articles, classified untimed at every point, to check predictions do not change
SWEEP_VALUES = {"batch_size": [1, 8, 16, 32], "concurrency": [1, 2, 4, 8]}
SWEEP_TEXTS = SWEEP_ITEMS * 4  # unseen articles one sweep needs
# Smallest sample for which each percentile is reported (p99 of 100 samples is just the maximum).
MIN_SAMPLES = {"p50": 1, "p90": 20, "p95": 100, "p99": 1000}


def percentiles(samples: list[float]) -> dict:
    """count, mean, max and the percentiles the sample size supports (others None)."""
    xs = sorted(samples)
    out: dict = {"count": len(xs), "mean": statistics.fmean(xs) if xs else None, "max": xs[-1] if xs else None}
    for name, minimum in MIN_SAMPLES.items():
        q = int(name[1:]) / 100
        if len(xs) >= minimum:
            pos = q * (len(xs) - 1)  # linear interpolation between order statistics
            lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
            out[name] = xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
        else:
            out[name] = None
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()}


def duration(seconds: float) -> str:
    """Seconds as a short human duration: 42s, 12m 03s, 1h 04m."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"


# Seconds between progress lines (BENCHMARK_PROGRESS_SECONDS; 0 = only the final line).
PROGRESS_SECONDS = float(os.environ.get("BENCHMARK_PROGRESS_SECONDS") or 30)


def write_status(path: Path | None, **fields) -> None:
    """Replace the run's status file (JSON) atomically, so a reader never sees half a file. None = no file."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({**fields, "pid": os.getpid(), "updated": time.time()}, indent=1))
    tmp.replace(path)


class Progress:
    """Prints "progress: <phase> 1,200/7,600 (15.8%) · 12m 03s elapsed · ~1h 04m left · 1.66 items/s" to stdout,
    at most every `every` seconds and once at 100%, so a long run shows how far it is (the runner echoes these
    lines from the app log). With `status`, it also rewrites that status file (fields `info` + the counts) on
    every line, for ./run status. Thread-safe: HTTP apps also report from a watcher thread while one call runs."""

    PREFIX = "progress:"

    def __init__(self, phase: str, total: int, every: float = PROGRESS_SECONDS, status: Path | None = None,
                 info: dict | None = None) -> None:
        self.phase, self.total, self.every = phase, total, every
        self.status, self.info = status, info or {}
        self.done = 0
        self.start = time.perf_counter()
        self.last_print = self.start
        self.lock = threading.Lock()
        self.write()

    def write(self, line: str | None = None) -> None:
        elapsed = time.perf_counter() - self.start
        write_status(self.status, **self.info, phase=self.phase, done=self.done, total=self.total,
                     elapsed_seconds=round(elapsed, 1), left_seconds=self.left(elapsed), line=line or self.line())

    def left(self, elapsed: float) -> float | None:
        """Estimated seconds left at the average rate so far (None before the first item and when done)."""
        return round(elapsed / self.done * (self.total - self.done), 1) if 0 < self.done < self.total else None

    def line(self) -> str:
        elapsed = time.perf_counter() - self.start
        share = f" ({self.done / self.total:.1%})" if self.total else ""
        parts = [f"{self.PREFIX} {self.phase} {self.done:,}/{self.total:,}{share}", f"{duration(elapsed)} elapsed"]
        if (left := self.left(elapsed)) is not None:
            parts.append(f"~{duration(left)} left")
        if self.done and elapsed > 0:
            parts.append(f"{self.done / elapsed:.2f} items/s")
        return " · ".join(parts)

    def update(self, done: int) -> None:
        """Record that `done` items are finished (never goes backwards); print if due, or at 100%."""
        if not self.total:
            return
        with self.lock:
            if done <= self.done:
                return
            self.done = min(done, self.total)
            now = time.perf_counter()
            if self.done == self.total or (self.every and now - self.last_print >= self.every):
                self.last_print = now
                line = self.line()
                print(line, flush=True)
                self.write(line)

    @contextmanager
    def watch(self, completed: Callable[[], int], interval: float = 1.0):
        """While the block runs, poll completed() (items finished so far, e.g. an HTTP client's successful
        requests) every `interval` seconds; items done = completed() - its value at the start."""
        base = completed()
        stop = threading.Event()

        def poll() -> None:
            while not stop.wait(interval):
                self.update(completed() - base)

        thread = threading.Thread(target=poll, name="progress", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()


def timed_calls(run: Callable[[list[str]], list], texts: list[str], step: int | None,
                probe: "GpuProbe | None" = None, progress: Progress | None = None,
                completed: Callable[[], int] | None = None) -> tuple[list, list[float]]:
    """Classify texts in calls of `step` articles (None = one call), timing each call.

    progress gets the number of items done after every call. Apps that classify the whole sample in one call
    (the HTTP apps, at their concurrency) can also pass completed(), a count of items finished so far, which
    is polled during a call so their progress shows too."""
    preds, seconds = [], []
    chunks = [texts] if not step else [texts[i:i + step] for i in range(0, len(texts), step)]
    # Poll only when the whole sample is one call; with several calls, the per-call updates already count.
    watching = progress.watch(completed) if progress and completed and not step else nullcontext()
    with watching:
        for chunk in chunks:
            start = time.perf_counter()
            preds += run(chunk)
            seconds.append(time.perf_counter() - start)
            if probe:
                probe.sample()
            if progress:
                progress.update(len(preds))
    return preds, seconds


def unseen_texts(full, sample, k: int, seed: int) -> tuple[list[str], bool]:
    """k test articles not in the sample (shuffled with seed); (texts, reused) where reused = some had to
    come from the sample because too few articles are left (full tier)."""
    pool = full.drop(index=sample["sample_id"], errors="ignore").sample(frac=1, random_state=seed)["text"].tolist()
    if len(pool) >= k:
        return pool[:k], False
    return pool + sample["text"].tolist()[: k - len(pool)], True


def interactive(run_one: Callable[[str], object], texts: list[str], progress: Progress | None = None) -> list[float]:
    """End-to-end seconds for each article classified on its own (batch 1 / concurrency 1)."""
    seconds = []
    for text in texts:
        start = time.perf_counter()
        run_one(text)
        seconds.append(time.perf_counter() - start)
        if progress:
            progress.update(len(seconds))
    return seconds


def sweep(run: Callable[[list[str], int], list], axis: str, texts: list[str]) -> list[dict]:
    """For each batch size / concurrency: throughput and per-call latency on its own SWEEP_ITEMS articles
    (fresh for every setting, so no server-side prompt cache helps a later setting), plus agreement with
    the first setting (batch 1 / concurrency 1) on a shared, untimed set of AGREEMENT_ITEMS articles."""
    values = SWEEP_VALUES[axis]
    slices = [texts[i * SWEEP_ITEMS:(i + 1) * SWEEP_ITEMS] for i in range(len(values))]
    shared = slices[0][:AGREEMENT_ITEMS]
    rows, reference = [], None
    for value, chunk_texts in zip(values, slices):
        step = value if axis == "batch_size" else None
        start = time.perf_counter()
        preds, calls = timed_calls(lambda chunk: run(chunk, value), chunk_texts, step)
        total = time.perf_counter() - start
        if reference is None:  # the first setting's predictions on the shared set come from its timed run
            reference = [p.label for p in preds[:len(shared)]]
            labels = reference
        else:
            labels = [p.label for p in run(shared, value)]  # untimed
        rows.append({"axis": axis, "value": value, "items": len(chunk_texts), "seconds": round(total, 4),
                     "items_per_second": round(len(chunk_texts) / total, 3),
                     "per_call": percentiles(calls) if step else None,
                     "agreement_with_first": round(sum(a == b for a, b in zip(labels, reference)) / len(shared), 4),
                     "agreement_items": len(shared)})
        print(f"  sweep {axis}={value}: {len(chunk_texts) / total:.1f} items/s", flush=True)
    return rows


@dataclass
class RequestStats:
    """Counters for an HTTP client; per-request seconds are measured inside the concurrency limit."""
    retries_observable: bool = True
    requests: int = 0
    retries: int = 0
    failures: int = 0
    timeouts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_reported: int = 0  # requests whose response carried token counts
    generation_seconds: float = 0.0  # server-reported generation time (Ollama eval_duration)
    request_seconds: list[float] = field(default_factory=list)

    def reset(self) -> None:
        self.__init__(retries_observable=self.retries_observable)

    def add_tokens(self, input_tokens: int | None, output_tokens: int | None, generation_seconds: float | None = None) -> None:
        if input_tokens is None and output_tokens is None:
            return
        self.tokens_reported += 1
        self.input_tokens += input_tokens or 0
        self.output_tokens += output_tokens or 0
        self.generation_seconds += generation_seconds or 0.0

    def summary(self) -> dict:
        tokens = self.tokens_reported > 0
        return {
            "requests": self.requests,
            "retries": self.retries if self.retries_observable else None,
            "retries_observable": self.retries_observable,
            "failures": self.failures,
            "timeouts": self.timeouts,
            "input_tokens": self.input_tokens if tokens else None,
            "output_tokens": self.output_tokens if tokens else None,
            "tokens_reported_by": f"{self.tokens_reported}/{self.requests} requests",
            "output_tokens_per_second": (round(self.output_tokens / self.generation_seconds, 2)
                                         if tokens and self.generation_seconds else None),
            "per_request": percentiles(self.request_seconds),
        }


class GpuProbe:
    """In-process GPU memory. MLX reports its own peak; PyTorch MPS has no peak counter, so the
    driver allocation is sampled after each call (a lower bound on the true peak)."""

    def __init__(self) -> None:
        self.kind, self.sampled_peak = None, 0
        if "mlx.core" in sys.modules:
            self.kind = "mlx"
            mx = sys.modules["mlx.core"]
            reset = getattr(mx, "reset_peak_memory", None) or getattr(getattr(mx, "metal", None), "reset_peak_memory", None)
            if reset:
                reset()
        elif "torch" in sys.modules:
            torch = sys.modules["torch"]
            if torch.backends.mps.is_available():
                self.kind = "mps"

    def sample(self) -> None:
        if self.kind == "mps":
            self.sampled_peak = max(self.sampled_peak, int(sys.modules["torch"].mps.driver_allocated_memory()))

    def result(self) -> dict | None:
        if self.kind == "mlx":
            mx = sys.modules["mlx.core"]
            peak = getattr(mx, "get_peak_memory", None) or getattr(getattr(mx, "metal", None), "get_peak_memory", None)
            return {"kind": "mlx", "peak_bytes": int(peak()), "method": "MLX peak memory"} if peak else None
        if self.kind == "mps":
            return {"kind": "mps", "peak_bytes": self.sampled_peak,
                    "method": "PyTorch MPS driver allocation, sampled after each call (lower bound)"}
        return None


def hf_model_bytes(repo_id: str | None) -> int | None:
    """Bytes of a model in the local Hugging Face cache (all downloaded files), or None if not cached."""
    if not repo_id or "/" not in repo_id:
        return None
    home = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    blobs = home / f"models--{repo_id.replace('/', '--')}" / "blobs"
    if not blobs.is_dir():
        return None
    return sum(f.stat().st_size for f in blobs.iterdir() if f.is_file())

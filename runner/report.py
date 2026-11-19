"""Build the HTML comparison report for one run folder (reports/runs/<timestamp>/).

Reads summary.json plus each app's predictions CSV and run metadata, computes every metric
from the predictions (so all apps are scored identically), and writes report.html.

Only results on the reference sample are compared: an app whose texts, labels or sample_sha1
differ is left out of rankings, significance tests, charts and tiles and listed separately.
Results the runner marked invalid are reported as failures. With several seeds or repeats
(summary "runs") a repeated-runs section adds mean / median / SD / IQR and prediction stability.

Comparisons (stats.py): exact McNemar tests of every model against the top point estimate with
Holm-adjusted p-values, paired accuracy differences with Tango score and paired-bootstrap 95%
intervals, bootstrap intervals for macro F1 / AUROC / ECE / Brier, bootstrap rank intervals and a
leading tier of models whose rank interval includes 1, separate quality / latency / resource / cost rankings and an
accuracy-vs-latency Pareto frontier. Bootstrap settings are written to report_meta.json.

Selective prediction and calibration (selective.py) are tie-aware: AURC / E-AURC, coverage at
target accuracy (optimistic and Wilson lower bound), accuracy at fixed coverage, reliability diagrams,
MCE over well-populated bins, classwise ECE, top-2 accuracy and calibration warnings.

Speed, memory and cost (perf.py): run configuration, per-call and interactive latency percentiles,
batch / concurrency sweeps, peak memory (app process, in-process GPU, server), model size, tokens,
retries, cost with its basis, measurement conditions (power source), and Pareto frontiers of
accuracy against throughput, interactive latency and memory.

Usage:
    python report.py runs/<timestamp>
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

import perf
import shared_path  # noqa: F401  (puts shared/ on sys.path)
import task as tasks
from apps import APPROACHES, APPS, APPS_BY_NAME, CONFIDENCE_KINDS, confidence_kind
from simple_report import FULL_HREF, simple_page
from selective import (COVERAGES, MCE_MIN_BIN, MIN_N_FOR_99, SATURATED, TARGETS, calibration_warnings,
                       held_out_at_target, selective_metrics)
from stats import (ALPHA, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, bootstrap_metrics, holm, leading_tier, rank_intervals,
                   paired_difference, pareto_front, percentile_ci, resample_weights, wilson)

# The run's task labels. Set by build_report from summary.json (default: AG News); every metric
# function reads these module-level lists, so one report is always about one task.
LABELS = tasks.load_task().label_ids
PROB_COLS = [f"p_{label}" for label in LABELS]
THRESHOLDS = [0.5, 0.7, 0.9]
# Fallback for summary.json files written before the runner recorded each app's supervision type.
APP_SUPERVISION = {a.name: a.supervision for a in APPS}


# ---------------------------------------------------------------- metrics

def auroc(conf: pd.Series, correct: pd.Series) -> float:
    """Probability that a correct prediction has higher confidence than a wrong one (ties count half)."""
    n_pos, n_neg = int(correct.sum()), int((~correct).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = conf.rank(method="average")
    return (ranks[correct].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def ece(conf: pd.Series, correct: pd.Series, bins: int = 10) -> float:
    idx = (conf.clip(0, 1) * bins).astype(int).clip(upper=bins - 1)
    total = 0.0
    for _, group in pd.DataFrame({"b": idx, "c": conf, "y": correct}).groupby("b"):
        total += abs(group["y"].mean() - group["c"].mean()) * len(group) / len(conf)
    return total


def per_item(meta: dict) -> float | None:
    """Classification seconds per item; recomputed when the stored value was rounded to 0 (sub-0.1 ms apps)."""
    if not meta:
        return None
    if meta.get("seconds_per_item"):
        return meta["seconds_per_item"]
    return max(meta["classify_seconds"], 1e-6) / meta["n"]


def training_seconds(meta: dict) -> float | None:
    """Train embedding + fit time of a supervised baseline; None for apps that are not trained."""
    tr = meta.get("training")
    if not tr:
        return None
    return (tr.get("train_embed_seconds") or 0) + (tr.get("train_seconds") or 0)


def frame_sha1(df: pd.DataFrame) -> str:
    """Sample identity of a predictions CSV (same definition as the apps' sample_sha1)."""
    return hashlib.sha1("\n".join(df["label"] + "\t" + df["text"]).encode()).hexdigest()


def spread(values: list[float]) -> dict:
    """Mean, median, sample SD and interquartile range (Q3 - Q1) of repeated measurements."""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return {"k": 0, "mean": float("nan"), "median": float("nan"), "sd": float("nan"), "iqr": float("nan")}
    s = pd.Series(vals, dtype=float)
    return {"k": len(vals), "mean": float(s.mean()), "median": float(s.median()),
            "sd": float(s.std(ddof=1)) if len(vals) > 1 else float("nan"),
            "iqr": float(s.quantile(0.75) - s.quantile(0.25))}


def prediction_stability(preds: list[pd.Series]) -> tuple[float, float]:
    """For runs on the same sample: share of articles with the same prediction in every run, and
    the mean share of articles on which two runs agree (over all pairs)."""
    frame = pd.concat([p.reset_index(drop=True) for p in preds], axis=1, keys=range(len(preds)))
    unanimous = float((frame.nunique(axis=1) == 1).mean())
    pairs = [(i, j) for i in range(len(preds)) for j in range(i + 1, len(preds))]
    pairwise = statistics.mean(float((frame[i] == frame[j]).mean()) for i, j in pairs)
    return unanimous, pairwise


def app_metrics(df: pd.DataFrame) -> dict:
    n = len(df)
    correct = df["label"] == df["pred"]
    k = int(correct.sum())
    lo, hi = wilson(k, n)
    per_class = {}
    for label in LABELS:
        tp = int(((df["pred"] == label) & (df["label"] == label)).sum())
        precision = tp / max(int((df["pred"] == label).sum()), 1)
        recall = tp / max(int((df["label"] == label).sum()), 1)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
    confusion = pd.crosstab(df["label"], df["pred"]).reindex(index=LABELS, columns=LABELS, fill_value=0)

    conf = df["confidence"].astype(float)
    gating = []
    for t in THRESHOLDS:
        kept = correct[conf >= t]
        gating.append({"t": t, "coverage": len(kept) / n, "accuracy": kept.mean() if len(kept) else float("nan")})
    m = {
        "n": n, "correct": k, "accuracy": k / n, "ci": (lo, hi),
        "macro_f1": sum(c["f1"] for c in per_class.values()) / len(LABELS),
        "per_class": per_class, "confusion": confusion,
        "conf_mean_correct": conf[correct].mean() if k else float("nan"),
        "conf_mean_wrong": conf[~correct].mean() if k < n else float("nan"),
        "gating": gating,
        "auroc": auroc(conf, correct), "ece": ece(conf, correct),
        "has_probs": all(c in df.columns for c in PROB_COLS),
    }
    if m["has_probs"]:
        probs = df[PROB_COLS].astype(float).clip(lower=0)
        probs = probs.div(probs.sum(axis=1).replace(0, 1), axis=0)
        onehot = pd.get_dummies(df["label"]).reindex(columns=LABELS, fill_value=0).astype(float).to_numpy()
        m["brier"] = float(((probs.to_numpy() - onehot) ** 2).sum(axis=1).mean())
        p_true = (probs.to_numpy() * onehot).sum(axis=1).clip(1e-6, 1)
        m["log_loss"] = float(-pd.Series(p_true).apply(math.log).mean())
    m.update(selective_metrics(df, LABELS))  # tie-aware curve, AURC, targets, calibration, top-2
    return m


def calibration_frame(run_dir: Path, entry: dict, eval_df: pd.DataFrame, eval_seed: int) -> tuple[pd.DataFrame, int] | None:
    """Another seed's predictions from the same app, minus any item also in the evaluation sample.

    Used to choose selective-prediction thresholds on items the evaluation never scores. Returns (frame, seed), or
    None when the run has no other seed with valid results.
    """
    for r in entry.get("runs", []):
        if r.get("seed") != eval_seed and r.get("status") in ("ok", "existing") and r.get("csv"):
            path = run_dir / r["csv"]
            if not path.exists():
                continue
            df = pd.read_csv(path)
            key = "sample_id" if "sample_id" in df.columns and "sample_id" in eval_df.columns else "text"
            df = df[~df[key].isin(set(eval_df[key]))]
            if len(df):
                return df, r["seed"]
    return None


def apply_held_out(m: dict, eval_df: pd.DataFrame, cal: tuple[pd.DataFrame, int] | None) -> None:
    """Replace the cross-fitted held-out results with thresholds fitted on a separate calibration sample."""
    if cal is None:
        return
    cal_df, seed = cal
    cal_conf, cal_right = cal_df["confidence"].astype(float), cal_df["label"] == cal_df["pred"]
    conf, right = eval_df["confidence"].astype(float), eval_df["label"] == eval_df["pred"]
    m["held_out"] = {t: {**held_out_at_target(cal_conf, cal_right, conf, right, t), "calibration_seed": seed}
                     for t in TARGETS}


def held_out_txt(r: dict) -> str:
    """How a held-out result's threshold was chosen, in a few words."""
    if r.get("method") == "held-out":
        return f"threshold fitted on seed {r['calibration_seed']} (n = {r['n_calibration']}), scored on this sample"
    return f"{r.get('folds', 2)}-fold cross-fitted: each half scored with the threshold fitted on the other half"


# ---------------------------------------------------------------- formatting

def esc(value) -> str:
    return html.escape(str(value))


def pct(x: float, digits: int = 1) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{digits}f}%"


def num(x, digits: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def secs(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if 0 < x < 0.01:
        return f"{x * 1000:.2f} ms"
    if x < 1:
        return f"{x * 1000:.0f} ms"
    if x < 120:
        return f"{x:.1f} s"
    return f"{x / 60:.1f} min"


def ci_txt(ci: tuple[float, float], fmt) -> str:
    lo, hi = ci
    return "–" if lo is None or math.isnan(lo) else f"{fmt(lo)} – {fmt(hi)}"


def signed_pct(x: float) -> str:
    return "–" if x is None or math.isnan(x) else f"{x * 100:+.1f} pp"


def p_cell(p: float | None, alpha: float = 0.05) -> str:
    """A p-value, highlighted when significant."""
    if p is None:
        return "–"
    return f"<span class='{'bad' if p < alpha else 'sig'}'>{p:.3f}</span>"


def size_txt(b) -> str:
    if b is None or (isinstance(b, float) and math.isnan(b)):
        return "–"
    for unit, scale in (("GB", 2**30), ("MB", 2**20), ("KB", 2**10)):
        if b >= scale:
            return f"{b / scale:.1f} {unit}"
    return f"{int(b)} B"


def money_txt(c: dict, key: str) -> str:
    """A cost figure with its currency, or – when unknown."""
    v = c.get(key)
    return "–" if v is None else f"{v:.4g} {c.get('currency') or ''}".strip()


def spread_cells(s: dict, fmt) -> str:
    return (f"<td>{fmt(s['mean'])}</td><td>{fmt(s['median'])}</td><td>{fmt(s['sd'])}</td>"
            f"<td>{fmt(s['iqr'])}</td>")


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_HREF = "{{MODELS_HREF}}"  # replaced per output file by the relative path to models/ (see write_page)

# Task-specific context for the task card: why the task is hard in its own way.
TASK_NOTES = {
    "ag_news": "Four broad news topics with long, information-rich texts. A coarse topic task: results here do not "
               "transfer to fine-grained intent routing such as Banking77.",
    "banking77": "77 fine-grained intents in short messages, many of them near-duplicates. Listing every label in a "
                 "prompt stresses context budgets, and models with a fixed option limit need a shortlist stage first; "
                 "a shortlist caps accuracy at its recall.",
}


def callout(reading: str, caution: str, formula: str = "") -> str:
    """A one-sentence interpretation and one caution beside an advanced metric; formula in a disclosure."""
    return (f"<p class='callout'><b>Reading it:</b> {reading} <b>Caution:</b> {caution}</p>"
            + (f"<details class='formula'><summary>Formula</summary><p class='note'>{formula}</p></details>" if formula else ""))


def task_card(task, summary: dict, sample: pd.DataFrame | None, primary: str, noun: str) -> str:
    """What the task looks like: one example input, label count and examples, split sizes, sampling, primary metric."""
    m = task.manifest
    try:
        train_n = f"{len(tasks.load_split(task, 'train')):,}"
    except Exception:  # noqa: BLE001 (the card is informative only; a missing split must not break the report)
        train_n = "–"
    try:
        test_n = f"{tasks.test_size(task):,}"
    except Exception:  # noqa: BLE001
        test_n = "–"
    names = task.label_names
    example_labels = ", ".join(f"<i>{esc(names[i])}</i>" for i in task.label_ids[:3])
    example = ""
    if sample is not None and len(sample):
        row = sample.iloc[0]
        text = str(row["text"])
        text = text if len(text) <= 240 else text[:237] + "…"
        example = (f"<dt>Example {esc(noun)}</dt><dd>“{esc(text)}” → <i>{esc(names.get(row['label'], row['label']))}</i>"
                   "</dd>")
    why = ("labels are balanced in the sample, so accuracy and macro F1 tell a similar story" if primary == "accuracy"
           else "the labels are imbalanced, so accuracy would mostly reflect the frequent labels")
    sampling = m.get("sampling", "balanced")
    rows = (f"<dt>Labels</dt><dd>{len(task.label_ids)} ({example_labels}, …)</dd>"
            f"<dt>Splits</dt><dd>train {train_n}, test {test_n}; this report evaluates {summary['n']:,} test {esc(noun)}s</dd>"
            f"<dt>Sampling</dt><dd>{esc(sampling)} stratified sample of the test split</dd>"
            f"<dt>Primary metric</dt><dd>{esc(primary.replace('_', ' '))}: {why}</dd>"
            f"<dt>License</dt><dd>{esc(m.get('license', 'not stated'))}</dd>")
    note = TASK_NOTES.get(task.name)
    return (f"<section class='card taskcard' aria-label='Task'><h2 class='tc'>Task: {esc(task.name)}</h2>"
            f"<p class='note'>{esc(m.get('description', ''))}</p><dl>{example}{rows}</dl>"
            + (f"<p class='note'><b>Context:</b> {esc(note)}</p>" if note else "") + "</section>")


def back_link() -> str:
    return "<p class='back'><a href='#overview'>↑ Back to overview</a></p>"


# Servers on the local network (Ollama on another machine, a GPU box): their addresses are not published in a report.
PRIVATE_HOST = re.compile(r"(?<![\w.])(?:https?://)?(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}"
                          r"|[A-Za-z0-9-]+\.(?:local|lan|home|internal|ts\.net))(?::\d{1,5})?(?:/[^\s<'\"]*)?")


def redact_hosts(page: str) -> str:
    """Replace private-network server addresses with a placeholder (localhost stays: it says nothing)."""
    return PRIVATE_HOST.sub("a LAN server", page)


SIMPLE_HREF = "{{SIMPLE_HREF}}"  # the summary report, linked from the full one (see write_page)


def write_page(page: str, path: Path, refs: dict[str, Path] | None = None) -> None:
    """Write a report page to path with model-page links (and each placeholder in refs) relative to that file."""
    rel = lambda target: Path(os.path.relpath(target, path.parent)).as_posix()  # noqa: E731
    page = page.replace(MODELS_HREF, rel(MODELS_DIR))
    for placeholder, target in (refs or {}).items():
        page = page.replace(placeholder, rel(target))
    path.write_text(redact_hosts(page))


# ---------------------------------------------------------------- charts (inline SVG, single hue)

def forest_plot(rows: list[dict], ref: str, width: int = 900, label_w: int = 360) -> str:
    """Paired accuracy differences vs the reference model with 95% intervals and a zero line.

    rows: name, diff, low, high (fractions). Intervals that include 0 use a hollow marker (shape, not only colour).
    """
    row_h, top, bottom = 30, 8, 46
    plot_w = width - label_w - 130
    height = top + row_h * len(rows) + bottom
    lo = min([r["low"] for r in rows] + [0.0])
    hi = max([r["high"] for r in rows] + [0.0])
    span = max(hi - lo, 0.01)
    lo, hi = lo - span * 0.05, hi + span * 0.05
    x = lambda v: label_w + (v - lo) / (hi - lo) * plot_w  # noqa: E731
    step = next(s for s in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5) if (hi - lo) / s <= 8)
    ticks = [i * step for i in range(math.ceil(lo / step), math.floor(hi / step) + 1)]
    parts = [f'<svg class="chart forest" viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="Paired accuracy difference of each app versus {esc(ref)}, with 95% intervals" '
             'preserveAspectRatio="xMinYMin meet">']
    for t in ticks:
        parts.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{height - bottom}"/>')
        parts.append(f'<text class="tick" x="{x(t):.1f}" y="{height - bottom + 16}" text-anchor="middle">{t * 100:+.0f}</text>')
    parts.append(f'<line class="zero" x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{top}" y2="{height - bottom}"/>')
    parts.append(f'<text class="axis" x="{label_w + plot_w / 2:.1f}" y="{height - 6}" text-anchor="middle">'
                 f'accuracy difference vs {esc(ref)} (percentage points)</text>')
    for i, r in enumerate(rows):
        cy = top + row_h * i + row_h / 2
        crosses = r["low"] <= 0 <= r["high"]
        tip = (f"{r['name']}: {r['diff'] * 100:+.1f} pp vs {ref}, 95% score CI {r['low'] * 100:+.1f} to "
               f"{r['high'] * 100:+.1f} pp" + ("; the interval includes 0" if crosses else "; the interval excludes 0"))
        parts.append(f'<g class="row" tabindex="0" data-tip="{esc(tip)}">')
        parts.append(f'<rect class="hit" x="0" y="{cy - row_h / 2:.1f}" width="{width}" height="{row_h}"/>')
        parts.append(f'<text class="rowlabel" x="{label_w - 12}" y="{cy + 4:.1f}" text-anchor="end">{esc(r["name"])}</text>')
        parts.append(f'<line class="whisker" x1="{x(r["low"]):.1f}" x2="{x(r["high"]):.1f}" y1="{cy:.1f}" y2="{cy:.1f}"/>')
        parts.append(f'<circle class="{"dot open" if crosses else "dot"}" cx="{x(r["diff"]):.1f}" cy="{cy:.1f}" r="5.5"/>')
        parts.append(f'<text class="value" x="{label_w + plot_w + 14:.1f}" y="{cy + 4:.1f}">{r["diff"] * 100:+.1f}'
                     + (" (incl. 0)" if crosses else "") + "</text>")
        parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)


def dot_plot(rows: list[dict], *, lo: float, hi: float, fmt, log: bool = False, ticks: list[float],
             label_w: int = 360, width: int = 900, x_title: str = "") -> str:
    """Horizontal dot plot, one row per app, optional interval whisker. rows: name, value, low?, high?, tip."""
    row_h, top, bottom = 30, 8, 46 if x_title else 30
    plot_w = width - label_w - 70
    height = top + row_h * len(rows) + bottom

    def x(v: float) -> float:
        if log:
            v = min(max(v, lo), hi)
            return label_w + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * plot_w
        return label_w + (min(max(v, lo), hi) - lo) / (hi - lo) * plot_w

    label = x_title or "one row per app; values in the adjacent table"
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Dot plot: {esc(label)}" '
             'preserveAspectRatio="xMinYMin meet">']
    for t in ticks:
        parts.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{height - bottom}"/>')
        parts.append(f'<text class="tick" x="{x(t):.1f}" y="{height - bottom + 18}" text-anchor="middle">{esc(fmt(t))}</text>')
    for i, r in enumerate(rows):
        cy = top + row_h * i + row_h / 2
        parts.append(f'<g class="row" tabindex="0" data-tip="{esc(r["tip"])}">')
        parts.append(f'<rect class="hit" x="0" y="{cy - row_h / 2:.1f}" width="{width}" height="{row_h}"/>')
        parts.append(f'<text class="rowlabel" x="{label_w - 12}" y="{cy + 4:.1f}" text-anchor="end">{esc(r["name"])}</text>')
        if r.get("low") is not None:
            parts.append(f'<line class="whisker" x1="{x(r["low"]):.1f}" x2="{x(r["high"]):.1f}" y1="{cy:.1f}" y2="{cy:.1f}"/>')
        parts.append(f'<circle class="dot" cx="{x(r["value"]):.1f}" cy="{cy:.1f}" r="5.5"/>')
        parts.append(f'<text class="value" x="{x(r.get("high", r["value"])) + 12:.1f}" y="{cy + 4:.1f}">{esc(fmt(r["value"]))}</text>')
        parts.append("</g>")
    if x_title:
        parts.append(f'<text class="axis" x="{label_w + plot_w / 2:.1f}" y="{height - 6}" text-anchor="middle">{esc(x_title)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def coverage_multiple(name: str, m: dict, y_lo: float) -> str:
    """Small multiple: accuracy of the most-confident k% of predictions, k = coverage."""
    w, h, left, right, top, bottom = 240, 130, 34, 20, 10, 22
    pw, ph = w - left - right, h - top - bottom

    def x(c: float) -> float:
        return left + c * pw

    def y(a: float) -> float:
        return top + (1 - (max(a, y_lo) - y_lo) / (1 - y_lo)) * ph

    pts = [(c, a) for c, a in m["curve"] if c >= 0.05]
    path = "M" + " L".join(f"{x(c):.1f},{y(a):.1f}" for c, a in pts)
    parts = [f'<figure class="multiple"><figcaption>{esc(name)}</figcaption>',
             f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(name)}: accuracy of the kept '
             'predictions against the share kept">']
    for a in (y_lo, (y_lo + 1) / 2, 1.0):
        parts.append(f'<line class="grid" x1="{left}" x2="{w - right}" y1="{y(a):.1f}" y2="{y(a):.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 5}" y="{y(a) + 3:.1f}" text-anchor="end">{a * 100:.0f}</text>')
    for c in (0, 0.5, 1):
        parts.append(f'<text class="tick" x="{x(c):.1f}" y="{h - 6}" text-anchor="middle">{c * 100:.0f}%</text>')
    parts.append(f'<line class="baseline" x1="{left}" x2="{w - right}" y1="{y(m["accuracy"]):.1f}" y2="{y(m["accuracy"]):.1f}"/>')
    parts.append(f'<path class="line" d="{path}"/>')
    for g in m["gating"]:
        if g["coverage"] > 0:
            tip = f'{name}: confidence ≥ {g["t"]:.1f} keeps {pct(g["coverage"])} at {pct(g["accuracy"])} accuracy'
            cx, cy = x(g["coverage"]), y(g["accuracy"])
            parts.append(f'<g data-tip="{esc(tip)}"><circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="11"/>'
                         f'<circle class="marker" cx="{cx:.1f}" cy="{cy:.1f}" r="4.5"/></g>')
    parts.append("</svg></figure>")
    return "".join(parts)


def reliability_multiple(name: str, m: dict, warnings: list[str]) -> str:
    """Small multiple: accuracy vs mean confidence per bin (dot area ~ count), count bars below, diagonal = perfect."""
    w, h, left, right, top, bars = 240, 200, 30, 10, 8, 34
    pw = w - left - right
    ph = h - top - bars - 22

    def x(c: float) -> float:
        return left + c * pw

    def y(a: float) -> float:
        return top + (1 - a) * ph

    bins = m["reliability"]
    most = max(b["count"] for b in bins) or 1
    caption = esc(name) + (f" <span class='warnmark' data-tip='{esc('; '.join(warnings))}'>⚠</span>" if warnings else "")
    parts = [f'<figure class="multiple"><figcaption>{caption}</figcaption>',
             f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(name)}: reliability diagram, '
             'accuracy against mean confidence per bin">']
    for a in (0, 0.5, 1):
        parts.append(f'<line class="grid" x1="{left}" x2="{w - right}" y1="{y(a):.1f}" y2="{y(a):.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 5}" y="{y(a) + 3:.1f}" text-anchor="end">{a * 100:.0f}</text>')
    parts.append(f'<line class="baseline" x1="{x(0):.1f}" y1="{y(0):.1f}" x2="{x(1):.1f}" y2="{y(1):.1f}"/>')
    base = top + ph + 6 + bars
    for b in bins:
        bw = pw / len(bins)
        bh = (bars - 4) * b["count"] / most
        tip = (f"{name}: confidence {b['low']:.1f}–{b['high']:.1f}: {b['count']} predictions"
               + (f", mean confidence {b['confidence']:.2f}, accuracy {pct(b['accuracy'])}" if b["count"] else ""))
        parts.append(f'<g data-tip="{esc(tip)}"><rect class="hit" x="{x(b["low"]):.1f}" y="{top}" width="{bw:.1f}" '
                     f'height="{base - top:.1f}"/><rect class="bar" x="{x(b["low"]) + 1:.1f}" y="{base - bh:.1f}" '
                     f'width="{bw - 2:.1f}" height="{bh:.1f}"/>')
        if b["count"]:
            r = 2 + 5 * math.sqrt(b["count"] / most)
            parts.append(f'<circle class="marker" cx="{x(b["confidence"]):.1f}" cy="{y(b["accuracy"]):.1f}" r="{r:.1f}"/>')
        parts.append("</g>")
    for c in (0, 0.5, 1):
        parts.append(f'<text class="tick" x="{x(c):.1f}" y="{h - 6}" text-anchor="middle">{c:.1f}</text>')
    parts.append("</svg></figure>")
    return "".join(parts)


def sweep_multiple(name: str, pts: list[dict]) -> str:
    """Small multiple: throughput (y) vs latency per call (x, log) for each sweep value, joined in sweep order."""
    w, h, left, right, top, bottom = 240, 150, 40, 14, 10, 30
    pw, ph = w - left - right, h - top - bottom
    lat = [p["latency"] for p in pts]
    x_lo, x_hi = 10 ** math.floor(math.log10(min(lat))), 10 ** math.ceil(math.log10(max(lat)))
    if x_lo == x_hi:
        x_hi *= 10
    y_hi = max(p["items_per_second"] for p in pts) * 1.15

    def x(v: float) -> float:
        return left + (math.log10(v) - math.log10(x_lo)) / (math.log10(x_hi) - math.log10(x_lo)) * pw

    def y(v: float) -> float:
        return top + (1 - v / y_hi) * ph

    parts = [f'<figure class="multiple"><figcaption>{esc(name)}</figcaption>',
             f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(name)}: throughput against '
             'batch size or concurrency">']
    for e in range(round(math.log10(x_lo)), round(math.log10(x_hi)) + 1):
        parts.append(f'<line class="grid" x1="{x(10 ** e):.1f}" x2="{x(10 ** e):.1f}" y1="{top}" y2="{top + ph}"/>')
        parts.append(f'<text class="tick" x="{x(10 ** e):.1f}" y="{top + ph + 14}" text-anchor="middle">{esc(secs(10 ** e))}</text>')
    for v in (0, y_hi / 2):
        parts.append(f'<text class="tick" x="{left - 5}" y="{y(v) + 3:.1f}" text-anchor="end">{v:.0f}</text>')
    parts.append(f'<path class="line" d="M{" L".join(f"{x(p["latency"]):.1f},{y(p["items_per_second"]):.1f}" for p in pts)}"/>')
    for p in pts:
        tip = (f"{name}: {p['axis'].replace('_', ' ')} {p['value']}: {p['items_per_second']:.1f} items/s, "
               f"{secs(p['latency'])} per call ({p['latency_method']}), agreement {pct(p['agreement_with_first'])}")
        parts.append(f'<g data-tip="{esc(tip)}"><circle class="hit" cx="{x(p["latency"]):.1f}" cy="{y(p["items_per_second"]):.1f}" r="10"/>'
                     f'<circle class="marker" cx="{x(p["latency"]):.1f}" cy="{y(p["items_per_second"]):.1f}" r="4"/>'
                     f'<text class="tick" x="{x(p["latency"]):.1f}" y="{y(p["items_per_second"]) - 8:.1f}" text-anchor="middle">{p["value"]}</text></g>')
    parts.append(f'<text class="tick" x="{left + pw / 2:.1f}" y="{h - 2}" text-anchor="middle">latency per call → · items/s ↑</text>')
    parts.append("</svg></figure>")
    return "".join(parts)


def pareto_chart(points: dict[str, tuple[float, float]], front: list[str], names: dict[str, str],
                 x_label: str = "time per item (log scale), faster to the left", fmt=None) -> str:
    """Accuracy (y) vs a cost on x (log): frontier apps filled and joined, others muted."""
    fmt = fmt or secs
    width, height, left, right, top, bottom = 820, 380, 56, 190, 14, 40
    pw, ph = width - left - right, height - top - bottom
    accs, costs = [a for a, _ in points.values()], [c for _, c in points.values()]
    x_lo, x_hi = 10 ** math.floor(math.log10(min(costs))), 10 ** math.ceil(math.log10(max(costs)))
    if x_lo == x_hi:
        x_hi *= 10
    y_lo = max(0.0, math.floor(min(accs) * 20) / 20)
    y_hi = 1.0

    def x(c: float) -> float:
        return left + (math.log10(c) - math.log10(x_lo)) / (math.log10(x_hi) - math.log10(x_lo)) * pw

    def y(a: float) -> float:
        return top + (1 - (a - y_lo) / max(y_hi - y_lo, 1e-9)) * ph

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Pareto chart: accuracy '
             f'against {esc(x_label)}; frontier apps are named in the text and table">']
    for e in range(round(math.log10(x_lo)), round(math.log10(x_hi)) + 1):
        parts.append(f'<line class="grid" x1="{x(10 ** e):.1f}" x2="{x(10 ** e):.1f}" y1="{top}" y2="{top + ph}"/>')
        parts.append(f'<text class="tick" x="{x(10 ** e):.1f}" y="{top + ph + 16}" text-anchor="middle">{esc(fmt(10 ** e))}</text>')
    step = 0.05 if y_hi - y_lo <= 0.3 else 0.1
    for i in range(round((y_hi - y_lo) / step) + 1):
        a = y_lo + i * step
        parts.append(f'<line class="grid" x1="{left}" x2="{left + pw}" y1="{y(a):.1f}" y2="{y(a):.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 6}" y="{y(a) + 4:.1f}" text-anchor="end">{a * 100:.0f}%</text>')
    parts.append(f'<text class="tick" x="{left + pw / 2:.1f}" y="{height - 4}" text-anchor="middle">{esc(x_label)}</text>')
    if len(front) > 1:
        path = "M" + " L".join(f"{x(points[k][1]):.1f},{y(points[k][0]):.1f}" for k in front)
        parts.append(f'<path class="line frontier" d="{path}"/>')
    for name, (acc, cost) in sorted(points.items(), key=lambda kv: kv[0] in front):
        on = name in front
        tip = f"{names[name]}: {pct(acc)} accuracy, {fmt(cost)}" + (" (Pareto frontier)" if on else "")
        parts.append(f'<g data-tip="{esc(tip)}"><circle class="hit" cx="{x(cost):.1f}" cy="{y(acc):.1f}" r="11"/>'
                     f'<circle class="{"dot" if on else "dot dim"}" cx="{x(cost):.1f}" cy="{y(acc):.1f}" '
                     f'r="{5.5 if on else 4.5}"/></g>')
        if on:
            parts.append(f'<text class="rowlabel" x="{x(cost) + 9:.1f}" y="{y(acc) - 7:.1f}">{esc(names[name])}</text>')
    parts.append("</svg>")
    return "".join(parts)


# Accuracy-vs-speed scatter: every classifier has its own mark, a hue (8 validated categorical slots, fixed order)
# combined with a shape. More than 3 hues cannot all be told apart in a scatter by color alone, so the shape, the
# legend (hover a name to highlight its point) and the tooltip carry identity too. Marks follow the app's position
# in the runner registry, never its rank, so a classifier looks the same in every report.
SCATTER_SHAPES = ("circle", "square", "triangle", "diamond")


def scatter_mark(style: int, cx: float, cy: float, r: float = 5.5) -> str:
    """The SVG mark for a style index: hue = style mod 8, shape = style div 8."""
    hue, shape = style % 8 + 1, SCATTER_SHAPES[style // 8 % len(SCATTER_SHAPES)]
    fill = f'class="mk" style="fill:var(--c{hue})"'
    if shape == "circle":
        return f'<circle {fill} cx="{cx:.1f}" cy="{cy:.1f}" r="{r}"/>'
    if shape == "square":
        a = r * 0.9
        return f'<rect {fill} x="{cx - a:.1f}" y="{cy - a:.1f}" width="{2 * a:.1f}" height="{2 * a:.1f}" rx="1.5"/>'
    if shape == "triangle":
        a = r * 1.25
        return f'<path {fill} d="M{cx:.1f},{cy - a:.1f} L{cx + a * 0.95:.1f},{cy + a * 0.7:.1f} L{cx - a * 0.95:.1f},{cy + a * 0.7:.1f} Z"/>'
    a = r * 1.3
    return f'<path {fill} d="M{cx:.1f},{cy - a:.1f} L{cx + a:.1f},{cy:.1f} L{cx:.1f},{cy + a:.1f} L{cx - a:.1f},{cy:.1f} Z"/>'


LABEL_CHAR_W, LABEL_H = 6.4, 12  # estimated width per character and height of an 11px scatter label


def place_labels(marks: list[tuple[str, float, float]], bounds: tuple[float, float, float, float],
                 r: float = 7.0) -> dict[str, tuple[float, float, str]]:
    """Greedy label placement: each label (key, mark x, mark y) goes beside its mark, right or left, as close to the
    mark's height as it can without covering another label or mark; farther away (up to 5 rows) if it must.
    Returns key -> (text x, text baseline y, anchor). bounds = (x0, y0, x1, y1) of the plot area."""
    x0, y0, x1, y1 = bounds
    taken = [(cx - r, cy - r, cx + r, cy + r) for _, cx, cy in marks]  # every mark, so no label covers one

    def free(box):
        a0, b0, a1, b1 = box
        inside = a0 >= x0 and a1 <= x1 + 90 and b0 >= y0 - 4 and b1 <= y1 + 4  # labels may run into the right margin
        return inside and all(a1 <= c0 or a0 >= c2 or b1 <= c1 or b0 >= c3 for c0, c1, c2, c3 in taken)

    placed = {}
    # Densest area first would be ideal; top-right first keeps the best classifiers' labels nearest their marks.
    for key, cx, cy in sorted(marks, key=lambda m: (-m[1], m[2])):
        w = len(key) * LABEL_CHAR_W
        choice = None
        for step in range(0, 6):
            for dy in ((0,) if step == 0 else (-step * LABEL_H, step * LABEL_H)):
                for side in ("start", "end"):
                    lx = cx + r + 2 if side == "start" else cx - r - 2 - w
                    box = (lx, cy + dy - LABEL_H / 2, lx + w, cy + dy + LABEL_H / 2)
                    if free(box):
                        choice = (box, side)
                        break
                if choice:
                    break
            if choice:
                break
        if not choice:  # nowhere free nearby: right of the mark, overlapping
            lx = cx + r + 2
            choice = ((lx, cy - LABEL_H / 2, lx + w, cy + LABEL_H / 2), "start")
        box, side = choice
        taken.append(box)
        placed[key] = (box[0] if side == "start" else box[2], box[3] - 3, side)
    return placed


def speed_scatter(points: list[dict]) -> str:
    """Speed in items/s (y, log; faster up) vs accuracy (x; more accurate to the right) with a hoverable legend.
    Every mark is labelled with the classifier's runner name (place_labels), with a leader line when the label had
    to move away from its mark. points: key, name, acc, per_item (seconds), style, kind."""
    width, height, left, right, top, bottom = 900, 480, 64, 100, 14, 42  # right: room for labels of the best apps
    pw, ph = width - left - right, height - top - bottom
    speeds = [1 / p["per_item"] for p in points]
    y_lo, y_hi = 10 ** math.floor(math.log10(min(speeds))), 10 ** math.ceil(math.log10(max(speeds)))
    if y_lo == y_hi:
        y_hi *= 10
    x_lo = max(0.0, math.floor(min(p["acc"] for p in points) * 10) / 10)

    def x(a: float) -> float:
        return left + (a - x_lo) / max(1.0 - x_lo, 1e-9) * pw

    def y(v: float) -> float:
        return top + (1 - (math.log10(v) - math.log10(y_lo)) / (math.log10(y_hi) - math.log10(y_lo))) * ph

    def speed_txt(v: float) -> str:
        return f"{v:,.0f}" if v >= 10 else f"{v:g}"

    svg = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Speed against accuracy, one mark '
           f'per classifier">']
    for e in range(round(math.log10(y_lo)), round(math.log10(y_hi)) + 1):
        svg.append(f'<line class="grid" x1="{left}" x2="{left + pw}" y1="{y(10 ** e):.1f}" y2="{y(10 ** e):.1f}"/>'
                   f'<text class="tick" x="{left - 6}" y="{y(10 ** e) + 4:.1f}" text-anchor="end">{speed_txt(10 ** e)}</text>')
    step = 0.05 if 1.0 - x_lo <= 0.3 else 0.1
    for i in range(round((1.0 - x_lo) / step) + 1):
        a = x_lo + i * step
        svg.append(f'<line class="grid" x1="{x(a):.1f}" x2="{x(a):.1f}" y1="{top}" y2="{top + ph}"/>'
                   f'<text class="tick" x="{x(a):.1f}" y="{top + ph + 16}" text-anchor="middle">{a * 100:.0f}%</text>')
    svg.append(f'<text class="tick" x="{left + pw / 2:.1f}" y="{height - 4}" text-anchor="middle">accuracy, more '
               f'accurate to the right →</text>'
               f'<text class="tick" transform="translate(12,{top + ph / 2:.1f}) rotate(-90)" text-anchor="middle">'
               f'speed: items per second (log scale), faster ↑</text>')
    xy = {p["key"]: (x(p["acc"]), y(1 / p["per_item"])) for p in points}
    tips = {p["key"]: (f"{p['name']}: {pct(p['acc'])} accuracy, {secs(p['per_item'])} per item "
                       f"({speed_txt(1 / p['per_item'])} items/s), {p['kind']}") for p in points}
    # Slowest first, so faster marks (usually denser) sit on top; the hovered one is raised by the script.
    for p in sorted(points, key=lambda p: p["per_item"], reverse=True):
        cx, cy = xy[p["key"]]
        svg.append(f'<g class="pt" data-key="{esc(p["key"])}" data-tip="{esc(tips[p["key"]])}">'
                   f'<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="12"/>{scatter_mark(p["style"], cx, cy)}</g>')
    # Labels: marks that (nearly) coincide share one label, "a · b", so a label never sits beside the wrong mark.
    clusters: list[list[str]] = []
    for key in sorted(xy, key=lambda k: (-xy[k][0], xy[k][1])):
        near = next((c for c in clusters if math.dist(xy[c[0]], xy[key]) < LABEL_H / 2), None)
        (near.append(key) if near else clusters.append([key]))
    centre = {c[0]: (sum(xy[k][0] for k in c) / len(c), sum(xy[k][1] for k in c) / len(c)) for c in clusters}
    text = {c[0]: " · ".join(c) for c in clusters}
    spots = place_labels([(text[c[0]], *centre[c[0]]) for c in clusters], (left, top, left + pw, top + ph))
    for c in clusters:
        cx, cy = centre[c[0]]
        lx, ly, anchor = spots[text[c[0]]]
        mid = ly - 3  # the label's vertical centre
        if abs(mid - cy) > 1:  # moved off the mark's row: a leader line points back to the mark
            end_x = lx - 2 if anchor == "start" else lx + 2
            svg.append(f'<line class="ldr" x1="{cx:.1f}" y1="{cy:.1f}" x2="{end_x:.1f}" y2="{mid:.1f}"/>')
        names = '<tspan class="sep"> · </tspan>'.join(
            f'<tspan data-key="{esc(k)}" data-tip="{esc(tips[k])}">{esc(k)}</tspan>' for k in c)
        svg.append(f'<text class="lbl" x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}">{names}</text>')
    svg.append("</svg>")
    legend = ["<ul class='legend'>"]
    order = list(APPROACHES)
    group = None
    for p in sorted(points, key=lambda p: (order.index(p["kind"]) if p["kind"] in order else len(order), -p["acc"])):
        if p["kind"] != group:  # one heading per approach, so like is compared with like
            group = p["kind"]
            legend.append(f"<li class='grp'>{esc(group)}</li>")
        legend.append(f"<li data-key='{esc(p['key'])}' data-tip='{esc(p['name'])}: {pct(p['acc'])}, "
                      f"{secs(p['per_item'])} per item' tabindex='0'><svg viewBox='0 0 18 18' width='18' height='18' "
                      f"aria-hidden='true'>{scatter_mark(p['style'], 9, 9.5, 6)}</svg>{esc(p['name'])}</li>")
    legend.append("</ul>")
    return f"<div class='scatter'>{''.join(svg)}{''.join(legend)}</div>"


# ---------------------------------------------------------------- page

CSS = """
:root { color-scheme: light;
  --bg:#f7f7f5; --surface:#fcfcfb; --border:#e3e2dd; --text:#0b0b0b; --text-2:#52514e; --muted:#6b6a65;
  --series:#1f69c2; --series-soft:#cde2fb; --grid:#ebeae6; --good:#0a7d0a; --bad:#b42f2f; --warn-bg:#fff4dc;
  --c1:#2a78d6; --c2:#eb6834; --c3:#1baf7a; --c4:#eda100; --c5:#e87ba4; --c6:#008300; --c7:#4a3aa7; --c8:#e34948; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark;
  --bg:#121211; --surface:#1a1a19; --border:#33332f; --text:#ffffff; --text-2:#c3c2b7; --muted:#8f8e86;
  --series:#3987e5; --series-soft:#1c3a5e; --grid:#2a2a27; --good:#5cc25c; --bad:#e66767; --warn-bg:#3a3020;
  --c1:#3987e5; --c2:#d95926; --c3:#199e70; --c4:#c98500; --c5:#d55181; --c6:#008300; --c7:#9085e9; --c8:#e66767; } }
:root[data-theme="dark"] { color-scheme: dark;
  --bg:#121211; --surface:#1a1a19; --border:#33332f; --text:#ffffff; --text-2:#c3c2b7; --muted:#8f8e86;
  --series:#3987e5; --series-soft:#1c3a5e; --grid:#2a2a27; --good:#5cc25c; --bad:#e66767; --warn-bg:#3a3020;
  --c1:#3987e5; --c2:#d95926; --c3:#199e70; --c4:#c98500; --c5:#d55181; --c6:#008300; --c7:#9085e9; --c8:#e66767; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 32px 16px 64px; overflow-wrap: anywhere; }
h1 { font-size: 26px; margin: 0 0 4px; } h2 { font-size: 19px; margin: 40px 0 6px; }
h3 { font-size: 15px; margin: 20px 0 6px; }
p.lede, .note { color: var(--text-2); } .note { font-size: 13px; margin: 4px 0 12px; max-width: 80ch; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(170px, 100%), 1fr)); gap: 12px; margin: 20px 0; }
.tile .k { color: var(--text-2); font-size: 12px; } .tile .v { font-size: 22px; font-weight: 600; margin-top: 2px; }
.tile .s { color: var(--muted); font-size: 12px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 13px; }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--border); text-align: right; white-space: nowrap; }
th { color: var(--text-2); font-weight: 600; font-size: 12px; background: var(--surface); position: sticky; top: 0; }
th:first-child, td:first-child { text-align: left; }
td.l, th.l { text-align: left; white-space: normal; }
td.nw { white-space: nowrap; }
tr.best td, tr.lead td { font-weight: 600; }
.legend li.grp { grid-column: 1 / -1; font-weight: 600; color: var(--text-2); cursor: default; margin-top: 4px; }
.pill.lead { color: var(--series); border-color: var(--series); font-weight: 600; }
.dot.dim { fill: var(--muted); opacity: .55; }
.frontier { stroke-dasharray: 4 3; }
.bar { fill: var(--series-soft); } .warnmark { color: var(--bad); cursor: help; font-weight: 400; }
td.dag::after { content: " †"; color: var(--muted); }
.rank-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(260px, 100%), 1fr)); gap: 12px; }
.rank-grid h3 { margin-top: 0; } .rank-grid td.sep { color: var(--muted); font-size: 12px; text-align: left; white-space: normal; }
tr.virtual td { color: var(--muted); font-style: italic; }
details summary { cursor: pointer; color: var(--series); } pre { font-size: 11px; margin: 6px 0 0; white-space: pre-wrap; }
.sig { color: var(--muted); } .bad { color: var(--bad); } .good { color: var(--good); }
.pill { display: inline-block; padding: 1px 8px; border-radius: 99px; font-size: 12px; border: 1px solid var(--border); }
.pill.ok { color: var(--good); } .pill.failed, .pill.invalid, .pill.incompatible { color: var(--bad); } .pill.skipped, .pill.existing, .pill.virtual { color: var(--text-2); }
.warn { background: var(--warn-bg); border-radius: 8px; padding: 10px 14px; margin: 12px 0; }
svg.chart { width: 100%; height: auto; display: block; font-size: 12px; }
.grid { stroke: var(--grid); stroke-width: 1; }
.tick { fill: var(--muted); font-size: 11px; } .rowlabel { fill: var(--text); } .value { fill: var(--text-2); }
.dot { fill: var(--series); stroke: var(--surface); stroke-width: 2; }
.whisker { stroke: var(--series); stroke-width: 2; stroke-linecap: round; opacity: .55; }
.line { fill: none; stroke: var(--series); stroke-width: 2; stroke-linejoin: round; }
.baseline { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; }
.marker { fill: var(--series); stroke: var(--surface); stroke-width: 2; }
.hit { fill: transparent; } g.row:hover .hit { fill: var(--series-soft); opacity: .35; }
.multiples { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(230px, 100%), 1fr)); gap: 12px; }
figure.multiple { margin: 0; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 10px; }
figure.multiple figcaption { font-size: 13px; font-weight: 600; margin-bottom: 4px; }
.cm { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(330px, 100%), 1fr)); gap: 12px; }
.cm .card { overflow-x: auto; }
.cm table td, .cm table th { padding: 3px 6px; font-size: 12px; }
.cm td.hitcell { background: var(--series-soft); }
#tip { position: fixed; pointer-events: none; background: var(--text); color: var(--bg); padding: 6px 10px;
  border-radius: 6px; font-size: 12px; max-width: 360px; opacity: 0; transition: opacity .1s; z-index: 10; }
code { font-size: 12px; }
.scatter .mk { stroke: var(--surface); stroke-width: 2; transition: opacity .12s; }
.scatter .lbl { font-size: 11px; fill: var(--text); paint-order: stroke; stroke: var(--surface); stroke-width: 3px;
  stroke-linejoin: round; transition: opacity .12s; }
.scatter .ldr { stroke: var(--text-2); stroke-width: 1; }
.scatter.focus .lbl tspan[data-key]:not(.on), .scatter.focus .lbl .sep { fill-opacity: .2; }
.scatter.focus .ldr { opacity: .3; }
.scatter .lbl tspan.on { font-weight: 600; }
.scatter.focus .pt:not(.on) .mk, .scatter.focus .legend li:not(.on) svg { opacity: .15; }
.scatter.focus .legend li:not(.on) { color: var(--muted); }
.scatter .pt.on .mk { stroke: var(--text); stroke-width: 1.5; }
.legend .mk { stroke: none; }
.legend svg { flex: none; }
.legend { list-style: none; margin: 12px 0 0; padding: 0; display: grid; grid-template-columns: repeat(auto-fill, minmax(min(230px, 100%), 1fr));
  gap: 4px 16px; font-size: 13px; }
.legend li { display: flex; align-items: center; gap: 8px; cursor: default; padding: 2px 4px; border-radius: 6px; }
.legend li.on { background: var(--series-soft); }
nav { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 13px; margin: 12px 0 0; }
nav a { color: var(--series); text-decoration: none; }
nav.sections { gap: 4px 18px; padding: 8px 0; background: var(--bg); border-bottom: 1px solid var(--border); z-index: 5; }
nav.sections .grp { display: inline-flex; flex-wrap: wrap; gap: 4px 10px; align-items: baseline; }
nav.sections .grp b { color: var(--text-2); font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }
@media (min-width: 900px) { nav.sections { position: sticky; top: 0; } h2[id], h3[id] { scroll-margin-top: 64px; } }
a { color: var(--series); } td a { color: inherit; text-decoration-color: var(--muted); }
.pill.kind { color: var(--text-2); cursor: help; }
.callout { font-size: 13px; color: var(--text-2); border-left: 3px solid var(--series-soft); padding: 2px 10px; margin: 0 0 10px; max-width: 90ch; }
details.formula { margin: -4px 0 12px; font-size: 13px; }
.back { font-size: 12px; margin: 6px 0 0; } .back a { text-decoration: none; }
.taskcard { margin: 16px 0 4px; } .taskcard h2.tc { margin: 0 0 4px; font-size: 16px; }
.taskcard dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 14px; margin: 8px 0; font-size: 13px; }
.taskcard dt { color: var(--text-2); } .taskcard dd { margin: 0; }
@media (max-width: 600px) { .taskcard dl { grid-template-columns: 1fr; } .taskcard dd { margin-bottom: 6px; } }
.zero { stroke: var(--text-2); stroke-width: 1.5; }
.dot.open { fill: var(--surface); stroke: var(--series); stroke-width: 2; }
.axis { fill: var(--text-2); font-size: 12px; }
@media (max-width: 700px) { svg.forest { min-width: 640px; } }
.scroll td:first-child, .scroll th:first-child { position: sticky; left: 0; background: var(--surface); z-index: 1; }
.scroll th:first-child { z-index: 2; }
g.row:focus { outline: none; } g.row:focus .hit { fill: var(--series-soft); opacity: .5; }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { transition: none !important; animation: none !important; }
  html { scroll-behavior: auto !important; } }
"""

JS = """
document.querySelectorAll('.scatter').forEach(root => {
  const set = key => {
    root.classList.toggle('focus', !!key);
    root.querySelectorAll('[data-key]').forEach(el => el.classList.toggle('on', el.dataset.key === key));
    if (key) { const g = root.querySelector('.pt.on'); g && g.parentNode.appendChild(g); }  // raise the hovered mark
  };
  root.addEventListener('mouseover', e => { const el = e.target.closest('[data-key]'); set(el ? el.dataset.key : null); });
  root.addEventListener('mouseleave', () => set(null));
  root.addEventListener('focusin', e => { const el = e.target.closest('[data-key]'); set(el ? el.dataset.key : null); });
  root.addEventListener('focusout', () => set(null));
});
const tip = document.getElementById('tip');
document.addEventListener('mousemove', e => {
  const t = e.target.closest('[data-tip]');
  if (!t) { tip.style.opacity = 0; return; }
  tip.textContent = t.dataset.tip; tip.style.opacity = 1;
  const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  tip.style.left = x + 'px'; tip.style.top = (e.clientY + 14) + 'px';
});
// Keyboard: focusing an element with a tooltip shows it next to the element.
document.addEventListener('focusin', e => {
  const t = e.target.closest('[data-tip]');
  if (!t) { tip.style.opacity = 0; return; }
  const r = t.getBoundingClientRect();
  tip.textContent = t.dataset.tip; tip.style.opacity = 1;
  tip.style.left = Math.min(r.left, window.innerWidth - tip.offsetWidth - 8) + 'px';
  tip.style.top = Math.min(r.bottom + 6, window.innerHeight - tip.offsetHeight - 8) + 'px';
});
document.addEventListener('focusout', () => { tip.style.opacity = 0; });
"""


def build_report(run_dir: Path, also: tuple[tuple[Path, Path], ...] = ()) -> Path:
    """Write the full report (run_dir/report.html) and the plain-language summary (run_dir/report_simple.html).

    Each (summary_path, full_path) pair in `also` gets another copy of both, each with its own relative links (the
    summary links to its full report and back). Returns the run folder's full report.
    """
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    global LABELS, PROB_COLS
    task = tasks.load_task(summary.get("task"))
    LABELS = summary.get("labels") or task.label_ids
    PROB_COLS = [f"p_{label}" for label in LABELS]
    primary = task.manifest.get("primary_metric", "accuracy")
    noun = task.manifest.get("item_noun", "item")
    apps, frames, metas, metrics = summary["apps"], {}, {}, {}
    usable = ("ok", "existing", "virtual")  # "invalid" results failed validation: reported as failures only
    for name, entry in apps.items():
        if entry.get("csv") and entry.get("status") in usable:
            frames[name] = pd.read_csv(run_dir / entry["csv"])
            if entry.get("run_json"):
                metas[name] = json.loads((run_dir / entry["run_json"]).read_text())

    # Same-sample check. Reference: the sample the runner computed (summary sample_sha1); for older
    # summaries, the sample most apps used. Texts, labels and each app's recorded sample_sha1 must match.
    shas = {name: frame_sha1(df) for name, df in frames.items()}
    ref_sha = summary.get("sample_sha1") or (pd.Series(list(shas.values())).mode().iloc[0] if shas else None)
    ref_name = next((n for n, s in shas.items() if s == ref_sha), None)
    incompatible: dict[str, str] = {}
    for name, df in frames.items():
        recorded = metas.get(name, {}).get("sample_sha1")
        if ref_name is None:
            incompatible[name] = "no app used the expected sample"
        elif shas[name] != ref_sha:
            ref = frames[ref_name]
            if len(df) != len(ref):
                incompatible[name] = f"{len(df)} articles, reference has {len(ref)}"
            else:
                incompatible[name] = (f"{int((df['text'].values != ref['text'].values).sum())} texts and "
                                      f"{int((df['label'].values != ref['label'].values).sum())} labels differ")
        elif recorded and recorded != ref_sha:
            incompatible[name] = "run metadata sample_sha1 differs from the reference sample"
    excluded = {name: (frames.pop(name), metas.pop(name, {})) for name in incompatible}
    for name, df in frames.items():
        metrics[name] = app_metrics(df)
        apply_held_out(metrics[name], df, calibration_frame(run_dir, apps[name], df, (summary.get("seeds") or [summary["seed"]])[0]))
    mismatched = list(incompatible)
    missing_probs = [n for n, m in metrics.items() if not m["has_probs"]]
    missing_meta = [n for n in metrics if n not in metas]

    # Virtual baselines (majority, random) are reference rows only: never "best", not in tiles or cross-app analyses.
    virtual = {n for n in metrics if apps[n].get("virtual")}
    # Ranked by the task's primary metric (accuracy, or macro F1 for imbalanced tasks), then the other one.
    other = "macro_f1" if primary == "accuracy" else "accuracy"
    ranked = sorted(metrics, key=lambda n: (-metrics[n][primary], -metrics[n][other]))
    models = [n for n in ranked if n not in virtual]
    best = models[0] if models else None  # top point estimate; not a declared winner unless alone in the leading tier

    # Paired comparisons with the top model: exact McNemar, Holm-adjusted over all models; paired differences.
    correct = {n: (frames[n]["label"] == frames[n]["pred"]).to_numpy() for n in ranked}
    comparisons = {n: paired_difference(correct[best], correct[n]) for n in ranked if n != best} if best else {}
    adjusted = holm({n: comparisons[n]["p"] for n in models if n != best})
    for name in ranked:
        metrics[name]["mcnemar"] = comparisons[name]["p"] if name in comparisons else None
        metrics[name]["holm"] = adjusted.get(name)

    # Paired bootstrap: the same resampled articles for every app in each replicate.
    n_items = len(next(iter(frames.values()))) if frames else 0
    replicates = BOOTSTRAP_REPLICATES if n_items <= 2000 else BOOTSTRAP_REPLICATES // 2  # bounds memory on the full set
    if n_items:
        W = resample_weights(n_items, replicates, BOOTSTRAP_SEED)
        boot = {n: bootstrap_metrics(frames[n], W, LABELS) for n in ranked}
        del W
        for name in ranked:
            metrics[name]["boot"] = {k: percentile_ci(v) for k, v in boot[name].items()}
            if name in comparisons:
                comparisons[name]["boot"] = percentile_ci(boot[name]["accuracy"] - boot[best]["accuracy"])
    # Leading tier: models whose 95% bootstrap rank interval includes 1 (not only "not significantly behind
    # the data-selected top", which overstates how far behind the others are).
    ranks = rank_intervals({n: boot[n]["accuracy"] for n in models}) if n_items and models else {}
    lead = [n for n in models if n in set(leading_tier(ranks))]  # in accuracy order
    report_meta = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "alpha": ALPHA,
        "bootstrap": {"method": "paired percentile bootstrap over articles", "seed": BOOTSTRAP_SEED,
                      "replicates": replicates, "interval": 1 - ALPHA},
        "multiple_comparisons": {"test": "exact two-sided McNemar vs the top point estimate",
                                 "correction": "Holm", "family": [n for n in models if n != best]},
        "top_point_estimate": best, "leading_tier": lead,
        "leading_tier_rule": "95% paired-bootstrap rank interval includes rank 1",
        "selective_thresholds": {n: metrics[n]["held_out"][0.95].get("method") for n in metrics},
        "rank_intervals": ranks,
    }

    pricing = perf.load_pricing()
    title = lambda n: apps[n]["title"]  # noqa: E731
    supervision = lambda n: apps[n].get("supervision") or APP_SUPERVISION.get(n, "zero-shot")  # noqa: E731
    reg = APPS_BY_NAME.get
    conf_kind = lambda n: confidence_kind(reg(n)) if reg(n) else None  # noqa: E731
    # The registry's current confidence definition; older summaries keep the text they were written with.
    conf_text = lambda n: reg(n).confidence if reg(n) else apps[n]["confidence"]  # noqa: E731

    def approach(n: str) -> str:
        if apps[n].get("virtual"):
            return "virtual"
        return apps[n].get("approach") or (reg(n).approach_on(task.name) if reg(n) else supervision(n))

    def is_remote(n: str) -> bool:
        return bool(apps[n].get("remote", reg(n).remote if reg(n) else False))

    def type_txt(n: str) -> str:
        t = approach(n)
        return (t + (" · remote" if is_remote(n) else "")
                + (" · task-calibrated conf." if t == "zero-shot" and conf_kind(n) == "task" else ""))

    def conf_badge(n: str) -> str:
        k = conf_kind(n)
        if not k:
            return "–"
        label, tip = CONFIDENCE_KINDS[k]
        return f"<span class='pill kind' data-tip='{esc(tip)}'>{esc(label)}</span>"

    def link(n: str) -> str:
        """The app's title, linked to its model page when one exists."""
        if (MODELS_DIR / f"{n}.html").exists():
            return f"<a href='{MODELS_HREF}/{esc(n)}.html'>{esc(title(n))}</a>"
        return esc(title(n))

    def execution(n: str) -> str:
        a = reg(n)
        return f"shortlist {a.shortlist}" if a and a.shortlist else ("–" if n in virtual else "direct")
    out: list[str] = []
    w = out.append
    started = datetime.fromisoformat(summary["started_at"]).astimezone()
    env = summary["environment"]

    w("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
      "<meta name='viewport' content='width=device-width, initial-scale=1'>"
      f"<title>Classifier Benchmark Report</title><style>{CSS}</style></head><body><main>")
    w("<h1>Classifier benchmark report</h1>")
    w(f"<p class='note'><a href='{SIMPLE_HREF}'>← Summary report</a>: the plain-language version of these results. "
      "This page is the full raw generated report.</p>")
    seeds = summary.get("seeds") or [summary["seed"]]
    tier = summary.get("tier") or tasks.tier_name(task, summary["n"])
    leak = summary.get("leakage") or {}
    leak_txt = (f" Test texts also in train: {leak.get('in_train', '–')}, in calibration: {leak.get('in_calibration', '–')}."
                if leak else "")
    repeat_txt = (f" Repeated runs: {len(seeds)} seed(s) × {summary.get('repeat', 1)} run(s); tables rank seed "
                  f"{seeds[0]}, <a href='#repeats'>Repeated runs</a> summarizes all.") if len(seeds) * summary.get("repeat", 1) > 1 else ""
    w(f"<p class='lede'>Task <b>{esc(task.name)}</b> ({esc(task.manifest.get('description', ''))} "
      f"{esc(task.manifest.get('license', 'license not stated'))}), <b>{esc(tier)}</b> tier: "
      f"{esc(task.manifest.get('sampling', 'balanced'))} stratified sample of {summary['n']} {noun}s "
      f"(seed {summary['seed']}), {len(LABELS)} labels, ranked by {primary.replace('_', ' ')}.{leak_txt} Run {started:%Y-%m-%d %H:%M} on {esc(env.get('machine'))}, "
      f"{env.get('memory_gb')} GB, {esc(env.get('os'))}. Mode: {esc(summary['mode'])}.{repeat_txt}"
      + (f" Sample SHA-1 <code>{esc(ref_sha[:12])}</code>." if ref_sha else "") + "</p>")
    w(f"<p class='note'>Report generated from these results on {esc(report_meta['generated_at'].replace('T', ' '))}. "
      f"Model guides: <a href='{MODELS_HREF}/index.html'>all models</a>; app names below link to each model's page.</p>")
    w(task_card(task, summary, frames.get(ref_name) if ref_name else None, primary, noun))
    nav_groups = [("Quality", [("overview", "Overview"), ("speed-accuracy", "Accuracy vs speed"), ("accuracy", "Accuracy"),
                               ("stats", "Statistics"), ("rankings", "Rankings"), ("classes", "Per class"),
                               ("confusion", "Confusion"), ("hardest", "Hardest")]),
                  ("Confidence", [("confidence", "Confidence"), ("selective", "Selective"), ("calibration", "Calibration")]),
                  ("Performance", [("pareto", "Pareto"), ("speed", "Speed"), ("latency", "Latency"), ("resources", "Memory"),
                                   ("cost", "Tokens &amp; cost"), ("training", "Training")]),
                  ("Reproducibility", [("repeats", "Repeated runs"), ("incompatible", "Incompatible"), ("status", "Run status"),
                                       ("env", "Environment"), ("defs", "Definitions")])]
    w("<nav class='sections' aria-label='Report sections'>" + "".join(
        f"<span class='grp'><b>{g}</b>" + "".join(f"<a href='#{a}'>{t}</a>" for a, t in links) + "</span>"
        for g, links in nav_groups) + "</nav>")

    invalid = [n for n, a in apps.items() if a.get("status") == "invalid"]
    on_battery = [n for n in metas if perf.on_battery(metas[n])]
    low_power = [n for n in metas if ((metas[n].get("runner") or {}).get("conditions") or {}).get("low_power_mode")]
    if on_battery or low_power:
        w("<div class='warn'><strong>Timings taken on battery or in Low Power Mode may be slower and are not directly "
          f"comparable with the others:</strong> {esc(', '.join(sorted(set(on_battery + low_power))))}. Rerun plugged in "
          "(<code>--require-ac</code> refuses battery runs).</div>")
    for msg, names in [("Different article samples, left out of every comparison (see Incompatible results):", mismatched),
                       ("Failed result validation, reported as failures (see Run status):", invalid),
                       ("No per-label probabilities (p_* columns); Brier and log loss unavailable:", missing_probs),
                       ("No run metadata (run_n*.json); timings unavailable:", missing_meta)]:
        if names:
            w(f"<div class='warn'><strong>{esc(msg)}</strong> {esc(', '.join(names))}</div>")

    # Accuracy vs speed scatter: the first chart, one mark per classifier (see speed_scatter).
    styles = {n: i for i, n in enumerate(n for n, a in apps.items() if not a.get("virtual"))}  # registry order
    scatter = [{"key": n, "name": title(n), "acc": metrics[n]["accuracy"], "per_item": per_item(metas[n]),
                "style": styles.get(n, len(styles)), "kind": approach(n)}
               for n in models if n in metas and per_item(metas[n])]
    if len(scatter) > 1:
        w("<h2 id='speed-accuracy'>Speed vs accuracy</h2><p class='note'>One mark per classifier: further right is "
          "more accurate, higher is faster (items per second with each app's own batching or concurrency, log "
          "scale), so the top-right corner is best. The legend is grouped by approach (zero-shot, in-domain checkpoint, "
          "classical supervised, fine-tuned encoder, from scratch): a rank across groups compares different amounts of "
          "task data and training, not only model quality. Each classifier has its own color and shape; hover a mark for "
          "its name and values, or a name in the legend to find its mark. Exact values are in the Overview table below."
          + (f" Not plotted (no timings): {esc(', '.join(title(n) for n in models if n not in {p['key'] for p in scatter}))}."
             if len(scatter) < len(models) else "") + "</p>")
        w("<div class='card'>" + speed_scatter(scatter) + "</div>")

    # Tiles
    ok = [n for n in frames if apps[n]["status"] in ("ok", "existing")]
    not_run = [n for n, a in apps.items() if not a.get("virtual") and n not in frames]
    n_real = sum(1 for a in apps.values() if not a.get("virtual"))
    fastest = min(metas, key=lambda n: per_item(metas[n])) if metas else None
    held95 = lambda n: metrics[n]["held_out"][0.95]  # noqa: E731
    gate_ok = [n for n in models if held95(n)["met"]]  # held-out accuracy actually reached 95%
    best_gate = max(gate_ok, key=lambda n: held95(n)["coverage"]) if gate_ok else None
    w("<div class='tiles'>")
    w(f"<div class='card tile'><div class='k'>Apps with results</div><div class='v'>{len(ok)} / {n_real}</div>"
      f"<div class='s'>{esc('no comparable results: ' + ', '.join(not_run)) if not_run else 'all apps'}</div></div>")
    if best and len(lead) > 1:
        m = metrics[best]
        w(f"<div class='card tile'><div class='k'>Bootstrap leading tier</div><div class='v'>{len(lead)} apps</div>"
          f"<div class='s'>{pct(min(metrics[n]['accuracy'] for n in lead))}–{pct(m['accuracy'])} accuracy at n = {m['n']}; "
          "each attained rank 1 within the central 95% of its bootstrap rank distribution</div></div>")
    elif best:
        m = metrics[best]
        w(f"<div class='card tile'><div class='k'>Most accurate</div><div class='v'>{pct(m['accuracy'])}</div>"
          f"<div class='s'>{esc(title(best))}, 95% CI {pct(m['ci'][0])}–{pct(m['ci'][1])}"
          + ("; the only app whose bootstrap rank interval includes 1"
             + ("; every Holm p < 0.05" if all((metrics[n].get("holm") or 1) < ALPHA for n in models if n != best)
                else "") if len(models) > 1 else "") + "</div></div>")
    if fastest:
        w(f"<div class='card tile'><div class='k'>Fastest average time per item (batched throughput)</div><div class='v'>{secs(per_item(metas[fastest]))}</div>"
          f"<div class='s'>{esc(title(fastest))}, {pct(metrics[fastest]['accuracy'])} accuracy</div></div>")
    if best_gate and held95(best_gate)["coverage"] > 0:
        g = held95(best_gate)
        w(f"<div class='card tile'><div class='k'>Most auto-accepted at 95% accuracy (held-out)</div><div class='v'>{pct(g['coverage'])}</div>"
          f"<div class='s'>{esc(title(best_gate))}: {g['accepted']} {noun}s accepted, {pct(g['accuracy'])} correct; "
          f"{esc(held_out_txt(g))}</div></div>")
    w("</div>")

    # Overview table
    w(f"<h2 id='overview'>Overview</h2><p class='note'>Sorted by {primary.replace('_', ' ')}. p (Holm) is an exact McNemar test against the "
      "top point estimate on the same articles, Holm-adjusted for testing every model: below 0.05 indicates evidence of a "
      "difference in paired error rates after correcting for the report's multiple comparisons. <b>Bootstrap leading "
      "tier</b> = apps whose percentile bootstrap rank interval includes rank 1 under this report's resampling rule. This "
      "is an uncertainty summary, not a test that the apps are equivalent or statistically tied; none is declared the "
      "winner. Results describe each app under this run's training data, hardware and settings, not a general "
      "ranking. Type is the app's approach: " + "; ".join(f"<b>{esc(k)}</b>: {esc(v.lower())}" for k, v in APPROACHES.items())
      + "; <b>remote</b> apps run on another machine, so their timings include the network. "
      "<b>zero-shot</b> apps make their decision without fitting on this task's labels; those marked "
      "<i>task-calibrated conf.</i> fit only a softmax temperature on labeled train rows, so their accuracy is "
      "zero-shot but their ECE, Brier, log loss and selective-prediction results are task-calibrated "
      f"(<a href='{MODELS_HREF}/index.html#pipeline'>how the benchmark runs a model</a>). "
      "<b>Confidence</b>: what the confidence score is (see <a href='#defs-confidence'>Definitions</a>). "
      "<b>Auto-accept</b>: share of items accepted by a confidence threshold that was fitted on <i>other</i> items "
      "(a separate calibration sample when the run has a second seed, otherwise 2-fold cross-fitting) with the rule "
      "'Wilson 95% lower bound ≥ 95%'; shown only when the accepted items were in fact ≥ 95% correct "
      "(see <a href='#selective'>Selective prediction</a>); <b>supervised</b> baselines are trained on the task's train split (never the test "
      "set); <b>in-domain trained</b> models are zero-shot elsewhere but saw this task's data in their own training "
      "(checked for test-text overlap at setup); <b>virtual</b> rows are trivial predictors generated by the runner (no model was run) and are never ranked "
      "best.</p>")
    w("<div class='card scroll'><table><thead><tr><th>App</th><th class='l'>Type</th><th class='l'>Runs on</th>"
      "<th>Accuracy</th><th>95% CI</th>"
      "<th>Macro F1</th><th>Top-2</th><th>p (Holm)</th><th>Per item</th><th>Items/s</th>"
      "<th>Auto-accept coverage at 95% (held-out)</th><th>AUROC</th><th>E-AURC</th><th>ECE</th><th class='l'>Confidence</th><th class='l'>Execution</th></tr></thead><tbody>")
    for name in ranked:
        m, meta = metrics[name], metas.get(name, {})
        p_txt = "top" if name == best else p_cell(m.get("holm"), ALPHA)
        row_class = "lead" if name in lead else ("virtual" if name in virtual else "")
        badge = " <span class='pill lead'>leading tier</span>" if name in lead and len(lead) > 1 else ""
        w(f"<tr class='{row_class}'><td>{link(name)}{badge}</td><td class='l nw'>{esc(type_txt(name))}</td>"
          f"<td class='l nw'>{esc(apps[name]['runs_on'])}</td>"
          f"<td>{pct(m['accuracy'])}</td><td>{pct(m['ci'][0])}–{pct(m['ci'][1])}</td><td>{num(m['macro_f1'])}</td>"
          f"<td>{pct(m['top2'])}</td>"
          f"<td>{p_txt}</td><td>{secs(per_item(meta))}</td><td>{num(meta.get('items_per_second'), 1)}</td>"
          f"<td>{pct(m['held_out'][0.95]['coverage']) if m['held_out'][0.95]['met'] else '–'}</td><td>{num(m['auroc'])}</td>"
          f"<td>{num(m['eaurc'])}</td><td>{num(m['ece'])}</td>"
          f"<td class='l nw'>{conf_badge(name)}</td><td class='l nw'>{esc(execution(name))}</td></tr>")
    w("</tbody></table></div>")
    w("<p class='note'>Confidence types: " + " ".join(f"<i>{esc(l)}</i>: {esc(t)}" for l, t in CONFIDENCE_KINDS.values())
      + "</p>")

    # Accuracy chart
    if ranked:
        lo_all = min(metrics[n]["ci"][0] for n in ranked)
        lo = max(0.0, math.floor(lo_all * 10) / 10)
        rows = [{"name": title(n), "value": metrics[n]["accuracy"], "low": metrics[n]["ci"][0], "high": metrics[n]["ci"][1],
                 "tip": f"{title(n)}: {pct(metrics[n]['accuracy'])} ({metrics[n]['correct']}/{metrics[n]['n']}), "
                        f"95% CI {pct(metrics[n]['ci'][0])}–{pct(metrics[n]['ci'][1])}"} for n in ranked]
        ticks = [lo + i * (1 - lo) / 4 for i in range(5)]
        w("<h2 id='accuracy'>Accuracy</h2><p class='note'>Dot = accuracy, line = 95% Wilson interval. These intervals "
          "describe uncertainty in each app's accuracy separately. Do not infer pairwise significance from whether "
          "they overlap; use the <a href='#stats'>paired comparisons</a>. A larger test sample may narrow uncertainty "
          "when more test data are available.</p>")
        w("<div class='card'>" + dot_plot(rows, lo=lo, hi=1.0, fmt=lambda v: pct(v, 0) if v in ticks else pct(v), ticks=ticks,
                                         x_title="accuracy (dot) with 95% Wilson interval") + "</div>")

    row_cls = lambda n: "lead" if n in lead else ("virtual" if n in virtual else "")  # noqa: E731

    # Statistical comparisons
    if best and len(ranked) > 1:
        w(f"<h2 id='stats'>Statistical comparisons</h2><p class='note'>Every app against the top point estimate "
          f"({esc(title(best))}) on the same {metrics[best]['n']} articles. Δ = accuracy difference in percentage "
          "points. Score CI: Tango score interval for a difference of paired proportions (valid with few discordant "
          "articles); bootstrap CI: percentile interval "
          f"from {replicates:,} paired resamples of the articles (seed {BOOTSTRAP_SEED}). McNemar p is the raw exact "
          f"test; Holm p adjusts it for the {len(adjusted)} comparisons made. An interval containing 0 or a Holm-adjusted "
          f"p of at least {ALPHA} indicates that the corresponding procedure did not detect a difference at the 5% "
          "level. This is not evidence that the apps are equivalent; the intervals and the p-value come from different "
          "procedures and need not agree. Rank: 95% interval of each app's rank over the same "
          "resamples; P(best): share of resamples where it ranks first (ties split). Tier comes from the rank "
          "interval, which reduces, but does not remove, the effect of choosing the top app from the same data. It is "
          "descriptive, not a pairwise test.</p>")
        forest_rows = [{"name": title(n), "diff": comparisons[n]["diff"], "low": comparisons[n]["low"],
                        "high": comparisons[n]["high"]} for n in models if n != best and n in comparisons]
        if forest_rows:
            w(f"<h3>Paired differences vs {esc(title(best))}</h3><p class='note'>Each row is that app's accuracy minus "
              f"the reference model's ({esc(title(best))}, the top point estimate) on the same articles, in percentage "
              "points, with the 95% Tango score interval for paired proportions. Hollow markers and “(incl. 0)” mark "
              "intervals that include 0: no difference was detected at that level, which is not evidence that the apps "
              f"are equivalent. Holm-adjusted McNemar p-values for the {len(adjusted)} comparisons are in the table "
              "below; hover or focus a row for exact values.</p>")
            w("<div class='card scroll'>" + forest_plot(forest_rows, title(best)) + "</div>")
        w("<div class='card scroll'><table><thead><tr><th>App</th><th>Accuracy</th><th>Δ vs top</th><th>95% CI (score)</th>"
          "<th>95% CI (bootstrap)</th><th>Only top right</th><th>Only this right</th><th>McNemar p</th><th>Holm p</th>"
          "<th>Rank (95% CI)</th><th>P(best)</th><th class='l'>Tier</th></tr></thead><tbody>")
        for name in ranked:
            r = ranks.get(name)
            rank_cells = (f"<td>{r['rank_low']}–{r['rank_high']}</td><td>{pct(r['p_best'])}</td>" if r
                          else "<td>–</td><td>–</td>")
            if name == best:
                w(f"<tr class='{row_cls(name)}'><td>{link(name)} (top)</td><td>{pct(metrics[name]['accuracy'])}</td>"
                  f"<td colspan='7' class='sig'>reference for Δ and the tests</td>{rank_cells}"
                  f"<td class='l'>{'leading' if name in lead else 'behind'}</td></tr>")
                continue
            c, m = comparisons[name], metrics[name]
            tier_txt = "leading" if name in lead else ("virtual (not tested)" if name in virtual else "behind")
            w(f"<tr class='{row_cls(name)}'><td>{link(name)}</td>"
              f"<td>{pct(m['accuracy'])}</td><td>{signed_pct(c['diff'])}</td>"
              f"<td>{ci_txt((c['low'], c['high']), signed_pct)}</td><td>{ci_txt(c.get('boot', (None, None)), signed_pct)}</td>"
              f"<td>{c['only_a']}</td><td>{c['only_b']}</td><td>{num(c['p'])}</td>"
              f"<td>{p_cell(m.get('holm'), ALPHA)}</td>{rank_cells}<td class='l'>{tier_txt}</td></tr>")
        w("</tbody></table></div>" + back_link())
        nan2 = (float("nan"), float("nan"))
        w("<h3>Bootstrap 95% intervals</h3><p class='note'>Percentile intervals over the same paired resamples. Wide "
          "intervals mean the sample is too small to pin the metric down; to compare two apps use the paired table "
          "above rather than overlap here.</p><div class='card scroll'><table><thead><tr><th>App</th><th>Accuracy</th>"
          "<th>95% CI</th><th>Macro F1</th><th>95% CI</th><th>AUROC</th><th>95% CI</th><th>ECE</th><th>95% CI</th>"
          "<th>Brier</th><th>95% CI</th></tr></thead><tbody>")
        for name in ranked:
            m, b = metrics[name], metrics[name].get("boot", {})
            w(f"<tr class='{row_cls(name)}'><td>{link(name)}</td>"
              f"<td>{pct(m['accuracy'])}</td><td>{ci_txt(b.get('accuracy', nan2), pct)}</td>"
              f"<td>{num(m['macro_f1'])}</td><td>{ci_txt(b.get('macro_f1', nan2), num)}</td>"
              f"<td>{num(m['auroc'])}</td><td>{ci_txt(b.get('auroc', nan2), num)}</td>"
              f"<td>{num(m['ece'])}</td><td>{ci_txt(b.get('ece', nan2), num)}</td>"
              f"<td>{num(m.get('brier'))}</td><td>{ci_txt(b.get('brier', nan2), num)}</td></tr>")
        w("</tbody></table></div>")

    # Separate rankings: quality, latency, resource, cost
    if models:
        w("<h2 id='rankings'>Rankings by dimension</h2><p class='note'>No single ordering fits every use, so each list "
          "ranks the models on one dimension only. Quality puts the bootstrap leading tier before the rest. "
          "Throughput uses each app's own batching or concurrency; interactive latency is one article at a time. "
          "Memory is the app process's peak footprint plus the model memory of the server it uses (Ollama, OpenJev); "
          "apps without a measurement are listed last. Cost uses runner/pricing.json and says when it is unknown.</p>"
          "<div class='rank-grid'>")
        w("<div class='card'><h3>Quality (accuracy)</h3><table><tbody>")
        for i, name in enumerate(models, start=1):
            if i == len(lead) + 1 and len(lead) < len(models):
                w("<tr><td class='sep' colspan='3'>below the leading tier</td></tr>")
            w(f"<tr class='{'lead' if name in lead else ''}'><td>{i}</td><td class='l'>{esc(title(name))}</td>"
              f"<td>{pct(metrics[name]['accuracy'])}</td></tr>")
        w("</tbody></table></div>")
        timed_models = sorted((n for n in models if n in metas), key=lambda n: per_item(metas[n]))
        w("<div class='card'><h3>Throughput (time per item)</h3><table><tbody>")
        for i, name in enumerate(timed_models, start=1):
            w(f"<tr><td>{i}</td><td class='l'>{esc(title(name))}<br><span class='sig'>{esc(perf.config_text(metas[name]))}</span></td>"
              f"<td>{secs(per_item(metas[name]))}</td></tr>")
        w("</tbody></table></div>")
        p50 = lambda n: perf.interactive(metas[n]).get("p50")  # noqa: E731
        interactive_models = sorted((n for n in models if n in metas and p50(n) is not None), key=p50)
        w("<div class='card'><h3>Interactive latency (p50)</h3><table><tbody>")
        for i, name in enumerate(interactive_models, start=1):
            w(f"<tr><td>{i}</td><td class='l'>{esc(title(name))}</td><td>{secs(p50(name))}</td></tr>")
        missing = [n for n in models if n not in interactive_models]
        if missing:
            w(f"<tr><td class='sep' colspan='3'>not measured: {esc(', '.join(title(n) for n in missing))}</td></tr>")
        w("</tbody></table></div>")
        mem_total = lambda n: perf.memory(metas[n])["total"] if n in metas else None  # noqa: E731
        measured = sorted((n for n in models if mem_total(n)), key=mem_total)
        w("<div class='card'><h3>Memory (app + server)</h3><table><tbody>")
        for i, name in enumerate(measured, start=1):
            w(f"<tr><td>{i}</td><td class='l'>{esc(title(name))}<br><span class='sig'>{esc(apps[name]['runs_on'])}</span></td>"
              f"<td>{size_txt(mem_total(name))}</td></tr>")
        missing = [n for n in models if n not in measured]
        if missing:
            w(f"<tr><td class='sep' colspan='3'>not measured: {esc(', '.join(title(n) for n in missing))}</td></tr>")
        w("</tbody></table></div>")
        cost_of = lambda n: getattr(APPS_BY_NAME.get(n), "cost", "free")  # noqa: E731
        costs = {n: perf.cost(n, cost_of(n), metas.get(n, {}), pricing) for n in models}
        w("<div class='card'><h3>Cost per 1,000 items</h3><table><tbody>")
        for basis, label in (("none", "No per-request charge (local or own server)"), ("measured", "Measured (billed)"),
                             ("estimated", "Estimated (tokens x configured price)"), ("unknown", "Unknown: no price configured")):
            members = [n for n in models if costs[n]["basis"] == basis]
            if members:
                w(f"<tr><td class='sep' colspan='2'>{esc(label)}</td></tr>")
                w("".join(f"<tr><td class='l'>{esc(title(n))}</td><td>{money_txt(costs[n], 'per_1000')}</td></tr>"
                          for n in members))
        w("</tbody></table></div></div>")

    # Pareto frontier: accuracy vs time per item (and, below, vs interactive latency and memory)
    pareto_points = {n: (metrics[n]["accuracy"], per_item(metas[n])) for n in models if n in metas and per_item(metas[n])}
    if len(pareto_points) > 1:
        front = pareto_front(pareto_points)
        report_meta["pareto_front_accuracy_vs_time_per_item"] = front
        w("<h2 id='pareto'>Accuracy vs speed: Pareto frontier</h2><p class='note'>An app is on the frontier when no "
          "other app is both at least as accurate and at least as fast (and strictly better on one); every app off it "
          "has an alternative that is better on both. Point estimates only: use the paired comparisons for accuracy "
          "differences. Hover a point for values.</p>"
          + callout("A frontier model is not dominated on the displayed axes.", "changing the axes or the "
                    "constraints can change the frontier, and the points ignore accuracy uncertainty."))
        w("<div class='card'>" + pareto_chart(pareto_points, front, {n: title(n) for n in pareto_points}) + "</div>")
        w("<div class='card scroll' style='margin-top:12px'><table><thead><tr><th>Frontier app (fastest first)</th>"
          "<th>Accuracy</th><th>Per item</th><th class='l'>Leading tier</th></tr></thead><tbody>")
        for name in front:
            w(f"<tr><td>{link(name)}</td><td>{pct(pareto_points[name][0])}</td><td>{secs(pareto_points[name][1])}</td>"
              f"<td class='l'>{'yes' if name in lead else 'no'}</td></tr>")
        w("</tbody></table></div>")
        for key, heading, x_label, fmt, value in (
                ("interactive_p50", "Accuracy vs interactive latency (p50, one article per call)",
                 "interactive latency p50 (log scale), faster to the left", secs,
                 lambda n: perf.interactive(metas[n]).get("p50")),
                ("memory", "Accuracy vs memory (app process + server model)",
                 "peak memory (log scale), smaller to the left", size_txt, lambda n: perf.memory(metas[n])["total"])):
            pts = {n: (metrics[n]["accuracy"], value(n)) for n in models if n in metas and value(n)}
            if len(pts) > 1:
                fr = pareto_front(pts)
                report_meta[f"pareto_front_accuracy_vs_{key}"] = fr
                w(f"<h3>{esc(heading)}</h3><p class='note'>Frontier: {esc(', '.join(title(n) for n in fr))}. "
                  f"{len(pts)} of {len(models)} models measured.</p>")
                w("<div class='card'>" + pareto_chart(pts, fr, {n: title(n) for n in pts}, x_label, fmt) + "</div>")

    # Speed
    timed = [n for n in ranked if n in metas]
    if timed:
        by_speed = sorted(timed, key=lambda n: per_item(metas[n]))
        vals = [per_item(metas[n]) for n in by_speed]
        lo_t = 10 ** math.floor(math.log10(min(vals)))
        hi_t = 10 ** math.ceil(math.log10(max(vals)))
        ticks = [10 ** e for e in range(round(math.log10(lo_t)), round(math.log10(hi_t)) + 1)]
        rows = [{"name": title(n), "value": per_item(metas[n]),
                 "tip": f"{title(n)}: {secs(per_item(metas[n]))} per item, {metas[n]['items_per_second']:.1f} items/s, "
                        f"{secs(metas[n]['classify_seconds'])} for {metas[n]['n']} items"} for n in by_speed]
        w("<h2 id='speed'>Speed</h2><p class='note'>Average time per item = classification time ÷ items, with each app's "
          "own batching or concurrency (a throughput figure, not single-request latency). Log scale. Model loading and the "
          "one-item warm-up are excluded and shown in the table.</p>")
        w("<div class='card'>" + dot_plot(rows, lo=lo_t, hi=hi_t, log=True, fmt=secs, ticks=ticks,
                                         x_title="time per item (log scale), faster to the left") + "</div>")
        w("<h3>Timings</h3><div class='card scroll'><table><thead><tr><th>App</th><th class='l'>Config</th><th>Load</th><th>Warm-up</th>"
          "<th>Training</th><th>Classification</th><th>Per item</th><th>Items/s</th><th>Process wall time</th><th>Server start</th>"
          "<th class='l'>Model</th>"
          "</tr></thead><tbody>")
        for n in by_speed:
            meta = metas[n]
            warm_items = (meta.get("timing") or {}).get("warmup_items", 1)
            w(f"<tr><td>{link(n)}</td><td class='l nw'>{esc(perf.config_text(meta))}</td><td>{secs(meta['load_seconds'])}</td>"
              f"<td data-tip='{warm_items} article(s)'>{secs(meta['warmup_seconds'])}</td>"
              f"<td>{secs(training_seconds(meta))}</td><td>{secs(meta['classify_seconds'])}</td><td>{secs(per_item(meta))}</td>"
              f"<td>{num(meta['items_per_second'], 1)}</td><td>{secs(meta.get('runner', {}).get('wall_seconds'))}</td>"
              f"<td>{secs(meta.get('runner', {}).get('server_start_seconds'))}</td>"
              f"<td class='l'><code>{esc(meta['model'])}</code></td></tr>")
        w("</tbody></table></div><p class='note'>Process wall time covers the whole evaluate.py run (imports, data, load, "
          "warm-up, classification, report) and is recorded when the runner runs the app. Server start: time for a "
          "server the runner launched (OpenJev) to load its model and answer, outside the app's own timings. Training: "
          "supervised baselines only (embedding the train articles + fitting), excluded from classification time; see "
          "<a href='#training'>Training</a>. Config: the batch size or concurrency the main run used. Warm-up: one full "
          "batch for batched apps (so GPU shape compilation is not timed), otherwise one article.</p>")

        # Latency
        w("<h3 id='latency'>Latency</h3><p class='note'><b>Interactive</b>: each of the first articles classified on its "
          "own (batch 1 / concurrency 1), timed end to end, as a single user would wait. <b>Per call</b>: the main run's "
          "calls: one per batch for batched apps, one per request for HTTP apps (under their concurrency, so it "
          "includes queueing and is not interactive latency). Percentiles need enough samples: p90 from 20, p95 from "
          "100, p99 from 1,000; fewer shows –.</p>")
        w("<div class='card scroll'><table><thead><tr><th>App</th><th class='l'>Config</th><th>Items/s</th>"
          "<th>Interactive n</th><th>p50</th><th>p90</th><th>p95</th><th>p99</th><th>max</th>"
          "<th class='l'>Per call</th><th>n</th><th>p50</th><th>p95</th></tr></thead><tbody>")
        for n in sorted(by_speed, key=lambda n: perf.interactive(metas[n]).get("p50") or math.inf):
            it, pc = perf.interactive(metas[n]), perf.per_call(metas[n])
            w(f"<tr><td>{link(n)}</td><td class='l nw'>{esc(perf.config_text(metas[n]))}</td>"
              f"<td>{num(metas[n]['items_per_second'], 1)}</td><td>{it.get('count', '–')}</td>"
              + "".join(f"<td>{secs(it.get(k))}</td>" for k in ("p50", "p90", "p95", "p99", "max"))
              + f"<td class='l'>{esc(pc.get('unit', '–'))}</td><td>{pc.get('count', '–')}</td>"
              f"<td>{secs(pc.get('p50'))}</td><td>{secs(pc.get('p95'))}</td></tr>")
        w("</tbody></table></div>")

        # Sweeps
        swept = {n: perf.sweep_points(metas[n]) for n in by_speed if perf.sweep_points(metas[n])}
        if swept:
            w("<h3 id='sweeps'>Batch size and concurrency sweeps</h3><p class='note'>Each app timed on the same "
              "articles at several batch sizes (batched apps) or concurrency levels (HTTP apps), in one process. "
              "Throughput rises with larger batches or more requests in flight until the hardware or server saturates; "
              "latency per call rises with it. Latency per call: batch p50, or concurrency ÷ throughput for HTTP apps "
              "(average time a request is in flight). Agreement: share of predictions identical to the batch 1 / "
              "concurrency 1 setting (below 100% means batching changes results). Sweeps and interactive latency "
              "use test articles outside the sample, so no article is sent twice (servers such as Ollama cache "
              f"repeated prompts). Ollama parallelism: {esc(env.get('ollama_num_parallel', '–'))}; "
              "concurrency above it only queues.</p>")
            w("<div class='multiples'>" + "".join(sweep_multiple(title(n), pts) for n, pts in swept.items()) + "</div>")
            w("<div class='card scroll' style='margin-top:12px'><table><thead><tr><th>App</th><th class='l'>Setting</th>"
              "<th>Items/s</th><th>Latency per call</th><th>Agreement</th></tr></thead><tbody>")
            for n, pts in swept.items():
                for p_ in pts:
                    w(f"<tr><td>{link(n)}</td><td class='l'>{esc(p_['axis'].replace('_', ' '))} {p_['value']}</td>"
                      f"<td>{num(p_['items_per_second'], 1)}</td><td>{secs(p_['latency'])}</td>"
                      f"<td class='{'' if p_['agreement_with_first'] == 1 else 'bad'}'>{pct(p_['agreement_with_first'])}</td></tr>")
            w("</tbody></table></div>")

        # Memory
        w("<h3 id='resources'>Memory and model size</h3><p class='note'><b>Peak footprint</b>: the app process's "
          "peak physical memory footprint (macOS <code>/usr/bin/time -l</code>), which on Apple Silicon includes GPU "
          "memory; <b>RSS</b> does not. <b>In-process GPU</b>: MLX peak, or PyTorch MPS allocation sampled after each "
          "call (a lower bound). <b>Server</b>: model memory held by the server the app talks to: Ollama's loaded model "
          "(/api/ps, with the GPU share) or the OpenJev server's peak RSS (polled). <b>Model on disk</b>: download size "
          "(Hugging Face cache or Ollama). Remote APIs have no local model.</p>")
        w("<div class='card scroll'><table><thead><tr><th>App</th><th class='l'>Runs on</th><th>Peak footprint</th>"
          "<th>Peak RSS</th><th>In-process GPU</th><th>Server model memory</th><th>of it on GPU</th>"
          "<th>App + server</th><th>Model on disk</th></tr></thead><tbody>")
        for n in sorted(by_speed, key=lambda n: perf.memory(metas[n])["total"] or math.inf):
            m_ = perf.memory(metas[n])
            w(f"<tr><td>{link(n)}</td><td class='l nw'>{esc(apps[n]['runs_on'])}</td><td>{size_txt(m_['footprint'])}</td>"
              f"<td>{size_txt(m_['rss'])}</td><td data-tip='{esc(m_['gpu_method'] or '')}'>{size_txt(m_['gpu'])}</td>"
              f"<td data-tip='{esc(m_['server_method'] or '')}'>{size_txt(m_['server'])}</td><td>{size_txt(m_['server_gpu'])}</td>"
              f"<td>{size_txt(m_['total'])}</td><td>{size_txt(m_['model_disk'])}</td></tr>")
        w("</tbody></table></div>")

        # Tokens and cost
        w("<h3 id='cost'>Requests, tokens and cost</h3><p class='note'>HTTP apps count their own requests, retries, "
          "failures and timeouts; jev's SDK retries internally, so its retries are not observable (–). Tokens as "
          "reported by the server (Ollama, jev). Cost basis: <b>measured</b> = billed amount from the API; "
          "<b>estimated</b> = tokens or requests × the price in runner/pricing.json; <b>unknown</b> = paid app "
          "without a configured price; <b>none</b> = no per-request charge (compute time is in the tables above).</p>")
        w("<div class='card scroll'><table><thead><tr><th>App</th><th>Requests</th><th>Retries</th><th>Failures</th>"
          "<th>Timeouts</th><th>Input tokens / item</th><th>Output tokens / item</th><th>Output tokens/s</th>"
          "<th class='l'>Cost basis</th><th>Per item</th><th>Per 1,000</th></tr></thead><tbody>")
        cost_of_app = lambda n: getattr(APPS_BY_NAME.get(n), "cost", "free")  # noqa: E731
        for n in by_speed:
            r_ = metas[n].get("requests") or {}
            c_ = perf.cost(n, cost_of_app(n), metas[n], pricing)
            items = metas[n]["n"]
            per = lambda v: "–" if v is None else f"{v / items:.1f}"  # noqa: E731
            w(f"<tr><td>{link(n)}</td><td>{r_.get('requests', '–')}</td>"
              f"<td>{'–' if r_.get('retries') is None else r_['retries']}</td><td>{r_.get('failures', '–')}</td>"
              f"<td>{r_.get('timeouts', '–')}</td><td>{per(r_.get('input_tokens'))}</td><td>{per(r_.get('output_tokens'))}</td>"
              f"<td>{num(r_.get('output_tokens_per_second'), 1)}</td>"
              f"<td class='l' data-tip='{esc(c_['note'])}'>{esc(c_['basis'])}</td><td>{money_txt(c_, 'per_item')}</td>"
              f"<td>{money_txt(c_, 'per_1000')}</td></tr>")
        w("</tbody></table></div>")

    # Training (supervised baselines)
    trained = [n for n in models if metas.get(n, {}).get("training")]
    if trained:
        w("<h2 id='training'>Supervised baselines: training</h2><p class='note'>These apps learn from labeled data: "
          "the task's train split, never the test set. Their main hyperparameter is chosen by 3-fold "
          "cross-validation on the train data only (Selected column; ⚠ = the choice sits at the edge of the "
          "candidate grid and CV accuracy is still rising there by more than one SD, so the grid is too narrow; "
          "\"edge, flat\" = at the edge but within noise of its neighbour). "
          "Zero-shot apps above use no labeled data of this task, so the gap between the two shows what the extra model "
          "complexity buys over a cheap supervised model. Train embedding, training and classification are timed "
          "separately; a cached train embedding or fitted model shows the time it took when it was computed.</p>")
        w("<div class='card scroll'><table><thead><tr><th>App</th><th>Accuracy</th><th class='l'>Train data</th>"
          "<th>Train articles</th><th class='l'>Selected</th><th>Train embedding</th><th>Training (fit)</th><th>Classification</th>"
          "<th class='l'>Probabilities</th><th>Test texts in train</th><th class='l'>Hyperparameters</th></tr></thead><tbody>")
        for n in trained:
            meta, tr = metas[n], metas[n]["training"]
            embed = tr.get("train_embed_seconds")
            embed_txt = "–" if embed is None else secs(embed) + (" (cached)" if tr.get("train_embed_cached") else "")
            params = json.dumps(tr.get("hyperparameters", {}), indent=1, sort_keys=True)
            fit_txt = secs(tr.get("train_seconds")) + (" (cached)" if tr.get("train_cached") else "")
            sel = tr.get("hyperparameter_selection")
            if not sel:
                sel_txt = "fixed (no selection recorded)" if tr.get("probabilities") != "temperature-scaled on held-out train" \
                    else "temperature on held-out train"
                sel_tip = ""
            elif "cv_accuracy" in sel:
                edge = " ⚠" if sel.get("edge_still_improving") else (" (edge, flat)" if sel.get("at_grid_edge") else "")
                sel_txt = f"{sel['param']} = {sel['chosen']:g} (CV){edge}"
                sel_tip = "; ".join(f"{sel['param']}={k}: {float(v):.4f}" for k, v in sel["cv_accuracy"].items())
            else:
                sel_txt, sel_tip = f"{sel['param']} = {sel['chosen']:g} ({sel.get('method', '')})", ""
            w(f"<tr><td>{link(n)}</td><td>{pct(metrics[n]['accuracy'])}</td><td class='l'>{esc(tr.get('train_split'))}</td>"
              f"<td>{tr.get('n_train', 0):,}</td><td class='l' data-tip='{esc(sel_tip)}'>{esc(sel_txt)}</td>"
              f"<td>{esc(embed_txt)}</td><td>{esc(fit_txt)}</td>"
              f"<td>{secs(meta['classify_seconds'])}</td>"
              f"<td class='l'><b>{esc(tr.get('probabilities', '–'))}</b>: {esc(tr.get('calibration', ''))}</td>"
              f"<td>{esc(tr.get('test_texts_in_train', '–'))}</td>"
              f"<td class='l'><details><summary>{len(tr.get('hyperparameters', {}))} values</summary>"
              f"<pre>{esc(params)}</pre></details></td></tr>")
        w("</tbody></table></div><p class='note'>Probabilities: <b>native</b> = the model's own predict_proba; "
          "<b>calibrated</b> = fitted afterwards on train data only (cross-validation); <b>uncalibrated</b> = a score "
          "that ranks labels but is not a probability, so its ECE, Brier and log loss are not meaningful as calibration. "
          "Test texts in train: test items whose text also appears verbatim in the train split (a known issue in "
          "some corpora, AG News among them).</p>")

    # Calibration caveats per app (used by the confidence tables and reliability diagrams).
    def warnings_for(n: str) -> list[str]:
        app = APPS_BY_NAME.get(n)
        return calibration_warnings(frames[n], metrics[n]["probs"],
                                    confidence_rule=app.confidence_rule if app else "max_p",
                                    probabilities=metas.get(n, {}).get("training", {}).get("probabilities"))
    cal_warnings = {n: warnings_for(n) for n in ranked}

    # Confidence
    if ranked:
        w("<h2 id='confidence'>Confidence</h2><p class='note'>How useful each app's confidence is for auto-accepting "
          "predictions and routing the rest to review. AUROC: chance that a correct prediction has higher confidence than "
          "a wrong one (0.5 = useless, 1 = perfect). ECE: average gap between confidence and actual accuracy (lower is "
          "better). Brier and log loss score the full label distribution (lower is better).</p>"
          + callout("AUROC for confidence measures whether correct predictions tend to receive higher confidence "
                    "than errors.", "it is not class-vs-class ROC-AUC, and a high AUROC says nothing about whether "
                    "the confidence values themselves are calibrated."))
        w("<div class='card scroll'><table><thead><tr><th>App</th><th class='l'>Confidence definition</th>"
          "<th>Mean (correct)</th><th>Mean (wrong)</th><th>AUROC</th><th>ECE</th><th>Brier</th><th>Log loss</th>"
          + "".join(f"<th>≥ {t:.1f} coverage</th><th>≥ {t:.1f} accuracy</th>" for t in THRESHOLDS)
          + "</tr></thead><tbody>")
        for n in sorted(ranked, key=lambda n: -(metrics[n]["auroc"] if not math.isnan(metrics[n]["auroc"]) else -1)):
            m = metrics[n]
            w(f"<tr><td>{link(n)}</td><td class='l'>{esc(conf_text(n))} {conf_badge(n)}</td>"
              f"<td>{num(m['conf_mean_correct'], 2)}</td><td>{num(m['conf_mean_wrong'], 2)}</td><td>{num(m['auroc'])}</td>"
              f"<td>{num(m['ece'])}</td><td>{num(m.get('brier'))}</td><td>{num(m.get('log_loss'))}</td>"
              + "".join(f"<td>{pct(g['coverage'])}</td><td>{pct(g['accuracy'])}</td>" for g in m["gating"]) + "</tr>")
        w("</tbody></table></div>")
        y_lo = max(0.0, math.floor(min(min(a for c, a in metrics[n]["curve"] if c >= 0.05) for n in models) * 10) / 10)
        w("<h3>Accuracy vs coverage</h3><p class='note'>Sort predictions by confidence and keep the most confident share: "
          "the line shows the accuracy of what is kept (x = share kept). A useful confidence rises to the left. Dashed = "
          "overall accuracy; dots = thresholds 0.5 / 0.7 / 0.9 (hover for values). Same axes in every panel. Where "
          "several predictions share one confidence, the line uses their expected accuracy in random order, since no "
          "threshold can separate them.</p>"
          + callout("Higher accuracy at the same coverage (equivalently, lower risk) is better.",
                    "choose thresholds on validation data, not on this test report.",
                    "AURC = (1/n) Σ<sub>k=1..n</sub> error rate of the k most confident predictions; "
                    "E-AURC = AURC − AURC of an ideal ordering that ranks every correct prediction first."))
        w("<div class='multiples'>" + "".join(coverage_multiple(title(n), metrics[n], y_lo) for n in models) + "</div>")

        # Selective prediction
        n_items = metrics[ranked[0]]["n"]
        targets = [tg for tg in TARGETS if tg < 0.99 or n_items >= MIN_N_FOR_99]
        w("<h3 id='selective'>Selective prediction</h3><p class='note'>Operating points that do not depend on how each "
          "app scales its confidence. <b>AURC</b>: average error rate of the accepted predictions over every coverage "
          "(lower is better); <b>E-AURC</b>: AURC minus that of a perfect ordering, i.e. the part caused by the "
          "confidence ranking itself (lower is better, 0 = perfect). <b>Coverage at target</b>: the largest share an "
          "actual threshold can auto-accept with accuracy ≥ the target. <b>Held-out</b> (the figure to use): the threshold "
          "is fitted on items this table does not score, with the rule 'Wilson 95% lower bound ≥ target', then "
          "applied unchanged to this sample; the cell shows the coverage and the accuracy the accepted items actually "
          "had (✗ = the target was missed on these items). The fitting items are a separate calibration sample (a "
          "second seed's run, with any overlapping items removed) when the run has one, otherwise 2-fold "
          "cross-fitting on this sample. <i>In-sample</i> columns pick the threshold on the same items they score "
          "(point estimate, or Wilson lower bound); they overstate what a threshold fixed in advance achieves and are "
          "kept only for comparison. <b>Accuracy at coverage</b>: accuracy of the most "
          "confident share; † = that share falls inside a group of equal confidences, so no threshold gives exactly it "
          "(expected accuracy in random order is shown)."
          + ("" if n_items >= MIN_N_FOR_99 else f" The 99% target is omitted: at n = {n_items} even a perfect accepted set "
             f"cannot show 99% (run the common or full tier).") + "</p>")
        w("<div class='card scroll'><table><thead><tr><th>App</th><th>AURC</th><th>E-AURC</th><th>Largest tie</th>"
          + "".join(f"<th>Cov. at {tg:.0%}<br>held-out</th>" for tg in targets)
          + "".join(f"<th>Cov. at {tg:.0%}<br>in-sample (optimistic)</th><th>Cov. at {tg:.0%}<br>in-sample Wilson LB</th>"
                    for tg in targets)
          + "".join(f"<th>Acc. at {c:.0%}</th>" for c in COVERAGES) + "</tr></thead><tbody>")
        for n in sorted(ranked, key=lambda n: metrics[n]["aurc"]):
            m = metrics[n]
            cells = ""
            for tg in targets:
                h = m["held_out"][tg]
                tip = (f"{h['accepted']} of {h['n']} accepted, {pct(h['accuracy'])} correct; {held_out_txt(h)}"
                       if h["accepted"] else f"no threshold accepted any item; {held_out_txt(h)}")
                cells += (f"<td data-tip='{esc(tip)}'>{pct(h['coverage'])}"
                          + ("" if h["met"] or not h["accepted"] else " ✗") + "</td>")
            for tg in targets:
                for kind in ("optimistic", "certified"):
                    r = m["at_target"][tg][kind]
                    tip = (f"confidence ≥ {r['threshold']:.4g}: {r['accepted']} accepted, {pct(r['accuracy'])} correct"
                           if r["threshold"] is not None else "no threshold reaches the target")
                    cells += f"<td data-tip='{esc(tip)}'>{pct(r['coverage'])}</td>"
            cells += "".join(f"<td class='{'' if a['reachable'] else 'dag'}'>{pct(a['accuracy'])}</td>" for a in m["at_coverage"])
            tie = m["largest_tie"]
            w(f"<tr class='{row_cls(n)}'><td>{link(n)}</td><td>{num(m['aurc'])}</td><td>{num(m['eaurc'])}</td>"
              f"<td class='{'bad' if tie >= n_items * 0.25 else ''}'>{tie}</td>{cells}</tr>")
        w("</tbody></table></div><p class='note'>Largest tie: most predictions sharing one confidence value (red when "
          "a quarter of the sample or more). Hover a coverage cell for its threshold and accepted count.</p>" + back_link())

        # Calibration
        w("<h3 id='calibration'>Calibration</h3><p class='note'>Does the reported confidence match the observed accuracy? "
          f"<b>ECE</b>: average gap over 10 confidence bins. <b>MCE</b>: largest gap, only over bins with at least "
          f"{MCE_MIN_BIN} predictions (bins used shown; smaller bins are too noisy). <b>Classwise ECE</b>: the same check "
          "for every label's probability p_label (not only the chosen one), averaged over labels. All lower is better. "
          "⚠ marks apps whose numbers do not measure probability calibration.</p>"
          + callout("Lower ECE is better.", "the value depends on the binning and can hide class-specific errors; "
                    "compare classwise ECE as well.",
                    "ECE = Σ<sub>b</sub> (n<sub>b</sub> / n) · |accuracy<sub>b</sub> − mean confidence<sub>b</sub>| over 10 "
                    "equal-width confidence bins; MCE = max<sub>b</sub> of the same gap over bins with enough predictions."))
        w("<div class='card scroll'><table><thead><tr><th>App</th><th>ECE</th><th>MCE</th><th>Bins used</th>"
          "<th>Classwise ECE</th><th>Brier</th><th>Log loss</th><th class='l'>Warnings</th></tr></thead><tbody>")
        for n in sorted(ranked, key=lambda n: (bool(cal_warnings[n]), metrics[n]["ece"])):
            m = metrics[n]
            w(f"<tr class='{row_cls(n)}'><td>{link(n)}</td><td>{num(m['ece'])}</td><td>{num(m['mce'])}</td>"
              f"<td>{m['mce_bins']}</td><td>{num(m['classwise_ece'])}</td><td>{num(m.get('brier'))}</td>"
              f"<td>{num(m.get('log_loss'))}</td><td class='l'>{esc('; '.join(cal_warnings[n])) or '–'}</td></tr>")
        w("</tbody></table></div>")
        w("<h3>Reliability diagrams</h3><p class='note'>Each dot is one confidence bin: x = mean confidence, y = accuracy, "
          "area ∝ number of predictions; bars below show the count per bin (hover for values). Dots on the dashed "
          "diagonal are perfectly calibrated; below it = overconfident. Bins with few predictions move a lot by chance.</p>"
          + callout("Closer to the diagonal is better.", "a good-looking diagram on few examples is still uncertain."))
        w("<div class='multiples'>" + "".join(reliability_multiple(title(n), metrics[n], cal_warnings[n]) for n in models) + "</div>")

    # Per-class
    if models:
        per_label = n_items / len(LABELS)
        w("<h2 id='classes'>Per-class F1</h2><p class='note'>Precision / recall / F1 per label; hover a cell for precision "
          "and recall.</p>"
          + (f"<div class='warn'><strong>About {per_label:.0f} items per label:</strong> each per-class score rests on "
             "a handful of items, so a single error moves it by a large step. Use these cells to spot gross failures, "
             "not to compare classifiers on a label (run the common or full tier for that).</div>"
             if per_label < 10 else "")
          + "<div class='card scroll'><table><thead><tr><th>App</th>"
          + "".join(f"<th>{esc(l)}</th>" for l in LABELS) + "<th>Macro F1</th></tr></thead><tbody>")
        for n in models:
            pc = metrics[n]["per_class"]
            cells = "".join(
                f"<td data-tip='{esc(l)}: precision {num(pc[l]['precision'])}, recall {num(pc[l]['recall'])}'>{num(pc[l]['f1'])}</td>"
                for l in LABELS)
            w(f"<tr><td>{link(n)}</td>{cells}<td>{num(metrics[n]['macro_f1'])}</td></tr>")
        w("</tbody></table></div>")

        w("<h2 id='confusion'>Confusion matrices</h2><p class='note'>Rows = true label, columns = predicted.</p><div class='cm'>")
        for n in models:
            cm = metrics[n]["confusion"]
            body = "".join(
                f"<tr><th>{esc(t)}</th>" + "".join(
                    f"<td class='{'hitcell' if t == p else ''}'>{int(cm.loc[t, p])}</td>" for p in LABELS) + "</tr>"
                for t in LABELS)
            w(f"<div class='card'><h3 style='margin-top:0'>{esc(title(n))}</h3><table><thead><tr><th></th>"
              + "".join(f"<th>{esc(p)}</th>" for p in LABELS) + f"</tr></thead><tbody>{body}</tbody></table></div>")
        w("</div>")

    # Hardest articles
    if any(n not in virtual for n in frames):
        names = [n for n in frames if n not in virtual]
        ref = frames[names[0]]
        wrong = sum((frames[n]["label"] != frames[n]["pred"]).astype(int) for n in names)
        hardest = wrong.sort_values(ascending=False, kind="stable").head(12)
        w(f"<h2 id='hardest'>Hardest articles</h2><p class='note'>Articles misclassified by the most apps "
          f"(of {len(names)}). Wrong answers across apps are listed with how many apps gave each.</p>")
        w("<div class='card scroll'><table><thead><tr><th>Wrong</th><th>True</th><th class='l'>Wrong answers</th>"
          "<th class='l'>Article</th></tr></thead><tbody>")
        for i, count in hardest.items():
            if count == 0:
                break
            preds = pd.Series([frames[n].at[i, "pred"] for n in names if frames[n].at[i, "pred"] != ref.at[i, "label"]]).value_counts()
            w(f"<tr><td>{count}/{len(names)}</td><td>{esc(ref.at[i, 'label'])}</td>"
              f"<td class='l'>{esc(', '.join(f'{p} ×{c}' for p, c in preds.items()))}</td>"
              f"<td class='l'>{esc(ref.at[i, 'text'][:220])}</td></tr>")
        w("</tbody></table></div>")

    # Repeated runs: every valid run of every app, across seeds and repeats.
    repeated = {n: [r for r in a.get("runs", []) if r.get("status") in ("ok", "existing") and r.get("csv")]
                for n, a in apps.items() if not a.get("virtual") and n not in incompatible}
    repeated = {n: rs for n, rs in repeated.items() if len(rs) > 1}
    if repeated:
        w("<h2 id='repeats'>Repeated runs</h2><p class='note'>Every valid run of each app (all seeds and repeats). "
          "Different seeds are different samples, so accuracy spread across seeds shows sampling variation; repeats "
          "on the same seed show run-to-run variation. SD is the sample standard deviation; IQR = Q3 − Q1. "
          "Stability: share of articles given the same label in every run on the same seed (a deterministic model "
          "scores 100%), and mean pairwise agreement between those runs.</p>")
        w("<div class='card scroll'><table><thead><tr><th>App</th><th>Runs</th><th>Seeds</th>"
          "<th>Accuracy mean</th><th>median</th><th>SD</th><th>IQR</th><th>Macro F1 mean</th>"
          "<th>Per item mean</th><th>median</th><th>SD</th><th>IQR</th><th>Items/s mean</th>"
          "<th>Interactive p50 median</th><th>SD</th>"
          "<th>Same label, all runs</th><th>Pairwise agreement</th></tr></thead><tbody>")
        for n, rs in sorted(repeated.items(), key=lambda kv: title(kv[0])):
            run_frames = [pd.read_csv(run_dir / r["csv"]) for r in rs]
            run_metas = [json.loads((run_dir / r["run_json"]).read_text()) if r.get("run_json") else {} for r in rs]
            ms = [app_metrics(df) for df in run_frames]
            acc, f1 = spread([m["accuracy"] for m in ms]), spread([m["macro_f1"] for m in ms])
            per = spread([per_item(meta) for meta in run_metas])
            ips = spread([meta.get("items_per_second") for meta in run_metas])
            ip50 = spread([perf.interactive(meta).get("p50") for meta in run_metas])
            by_seed: dict[int, list[pd.Series]] = {}
            for r, df in zip(rs, run_frames):
                by_seed.setdefault(r["seed"], []).append(df["pred"])
            stab = [prediction_stability(p) for p in by_seed.values() if len(p) > 1]
            unanimous = statistics.mean(s[0] for s in stab) if stab else None
            pairwise = statistics.mean(s[1] for s in stab) if stab else None
            flag = "" if unanimous is None or unanimous == 1 else " class='bad'"
            w(f"<tr><td>{link(n)}</td><td>{len(rs)}</td><td>{len(by_seed)}</td>" + spread_cells(acc, pct)
              + f"<td>{num(f1['mean'])}</td>" + spread_cells(per, secs) + f"<td>{num(ips['mean'], 1)}</td>"
              f"<td>{secs(ip50['median'])}</td><td>{secs(ip50['sd'])}</td>"
              f"<td{flag}>{pct(unanimous)}</td><td>{pct(pairwise)}</td></tr>")
        w("</tbody></table></div><p class='note'>Stability is blank when no seed has two or more runs "
          "(use <code>--repeat</code>). Invalid or failed runs are not counted; see Run status.</p>")

    # Incompatible results: kept out of every comparison above.
    if excluded:
        w("<h2 id='incompatible'>Incompatible results</h2><p class='note'>These results were not produced on the "
          "reference sample (different texts, labels, order or sample hash), so they are left out of rankings, "
          "significance tests, charts, tiles and cross-app analyses. Their accuracy is on their own sample only.</p>")
        w("<div class='card scroll'><table><thead><tr><th>App</th><th class='l'>Why</th><th>Articles</th>"
          "<th>Seed</th><th>Sample SHA-1</th><th>Accuracy (own sample)</th></tr></thead><tbody>")
        for n, (df, meta) in excluded.items():
            acc = float((df["label"] == df["pred"]).mean()) if len(df) else float("nan")
            w(f"<tr><td>{link(n)}</td><td class='l'>{esc(incompatible[n])}</td><td>{len(df)}</td>"
              f"<td>{esc(meta.get('seed', '–'))}</td><td><code>{esc(frame_sha1(df)[:12])}</code></td><td>{pct(acc)}</td></tr>")
        w("</tbody></table></div>")

    # Run status
    w("<h2 id='status'>Run status</h2><div class='card scroll'><table><thead><tr><th>App</th><th>Status</th>"
      "<th class='l'>Reason / note</th><th>Finished</th><th class='l'>Log</th></tr></thead><tbody>")
    for name, a in apps.items():
        meta = metas.get(name) or excluded.get(name, (None, {}))[1]
        status = "incompatible" if name in incompatible else a["status"]
        reason = a.get("reason", "") if name not in incompatible else f"different sample: {incompatible[name]}"
        bad_runs = [r for r in a.get("runs", []) if r.get("status") in ("failed", "invalid")]
        if bad_runs:
            reason += f" ({len(bad_runs)} of {len(a['runs'])} runs failed or invalid: " + "; ".join(
                f"seed {r['seed']} run {r['rep']}: {r.get('reason', r['status'])}" for r in bad_runs[:3]) + ")"
        w(f"<tr><td>{esc(a['title'])}</td><td><span class='pill {esc(status)}'>{esc(status)}</span></td>"
          f"<td class='l'>{esc(reason)}</td><td>{esc(meta.get('finished_at', '–'))}</td>"
          f"<td class='l'>{'<code>' + esc(a['log']) + '</code>' if a.get('log') else '–'}</td></tr>")
    w("</tbody></table></div>")

    # Environment
    w("<h2 id='env'>Environment and versions</h2><div class='card scroll'><table><tbody>")
    cond = env.get("conditions_at_start") or {}
    power = (f"{cond.get('power')} ({cond.get('battery_percent')}%)" if cond.get("on_battery") else cond.get("power")) or "–"
    for k, v in [("Machine", env.get("machine")), ("Memory", f"{env.get('memory_gb')} GB"), ("OS", env.get("os")),
                 ("Power at start", power + (", Low Power Mode" if cond.get("low_power_mode") else "")),
                 ("Load average at start (1 min)", cond.get("load_1m", "–")),
                 ("Ollama", env.get("ollama_version") or "not running"),
                 ("Ollama parallel requests", env.get("ollama_num_parallel", "–"))]:
        w(f"<tr><td>{esc(k)}</td><td class='l'>{esc(v)}</td></tr>")
    for app_name, om in env.get("ollama_models", {}).items():
        w(f"<tr><td>Ollama model ({esc(app_name)})</td><td class='l'><code>{esc(om['tag'])}</code> {esc(om['digest'])}"
          + (f" at {esc(om['host'])}" if om.get("host") else "") + "</td></tr>")
    for k, v in env.get("pinned_commits", {}).items():
        w(f"<tr><td>Pinned commit: {esc(k)}</td><td class='l'><code>{esc(v or '–')}</code></td></tr>")
    w("</tbody></table></div>")
    if metas:
        w("<h3>Per-app Python and packages</h3><div class='card scroll'><table><thead><tr><th>App</th><th>Python</th>"
          "<th class='l'>Packages</th></tr></thead><tbody>")
        for n in ranked:
            if n in metas:
                pk = ", ".join(f"{k} {v}" for k, v in metas[n]["packages"].items())
                w(f"<tr><td>{link(n)}</td><td>{esc(metas[n]['python'])}</td><td class='l'>{esc(pk)}</td></tr>")
        w("</tbody></table></div>")

    # Definitions
    task_tiers = tasks.tiers(task)
    strategy = task.manifest.get("sampling", "balanced")
    how = ("n // k items per label (k labels), the n mod k leftover items going one each to the first labels in "
           "alphabetical order" if strategy == "balanced" else
           "each label's share of the test set times n, rounded down, the leftover items going to the labels with the "
           "largest remainders, so rare labels stay rare")
    sample_def = (f"<li><b>Sample and tiers</b>: a {strategy} stratified sample of the {esc(task.name)} test split: {how}, "
                  "drawn with the run seed. Tiers: " + ", ".join(f"<i>{k}</i> = {v:,}" for k, v in task_tiers.items())
                  + "; other sizes are <i>custom</i>. Only results on exactly this sample (same texts, labels, order and "
                  "SHA-1) are compared. Calibration data for zero-shot apps comes from the train split, never from test.</li>")
    w("""<h2 id='defs'>Definitions</h2><div class='card'><ul class='note' style='max-width:none'>
{SAMPLE_DEF}
<li><b>Validation</b>: before a result is accepted, the runner checks columns, row count, labels, unique sample ids,
probabilities (finite, in [0, 1], rows summing to 1), pred = argmax p, the app's confidence rule, that the files were
written by this run, and that CSV, metadata, seed, n and sample hash agree. Failing results are shown as
<i>invalid</i> and never ranked.</li>
<li><b>Accuracy</b>: share of articles whose predicted label equals the true label. <b>95% CI</b>: Wilson score interval.</li>
<li><b>Macro F1</b>: unweighted mean of the per-class F1 scores; the primary metric for imbalanced tasks, where
accuracy mostly reflects the frequent labels.</li>
<li><b>McNemar p</b>: exact two-sided McNemar test against the top point estimate, on the articles where exactly one
of the two apps is right. <b>Holm p</b>: the same p-values adjusted with Holm's step-down method over all models tested
(family-wise error rate 5%). <b>Rank interval</b>: 2.5th–97.5th percentile of each model's accuracy rank over the
paired bootstrap resamples (ties share the better rank); <b>P(best)</b>: share of resamples where it ranks first.
<b>Bootstrap leading tier</b>: models whose percentile bootstrap rank interval includes 1. Because the top point
estimate is itself chosen from this data, testing only against it would make the others look further behind than they
are; the rank interval reduces, but does not remove, that effect. It is an uncertainty summary, not a test that the
models are equivalent or statistically tied. The report names no single winner when the tier has more than one app.</li>
<li><b>Non-significant results</b>: an interval containing 0 or a Holm p of at least 0.05 means the procedure did not
detect a difference at the 5% level, not that the apps are equivalent.</li>
<li><b>Paired difference CI</b>: accuracy difference d = (only-this-right − only-top-right) / n with the Tango (1998)
score interval, which stays valid when few articles are discordant (a Wald interval collapses to 0–0 for two apps
that always agree). <b>Bootstrap CI</b>: 2.5th–97.5th percentile over paired resamples of the articles (every app
scored on the same resample); the seed and replicate count are in <code>report_meta.json</code>.</li>
<li><b>Pareto frontier</b>: apps not dominated on (higher accuracy, lower time per item); the same for interactive
latency and for memory where measured.</li>
<li><b>Interactive latency</b>: seconds to classify one article on its own (batch 1, concurrency 1), end to end, after
warm-up. <b>Throughput</b>: items per second with the app's own batching or concurrency. The two answer different
questions: one user waiting vs a queue of work.</li>
<li><b>Peak memory footprint</b>: macOS physical footprint of the app process at its peak (includes GPU memory on
Apple Silicon). <b>App + server</b>: footprint plus the model memory of the Ollama or OpenJev server the app uses.</li>
<li><b>Load</b>: constructing the classifier (weights into memory for local apps; client setup for API/server apps).
<b>Warm-up</b>: one article before timing, which absorbs model loading inside Ollama/servers, kernel compilation and
the first connection. <b>Classification</b>: the timed run over all articles.</li>
<li><b>Zero-shot / supervised / virtual</b>: zero-shot apps decide from the label names or descriptions, with no
weights or label ranking fitted on task examples. Some (confidence <i>task-calibrated</i>) fit one softmax temperature on
labeled train rows: a single positive temperature does not change the predicted label, so their accuracy stays
zero-shot, but their confidence and calibration metrics are task-calibrated;
supervised baselines (TF-IDF + logistic regression, linear SVM, Complement Naive Bayes; bge-m3 embeddings + class
centroids or logistic regression) are trained on the task's train split; virtual baselines are generated
by the runner without a model: <i>majority class</i> always predicts the most frequent train label, <i>stratified
random</i> draws a label from the train class frequencies with a generator seeded by the run seed.</li>
<li><b>Training</b>: embedding the train articles (embedding baselines) plus fitting the vectorizer and classifier;
reported separately and never included in load, warm-up or classification time.</li>
<li><b>Calibrated vs uncalibrated</b>: the linear SVM appears twice. The calibrated variant uses sigmoid (Platt)
calibration fitted by 5-fold cross-validation on the train split; the uncalibrated variant reports softmax of the raw
SVM margins, which ranks labels but is not a probability.</li>
<li><b>Per item</b>: classification time ÷ number of items, at each app's own batch size or concurrency.</li>
<li id='defs-confidence'><b>Confidence</b>: the number each app reports for gating; its definition differs by app (see the
Confidence table). Most apps use max p, the largest of their per-label scores. What those scores mean:
{CONF_KINDS}</li>
<li><b>Auto-accept ≥ t</b>: share of predictions with confidence ≥ t (coverage) and the accuracy on that share.
Fixed thresholds are not comparable across apps whose confidences are scaled differently; prefer coverage at a target
accuracy.</li>
<li><b>Ties</b>: predictions with equal confidence are accepted or rejected together by any threshold. Curves, AURC and
accuracy at a fixed coverage use the expected accuracy with ties in random order, so they never depend on row order.</li>
<li><b>AURC</b>: mean over k = 1..n of the error rate among the k most confident predictions (area under the
risk-coverage curve); lower is better. <b>E-AURC</b>: AURC minus the AURC of an ideal ordering that puts every correct
prediction first; it isolates the confidence ranking from overall accuracy (0 = perfect).</li>
<li><b>Coverage at target accuracy</b>: the largest share of predictions a threshold on confidence can accept while their
accuracy is at least the target, checked at every achievable threshold. <i>Optimistic</i>: point accuracy, threshold chosen
on the test sample (overstates a threshold fixed in advance). <i>Wilson LB</i>: the Wilson 95% lower bound must reach the
target. Both thresholds and bounds are selected on this evaluation sample; they are not a guarantee for future or
shifted data, so validate a chosen threshold on an independent calibration or validation set before deployment. The 99% target is shown only from n = 1,000.</li>
<li><b>MCE</b>: maximum calibration gap over confidence bins with at least 10 predictions. <b>Classwise ECE</b>: for each
label, ECE of p_label against "the true label is this label" over all predictions, averaged over labels.</li>
<li><b>Top-2 accuracy</b>: share of articles whose true label is one of the two highest p_*; ties count in expectation,
so one-hot outputs score as chance for the second slot.</li>
<li><b>AUROC</b>: ranking quality of confidence for separating correct from wrong predictions.
<b>ECE</b>: expected calibration error of the reported confidence, 10 equal-width bins.</li>
<li><b>Brier</b>: mean squared error of the label probability vector (p_* columns) against the one-hot truth.
<b>Log loss</b>: mean −ln p(true label), probabilities floored at 1e-6.</li>
</ul></div>""".replace("{SAMPLE_DEF}", sample_def).replace("{CONF_KINDS}", "<ul>" + "".join(
        f"<li><i>{esc(label)}</i>: {esc(text)}</li>" for label, text in CONFIDENCE_KINDS.values()) + "</ul>"))
    w("<div id='tip' role='tooltip'></div>")
    w(f"<script>{JS}</script></main></body></html>")

    path = run_dir / "report.html"
    page = "".join(out)
    simple = simple_page(task=task, summary=summary, primary=primary, noun=noun, apps=apps, metrics=metrics,
                         metas=metas, ranked=ranked, models=models, virtual=virtual, lead=lead, best=best,
                         title=title, link=link, supervision=supervision, per_item=per_item,
                         approach=approach, remote=is_remote,
                         latency=lambda meta: perf.interactive(meta).get("p50"),
                         slow_conditions=set(on_battery) | set(low_power),
                         example=frames.get(ref_name) if ref_name else None)
    for simple_path, full_path in ((run_dir / "report_simple.html", path), *also):
        simple_path, full_path = Path(simple_path), Path(full_path)
        write_page(page, full_path, {SIMPLE_HREF: simple_path})
        write_page(simple, simple_path, {FULL_HREF: full_path})
    (run_dir / "report_meta.json").write_text(json.dumps(report_meta, indent=2) + "\n")
    return path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(build_report(Path(sys.argv[1])))

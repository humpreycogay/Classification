"""The plain-language summary report: the benchmark results for one task, written to be read like a short chapter.

build_report (report.py) computes every metric once and passes them here; this module only chooses what to show
and how to say it. It covers what was tested, which models did best, speed against accuracy, which labels are hard,
when an answer can be trusted, and the main cautions. Everything else (statistical tests, calibration, latency
percentiles, run details) stays in the full report, which this page links to.

The page uses two placeholders that report.write_page replaces per output file: MODELS_HREF (the models/ folder,
for fonts and model pages) and FULL_HREF (the full report).
"""

from __future__ import annotations

import html
import math
from datetime import datetime

import perf
from stats import wilson

MODELS_HREF = "{{MODELS_HREF}}"
FULL_HREF = "{{FULL_HREF}}"

TASK_TITLES = {"ag_news": "AG News", "banking77": "Banking77"}

# What the task is, in plain words (the manifest description is terse).
TASK_STORY = {
    "ag_news": "AG News is a collection of news articles, each filed under one of four topics: World, Sports, "
               "Business, or Science/Technology. The articles are fairly long and the topics are broad, so a model "
               "usually has plenty of clues to go on. This makes it a good first check of whether a classifier "
               "works at all, but a high score here says little about harder, finer-grained tasks.",
    "banking77": "Banking77 is a collection of short messages that customers sent to an online bank, each tagged "
                 "with one of 77 reasons for getting in touch, such as a lost card, a pending transfer, or a "
                 "question about exchange rates. The messages are brief and many of the reasons are close cousins "
                 "of one another, so telling them apart is much harder than sorting news by topic.",
}

# The approach categories (apps.APPROACHES) in plain words, in the order the summary lists them.
APPROACH_TEXT = {
    "zero-shot": "given only the label names, never an example",
    "in-domain checkpoint": "released models whose own training already included this task's data",
    "classical supervised": "word-count (TF-IDF) or embedding features with a simple classifier, fitted on the "
                            "task's training examples",
    "fine-tuned encoder": "pre-trained language models adapted to the task with its training examples",
    "from scratch": "neural networks trained only on the task's training examples, starting from random weights",
}

EXPLORATORY_BELOW = 1000  # samples smaller than the common tier get the "exploratory" notice

CSS = """
@font-face { font-family: "Inter Tight"; font-style: normal; font-weight: 100 900; font-display: swap;
  src: url("{{MODELS_HREF}}/fonts/InterTight-latin-var.woff2") format("woff2"); }
:root { color-scheme: light;
  --bg:#f1f0eb; --surface:#f7f6f2; --ink:#141413; --muted:#5f5e58; --faint:#6c6b64; --line:#d9d8d1;
  --line-strong:#bebdb5; --accent:#e8922f; --accent-soft:#fff0dd; --mark:#914508; --bar:#c86f16; --track:#e6e5df;
  --base:#9a998f; --sans:"Inter Tight", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark;
  --bg:#131312; --surface:#1a1a18; --ink:#ecebe5; --muted:#a9a8a0; --faint:#8d8c84; --line:#2d2d2a;
  --line-strong:#44443f; --accent-soft:#3a2818; --mark:#f3a34f; --bar:#f3a34f; --track:#2a2a27; --base:#6c6b64; } }
:root[data-theme="dark"] { color-scheme: dark;
  --bg:#131312; --surface:#1a1a18; --ink:#ecebe5; --muted:#a9a8a0; --faint:#8d8c84; --line:#2d2d2a;
  --line-strong:#44443f; --accent-soft:#3a2818; --mark:#f3a34f; --bar:#f3a34f; --track:#2a2a27; --base:#6c6b64; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 400 17px/1.7 var(--sans);
  -webkit-font-smoothing: antialiased; }
main { max-width: 760px; margin: 0 auto; padding: 40px 16px 72px; overflow-wrap: anywhere; }
a { color: inherit; text-decoration: underline; text-decoration-color: var(--line-strong); text-underline-offset: 3px; }
a:hover { text-decoration-color: currentColor; }
:focus-visible { outline: 2px solid var(--mark); outline-offset: 2px; }
.crumb { font-size: 14px; color: var(--muted); margin: 0 0 28px; }
.eyebrow { font-size: 13px; letter-spacing: .08em; text-transform: uppercase; color: var(--mark); font-weight: 600; margin: 0; }
h1 { font-size: 36px; line-height: 1.15; margin: 6px 0 12px; font-weight: 650; letter-spacing: -.01em; }
h2 { font-size: 23px; line-height: 1.3; margin: 48px 0 10px; font-weight: 620; }
p { margin: 0 0 16px; }
.lede { font-size: 19px; color: var(--muted); }
.meta { font-size: 14px; color: var(--faint); }
.glance { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(200px, 100%), 1fr)); gap: 12px; margin: 28px 0 8px; }
.glance div { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
.glance .k { font-size: 13px; color: var(--muted); }
.glance .v { font-size: 26px; font-weight: 650; line-height: 1.25; margin: 2px 0; font-variant-numeric: tabular-nums; }
.glance .s { font-size: 14px; color: var(--faint); line-height: 1.45; }
blockquote { margin: 0 0 16px; padding: 12px 18px; border-left: 3px solid var(--accent); background: var(--surface);
  border-radius: 0 8px 8px 0; color: var(--muted); }
blockquote b { color: var(--ink); font-weight: 600; }
dl.terms { margin: 0 0 16px; }
dl.terms dt { font-weight: 620; margin-top: 10px; }
dl.terms dd { margin: 0; color: var(--muted); }
.table { overflow-x: auto; margin: 16px 0 8px; border: 1px solid var(--line); border-radius: 10px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 15px; font-variant-numeric: tabular-nums; }
th, td { padding: 8px 12px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; vertical-align: middle; }
tr:last-child td { border-bottom: 0; }
th { font-size: 13px; font-weight: 600; color: var(--muted); }
th.l, td.l { text-align: left; }
td.name { white-space: normal; min-width: 170px; }
td.kind { color: var(--muted); font-size: 14px; }
tr.base td { color: var(--faint); }
tr.top td.name { font-weight: 620; }
.bar { display: flex; align-items: center; gap: 10px; justify-content: flex-end; }
.bar span.track { width: 110px; height: 8px; background: var(--track); border-radius: 4px; overflow: hidden; flex: none; }
.bar span.fill { display: block; height: 100%; background: var(--bar); border-radius: 4px; }
tr.base .bar span.fill { background: var(--base); }
.small { font-size: 14px; color: var(--faint); }
.notice { margin: 20px 0 8px; padding: 12px 16px; border: 1px solid var(--line-strong); border-left: 3px solid var(--accent);
  border-radius: 0 8px 8px 0; background: var(--accent-soft); font-size: 15px; line-height: 1.55; }
ul.plain { padding-left: 20px; margin: 0 0 16px; } ul.plain li { margin-bottom: 8px; }
.full { margin-top: 56px; padding: 20px 22px; border: 1px solid var(--line); border-radius: 12px; background: var(--surface); }
.full h2 { margin-top: 0; font-size: 20px; }
.btn { display: inline-block; margin-top: 4px; padding: 9px 16px; border-radius: 8px; background: var(--accent);
  color: #141413; text-decoration: none; font-weight: 600; font-size: 15px; }
.btn:hover { filter: brightness(1.05); }
"""


def esc(value) -> str:
    return html.escape(str(value))


def nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def pct(x, digits: int = 0) -> str:
    return "–" if nan(x) else f"{x * 100:.{digits}f}%"


def secs(x) -> str:
    """A time per item in words a reader can picture."""
    if nan(x) or x <= 0:
        return "–"
    if x < 0.001:
        return f"{x * 1e6:.0f} µs"
    if x < 1:
        return f"{x * 1000:.0f} ms" if x >= 0.01 else f"{x * 1000:.1f} ms"
    if x < 120:
        return f"{x:.1f} s"
    return f"{x / 60:.1f} min"


def size(b) -> str:
    if nan(b) or not b:
        return "–"
    return f"{b / 2**30:.1f} GB" if b >= 2**30 else f"{b / 2**20:.0f} MB"


def join(items: list[str]) -> str:
    """English list: a, b and c."""
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def rough(x: float) -> str:
    """A large ratio rounded to one significant figure: 163,634 -> 200,000."""
    if x < 10:
        return f"{x:.0f}"
    scale = 10 ** int(math.floor(math.log10(x)))
    return f"{round(x / scale) * scale:,.0f}"


def count_word(k: int) -> str:
    words = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    return words[k] if 0 <= k < len(words) else str(k)


def simple_page(*, task, summary: dict, primary: str, noun: str, apps: dict, metrics: dict, metas: dict,
                ranked: list[str], models: list[str], virtual: set[str], lead: list[str], best: str | None,
                title, link, supervision, per_item, approach, remote, latency, slow_conditions=frozenset(),
                example=None) -> str:
    """The summary report as one HTML page (placeholders MODELS_HREF and FULL_HREF still in it).

    title(app) is the display name, link(app) the name linked to its model page, supervision(app) how it learned,
    approach(app) its comparison category (apps.APPROACHES), remote(app) whether it ran on another machine,
    per_item(meta) its batched seconds per item, latency(meta) its single-request median (None if not measured);
    slow_conditions holds apps timed on battery or in Low Power Mode; example is the shared sample (first row shown).
    """
    name = TASK_TITLES.get(task.name, task.name)
    n = summary["n"]
    labels = task.label_ids
    label_names = task.label_names
    started = datetime.fromisoformat(summary["started_at"]).astimezone()
    env = summary.get("environment") or {}
    nouns = f"{noun}s"
    metric_word = "accuracy" if primary == "accuracy" else primary.replace("_", " ")

    def acc(n_: str) -> float:
        return metrics[n_]["accuracy"]

    def time_of(n_: str):
        return per_item(metas[n_]) if n_ in metas else None

    def mem_of(n_: str):
        if n_ not in metas:
            return None
        m = perf.memory(metas[n_])
        return m["total"] or m["footprint"]

    out: list[str] = []
    w = out.append
    w("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
      "<meta name='viewport' content='width=device-width, initial-scale=1'>"
      f"<meta name='description' content='{esc(name)} benchmark results in plain language.'>"
      f"<title>{esc(name)} Benchmark Results</title><style>{CSS}</style></head><body><main>")
    w(f"<p class='crumb'><a href='{MODELS_HREF}/index.html'>Text Classification Benchmark</a> › {esc(name)} results</p>")
    w("<p class='eyebrow'>Benchmark results</p>")
    w(f"<h1>How well do text classifiers sort {esc(name)}?</h1>")
    w(f"<p class='lede'>We gave {len(models)} classifiers the same {n:,} {esc(nouns)} and checked how many each one "
      f"labeled correctly, how fast it worked, and how much memory it needed. The results hold for this sample, "
      f"hardware and setup; the full report has every statistic behind them.</p>")
    w(f"<p class='meta'>Results from {started:%-d %B %Y} · {esc(env.get('machine') or 'one machine')}"
      + (f", {env.get('memory_gb')} GB memory" if env.get("memory_gb") else "")
      + f" · <a href='{FULL_HREF}'>Full raw report</a></p>")

    # ---- exploratory banner: small samples cannot rank close classifiers
    lo90, hi90 = wilson(round(0.9 * n), n)
    half = (hi90 - lo90) / 2 * 100
    per_label = n / len(labels)
    if n < EXPLORATORY_BELOW:
        w(f"<div class='notice'><b>Exploratory results.</b> This run used {n:,} {esc(nouns)}, about "
          f"{per_label:.0f} per label. At this size an accuracy near 90% is only known to within about "
          f"±{half:.0f} points, so read the order of classifiers that are close together as provisional.</div>")

    # ---- at a glance
    timed_lat = {n_: latency(metas[n_]) for n_ in models if n_ in metas and latency(metas[n_])}
    fastest_lat = min(timed_lat, key=timed_lat.get, default=None)
    fastest = min((n_ for n_ in models if time_of(n_)), key=time_of, default=None)
    majority = next((n_ for n_ in ranked if n_ in virtual and "majority" in n_.lower()), None)
    w("<div class='glance'>")
    if best:
        w(f"<div><div class='k'>Highest {esc(metric_word)}</div><div class='v'>{pct(acc(best))}</div>"
          f"<div class='s'>{esc(title(best))}<br>{esc(approach(best))}, on this sample</div></div>")
    if fastest_lat:
        w(f"<div><div class='k'>Fastest single request</div><div class='v'>{secs(timed_lat[fastest_lat])}</div>"
          f"<div class='s'>median, one {esc(noun)} at a time: {esc(title(fastest_lat))} "
          f"({pct(acc(fastest_lat))} correct)</div></div>")
    elif fastest:
        w(f"<div><div class='k'>Fastest average (batched)</div><div class='v'>{secs(time_of(fastest))}</div>"
          f"<div class='s'>per {esc(noun)} with batching: {esc(title(fastest))} ({pct(acc(fastest))} correct)</div></div>")
    if majority:
        w(f"<div><div class='k'>Always picking one label</div><div class='v'>{pct(acc(majority))}</div>"
          f"<div class='s'>the most common label, ignoring the text</div></div>")
    w("</div>")

    # ---- what was tested
    w("<h2>What was tested</h2>")
    w(f"<p>{esc(TASK_STORY.get(task.name, task.manifest.get('description', '')))}</p>")
    if example is not None and len(example):
        row = example.iloc[0]
        text = str(row["text"])
        text = text if len(text) <= 260 else text[:257] + "…"
        w(f"<blockquote>“{esc(text)}”<br><b>Correct label:</b> {esc(label_names.get(row['label'], row['label']))}"
          "</blockquote>")
    w(f"<p>Every classifier was given the same {n:,} {esc(nouns)}, drawn from the test set with "
      + (f"an equal number from each of the {len(labels)} labels" if n % len(labels) == 0
         else f"the {len(labels)} labels spread as evenly as possible")
      + f". None of these {esc(nouns)} were used to train any of the models. Each classifier returned one label per "
      f"{esc(noun)} and, in most cases, a confidence score.</p>")
    counts = {a: sum(1 for n_ in models if approach(n_) == a) for a in APPROACH_TEXT}
    n_remote = sum(1 for n_ in models if remote(n_))
    w("<p>The classifiers differ in how much they were told about the task before the test, and that matters as "
      "much as the model itself when comparing scores:</p><ul class='plain'>"
      + "".join(f"<li><b>{esc(a.capitalize())}</b> ({k}): {esc(APPROACH_TEXT[a])}</li>"
                for a, k in counts.items() if k)
      + "</ul>")
    if n_remote:
        w(f"<p>{count_word(n_remote).capitalize()} of them ran on a remote service rather than the test machine, so "
          f"{'its times include' if n_remote == 1 else 'their times include'} the network and unknown hardware.</p>")

    # ---- main findings
    w("<h2>Main findings</h2>")
    if best:
        if len(lead) > 1:
            lo = min(acc(n_) for n_ in lead)
            spread = (f"all scored {pct(lo)}" if pct(lo) == pct(acc(best)) else f"scored between {pct(lo)} and {pct(acc(best))}")
            w(f"<p>The top {count_word(len(lead))} classifiers {spread}: "
              f"{join([link(n_) for n_ in lead])}. When the {n:,} {esc(nouns)} are resampled (the paired bootstrap in "
              "the full report), each of them comes first often enough that their order here should not be read as a "
              "ranking.</p>")
        else:
            w(f"<p>{link(best)} is the most accurate on this sample, with {pct(acc(best))} of the {esc(nouns)} right "
              f"({metrics[best]['correct']:,} of {n:,}). When the {esc(nouns)} are resampled (the paired bootstrap in "
              "the full report), no other classifier comes first often enough to share the lead.</p>")
        best_by = {}
        for n_ in models:  # models is in accuracy order, so the first per approach is its best
            best_by.setdefault(approach(n_), n_)
        if len(best_by) > 1:
            w("<p>Because the approaches start from such different positions, the fairer comparison is within each "
              "group:</p>")
            w("<div class='table'><table><thead><tr><th class='l'>Approach</th><th class='l'>Best classifier</th>"
              f"<th>Accuracy</th><th>95% CI</th></tr></thead><tbody>"
              + "".join(f"<tr><td class='l'>{esc(a)}</td><td class='l name'>{link(n_)}</td><td>{pct(acc(n_))}</td>"
                        f"<td>{pct(metrics[n_]['ci'][0])}–{pct(metrics[n_]['ci'][1])}</td></tr>"
                        for a, n_ in sorted(best_by.items(), key=lambda kv: -acc(kv[1])))
              + "</tbody></table></div>")
        zs = best_by.get("zero-shot")
        trained = next((n_ for n_ in models if approach(n_) != "zero-shot"), None)
        if zs and trained:
            gap = round((acc(trained) - acc(zs)) * 100)
            if gap > 2:
                w(f"<p>The best zero-shot classifier, {link(zs)}, is {gap} points behind the best one that learned from "
                  f"this task's data ({link(trained)}): roughly what labeled examples are worth on {esc(name)}.</p>")
            elif gap >= -2:
                w(f"<p>The best zero-shot classifier, {link(zs)}, is within {max(abs(gap), 1)} point"
                  f"{'s' if abs(gap) > 1 else ''} of the best one that learned from this task's data ({link(trained)}), "
                  f"so on {esc(name)} labeled examples add little.</p>")
            else:
                w(f"<p>The best zero-shot classifier, {link(zs)}, scored {-gap} points above every classifier trained "
                  f"on this task's data.</p>")
        if majority:
            above = sum(1 for n_ in models if acc(n_) > acc(majority) + 0.10)
            beat = ("every classifier beat that" if above == len(models)
                    else f"{above} of the {len(models)} classifiers beat that")
            w(f"<p>Always answering with the most common label scores {pct(acc(majority))}; {beat} by more than ten "
              "points.</p>")

    # ---- how to read
    w("<h2>How to read the numbers</h2>")
    w("<dl class='terms'>"
      f"<dt>Accuracy</dt><dd>The share of {esc(nouns)} the classifier labeled correctly. 90% means 9 out of 10 "
      "were right.</dd>"
      f"<dt>Accuracy range (95% CI)</dt><dd>A statistical estimate (Wilson 95% confidence interval) of the "
      f"uncertainty in the measured accuracy. With only {n:,} {esc(nouns)}, the true accuracy could plausibly be "
      "higher or lower than the measured figure. Each interval describes one classifier on its own; it is not a test "
      "of whether two classifiers differ, and overlapping intervals do not mean two classifiers are equal. The full "
      "report has paired tests for that.</dd>"
      f"<dt>Average time per {esc(noun)} (batched)</dt><dd>Total classification time divided by the number of "
      f"{esc(nouns)}, with each classifier processing the sample in its usual batches or parallel requests. Model "
      "loading is not included. This measures throughput: it is not how long one request takes.</dd>"
      "<dt>Single request</dt><dd>The median time to classify one item sent on its own, which is what a user waiting "
      "for one answer experiences. A dash means it was not measured in this run.</dd>"
      "<dt>Memory</dt><dd>The most memory the classifier used while running, including any separate model server. "
      "A dash means it was not measured, for example for an online service.</dd>"
      "</dl>")

    # ---- results table
    w("<h2>All results</h2>")
    n_base = sum(1 for r in ranked if r in virtual)
    w("<p>Every classifier, ordered by accuracy on this sample."
      + (f" The {'grey row' if n_base == 1 else count_word(n_base) + ' grey rows'} at the bottom "
         f"{'is a simple rule, not a real model; it marks' if n_base == 1 else 'are simple rules, not real models; they mark'} the floor any "
         "classifier should clear." if n_base else "") + "</p>")
    w("<div class='table'><table><thead><tr><th class='l'>#</th><th class='l'>Classifier</th><th class='l'>Approach</th>"
      f"<th>Accuracy</th><th>95% CI</th><th>Average time per {esc(noun)} (batched)</th><th>Single request</th>"
      "<th>Memory</th></tr></thead><tbody>")
    rank = 0
    flagged = False
    for n_ in [r for r in ranked if r not in virtual] + [r for r in ranked if r in virtual]:
        m = metrics[n_]
        base = n_ in virtual
        if not base:
            rank += 1
        cls = "base" if base else ("top" if n_ in lead or n_ == best else "")
        mark = ""
        if not base and n_ in slow_conditions:
            mark, flagged = "<sup>*</sup>", True
        kind = "simple rule" if base else approach(n_) + (", remote" if remote(n_) else "")
        lat = latency(metas[n_]) if n_ in metas else None
        w(f"<tr class='{cls}'><td class='l'>{'' if base else rank}</td><td class='l name'>{link(n_)}</td>"
          f"<td class='l kind'>{esc(kind)}</td>"
          f"<td><div class='bar'><span class='track'><span class='fill' style='width:{max(m['accuracy'], 0) * 100:.1f}%'>"
          f"</span></span>{pct(m['accuracy'])}</div></td>"
          f"<td>{pct(m['ci'][0])}–{pct(m['ci'][1])}</td>"
          f"<td>{'–' if base else secs(time_of(n_)) + mark}</td><td>{'–' if base else secs(lat) + (mark if lat else '')}</td>"
          f"<td>{'–' if base else size(mem_of(n_))}</td></tr>")
    w("</tbody></table></div>")
    if flagged:
        w("<p class='small'><sup>*</sup> Timed on battery or in Low Power Mode, so probably slower than it would be "
          "plugged in.</p>")
    missing = [title(a) for a, e in apps.items() if not e.get("virtual") and a not in metrics]
    if missing:
        w(f"<p class='small'>Not included because they have no usable results for this test: {esc(join(missing))}.</p>")

    # ---- speed and accuracy
    timed = {n_: (acc(n_), time_of(n_)) for n_ in models if time_of(n_)}
    if len(timed) > 2:
        from stats import pareto_front
        front = pareto_front(timed)
        fast = min(timed, key=lambda n_: timed[n_][1])
        slow = max(timed, key=lambda n_: timed[n_][1])
        w("<h2>Speed versus accuracy</h2>")
        speed = []
        if timed[slow][1] > 2 * timed[fast][1]:
            speed.append(f"Average batched times range from {secs(timed[fast][1])} per {esc(noun)} ({esc(title(fast))}) to "
              f"{secs(timed[slow][1])} ({esc(title(slow))}), a factor of about "
              f"{rough(timed[slow][1] / timed[fast][1])}.")
        both = [n_ for n_ in timed if n_ in timed_lat]
        if both:
            slower = [n_ for n_ in both if timed_lat[n_] > 2 * timed[n_][1]]
            worst = max(both, key=lambda n_: timed_lat[n_] / timed[n_][1])
            speed.append(f"These are throughput figures, not response times. Sent one {esc(noun)} at a time, "
              f"{len(slower)} of the {len(both)} classifiers with both measurements took more than twice their batched "
              f"average; the largest gap is {esc(title(worst))}, at {secs(timed[worst][1])} batched against "
              f"{secs(timed_lat[worst])} for a single request.")
        if speed:
            w("<p>" + " ".join(speed) + "</p>")
        w(f"<p>{count_word(len(front)).capitalize()} classifier{'s are' if len(front) != 1 else ' is'} on the "
          "speed–accuracy frontier: no other classifier is both faster (batched) and more accurate. Every classifier "
          "not listed is beaten on both counts by at least one that is.</p>")
        w("<ul class='plain'>" + "".join(
            f"<li>{link(n_)}: {pct(acc(n_))} correct, {secs(time_of(n_))} per {esc(noun)} batched"
            + (f", {secs(timed_lat[n_])} single request" if n_ in timed_lat else "") + "</li>" for n_ in front) + "</ul>")
        w(f"<p class='small'>All local times come from one machine{' (' + esc(env['machine']) + ')' if env.get('machine') else ''}; "
          "the order of the faster classifiers can change on other hardware.</p>")

    # ---- hard labels
    real = [n_ for n_ in models]
    if real:
        w("<h2>Where classifiers go wrong</h2>")
        if per_label >= 10:
            recall = {lab: sum(metrics[n_]["per_class"][lab]["recall"] for n_ in real) / len(real) for lab in labels}
            order = sorted(labels, key=lambda lab: recall[lab])
            if len(labels) <= 10:
                w(f"<p>Averaged over all classifiers, this is how often the {esc(nouns)} under each label were "
                  "labeled correctly:</p>")
                w("<div class='table'><table><thead><tr><th class='l'>Label</th><th>Labeled correctly</th></tr></thead><tbody>"
                  + "".join(f"<tr><td class='l'>{esc(label_names[lab])}</td><td><div class='bar'><span class='track'>"
                            f"<span class='fill' style='width:{recall[lab] * 100:.1f}%'></span></span>{pct(recall[lab])}"
                            "</div></td></tr>" for lab in reversed(order)) + "</tbody></table></div>")
            else:
                w(f"<p>Averaged over all classifiers, the hardest labels were "
                  f"{join([f'<i>{esc(label_names[lab])}</i> ({pct(recall[lab])})' for lab in order[:5]])}, "
                  f"while {join([f'<i>{esc(label_names[lab])}</i>' for lab in order[-3:]])} were almost always "
                  "spotted.</p>")
        else:
            w(f"<p>With {len(labels)} labels and {n:,} {esc(nouns)}, each label appears only about "
              f"{per_label:.0f} time{'s' if round(per_label) != 1 else ''} in this test, too few to say which "
              "labels are hardest. The mix-ups below are more informative.</p>")
        total = None
        for n_ in real:
            cm = metrics[n_]["confusion"]
            total = cm if total is None else total.add(cm, fill_value=0)
        pairs = sorted(((int(total.at[t, p]), t, p) for t in labels for p in labels if t != p and total.at[t, p] > 0),
                       reverse=True)[:3]
        if pairs:
            w(f"<p>The most common mix-ups, counted across all {len(real)} classifiers:</p><ul class='plain'>"
              + "".join(f"<li><i>{esc(label_names[t])}</i> {esc(nouns)} labeled as <i>{esc(label_names[p])}</i>: "
                        f"{k:,} time{'s' if k != 1 else ''}, made by {sum(1 for n_ in real if metrics[n_]['confusion'].at[t, p] > 0)} of the "
                        f"{len(real)} classifiers</li>" for k, t, p in pairs) + "</ul>")
            k, t, p = pairs[0]
            back = int(total.at[p, t])
            if any((t2, p2) == (p, t) for _, t2, p2 in pairs):
                w(f"<p>The first mix-up also appears in reverse in this list, so <i>{esc(label_names[t])}</i> and "
                  f"<i>{esc(label_names[p])}</i> overlap rather than one label absorbing the other.</p>")
            elif back * 3 < k and k >= 10:
                w(f"<p>The reverse of the first mix-up is much rarer ({back:,} time{'s' if back != 1 else ''}), so when unsure between the two, "
                  f"classifiers lean towards <i>{esc(label_names[p])}</i>.</p>")

    # ---- trust
    held = {n_: metrics[n_]["held_out"][0.95] for n_ in models if (metrics[n_].get("held_out") or {}).get(0.95)}
    top_gate = sorted((n_ for n_ in held if held[n_]["met"] and held[n_]["coverage"] > 0),
                      key=lambda n_: -held[n_]["coverage"])[:3]
    if held:
        methods = {h.get("method") for h in held.values()}
        w("<h2>Knowing when to trust an answer</h2>")
        w(f"<p>A confidence score allows a split workflow: the classifier answers the {esc(nouns)} it is confident "
          "about and passes the rest to a person. For each classifier, the confidence cut-off for a 95% accuracy target "
          "was chosen on one set of " + esc(nouns) + " and then checked on different ones, so the figures below are "
          "not tuned to the items they are measured on.</p>")
        if top_gate:
            g = held[top_gate[0]]
            w(f"<p>{link(top_gate[0])} answered {pct(g['coverage'])} of the {esc(nouns)} on its own and was "
              f"{pct(g['accuracy'])} correct on them"
              + (", followed by " + join([f"{link(n_)} ({pct(held[n_]['coverage'])})" for n_ in top_gate[1:]])
                 if len(top_gate) > 1 else "") + ".")
            if best and top_gate[0] != best:
                w(f" It is not the top scorer on this sample ({link(best)} is): ranking its own answers by "
                  "confidence is a separate skill from getting them right.")
            w("</p>")
        else:
            w(f"<p>No classifier reached 95% accuracy on the {esc(nouns)} it accepted once the cut-off was checked on "
              "separate items.</p>")
        if "cross-fitted" in methods:
            w(f"<p class='small'>Cut-offs were chosen by cross-fitting: the sample was split in half, and each half was "
              f"scored with the cut-off chosen on the other. With about {n // 2:,} {esc(nouns)} per half these figures "
              "are rough; a separate calibration run gives firmer ones.</p>")
        else:
            w("<p class='small'>Cut-offs were chosen on a separate calibration sample (a second test-set sample with "
              "no items in common with this one).</p>")

    # ---- caution
    w("<h2>Before you draw conclusions</h2><ul class='plain'>"
      f"<li><b>The test is a sample.</b> With {n:,} {esc(nouns)}, an accuracy near 90% is only known to within "
      f"about ±{half:.0f} points. That separates good classifiers from poor ones, but not classifiers a few points "
      "apart.</li>"
      "<li><b>Scores describe this setup.</b> Some classifiers learned from labeled examples of this task and "
      "others from none; some ran locally and others on a remote service. A higher score means better "
      "under these conditions, not a better model in general.</li>"
      f"<li><b>Results belong to this task.</b> A classifier that does well on {esc(name)} may do worse on "
      "different text or different labels; compare with the other benchmark task.</li>"
      "<li><b>Speed depends on hardware and how the classifier is called.</b> Batched averages measure throughput, "
      "and single-request times measure response time. Both come from one machine.</li></ul>")

    # ---- full report
    w("<section class='full'><h2>The full raw report</h2>"
      "<p>This page leaves out most of the detail. The full report is generated automatically from the same run and "
      "contains every score, statistical test and chart, plus the settings needed to reproduce the results.</p>"
      f"<a class='btn' href='{FULL_HREF}'>Open the full raw generated report</a></section>")
    w(f"<p class='meta' style='margin-top:32px'>Model guides: <a href='{MODELS_HREF}/index.html'>all classifiers</a>. "
      f"Classifier names above link to their own pages.</p>")
    w("</main></body></html>")
    return "".join(out)

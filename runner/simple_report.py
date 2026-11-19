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

KIND_TEXT = {
    "zero-shot": "label names only",
    "supervised": "trained on this task",
    "in-domain trained": "saw this task before",
    "virtual": "simple rule",
}

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
                title, link, supervision, per_item, example=None) -> str:
    """The summary report as one HTML page (placeholders MODELS_HREF and FULL_HREF still in it).

    title(app) is the display name, link(app) the name linked to its model page, supervision(app) how it learned,
    per_item(meta) its seconds per item; example is the shared sample (its first row is shown).
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
      f"labeled correctly, how fast it worked, and how much memory it needed. This page tells the story of the "
      f"results in plain language.</p>")
    w(f"<p class='meta'>Results from {started:%-d %B %Y} · {esc(env.get('machine') or 'one machine')}"
      + (f", {env.get('memory_gb')} GB memory" if env.get("memory_gb") else "")
      + f" · <a href='{FULL_HREF}'>Full raw report</a></p>")

    # ---- at a glance
    fastest = min((n_ for n_ in models if time_of(n_)), key=time_of, default=None)
    majority = next((n_ for n_ in ranked if n_ in virtual and "majority" in n_.lower()), None)
    w("<div class='glance'>")
    if best:
        w(f"<div><div class='k'>Highest {esc(metric_word)}</div><div class='v'>{pct(acc(best))}</div>"
          f"<div class='s'>{esc(title(best))}</div></div>")
    if fastest:
        w(f"<div><div class='k'>Fastest</div><div class='v'>{secs(time_of(fastest))}</div>"
          f"<div class='s'>per {esc(noun)}, {esc(title(fastest))} ({pct(acc(fastest))} correct)</div></div>")
    if majority:
        w(f"<div><div class='k'>Blind guessing scores</div><div class='v'>{pct(acc(majority))}</div>"
          f"<div class='s'>always picking the most common label</div></div>")
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
    per_label = n / len(labels)
    w(f"<p>Every classifier was given the same {n:,} {esc(nouns)}, drawn from the test set with "
      + (f"an equal number from each of the {len(labels)} labels" if n % len(labels) == 0
         else f"the {len(labels)} labels spread as evenly as possible")
      + f". None of these {esc(nouns)} were used to train any of the models. Each classifier returned one label per "
      f"{esc(noun)} and, in most cases, a score saying how sure it was.</p>")
    kinds = {k: sum(1 for n_ in models if supervision(n_) == k) for k in ("zero-shot", "supervised", "in-domain trained")}
    w("<p>The classifiers fall into two broad families. Some <b>learned from labeled examples</b> of this very task "
      f"({kinds['supervised'] + kinds['in-domain trained']} of them): they were shown thousands of {esc(nouns)} with "
      "the right answer attached before the test. The others work <b>from the label names alone</b> "
      f"({kinds['zero-shot']} of them, often called zero-shot): they are told what the labels are, but never see an "
      "example. The second kind is harder to build well, but it can be pointed at a new task without collecting "
      "training data.</p>")

    # ---- the short answer
    w("<h2>The short answer</h2>")
    if best:
        if len(lead) > 1:
            lo = min(acc(n_) for n_ in lead)
            w(f"<p>There is no single winner. The top {count_word(len(lead))} classifiers scored between "
              f"{pct(lo)} and {pct(acc(best))}, and with {n:,} test {esc(nouns)} the gaps between them are too small "
              f"to call: on a different set of {esc(nouns)} their order could easily change. They are "
              f"{join([link(n_) for n_ in lead])}.</p>")
        else:
            w(f"<p>{link(best)} is the most accurate, getting {pct(acc(best))} of the {esc(nouns)} right "
              f"({metrics[best]['correct']:,} of {n:,}). Its lead is large enough that it is unlikely to be down to "
              f"the particular {esc(nouns)} we happened to test.</p>")
        if supervision(best) != "zero-shot":
            zs = next((n_ for n_ in models if supervision(n_) == "zero-shot"), None)
            if zs:
                gap = round(abs(acc(best) - acc(zs)) * 100)
                w(f"<p>The best classifier that worked from the label names alone is {link(zs)}, at {pct(acc(zs))}. "
                  + (f"That is only {gap} point{'s' if gap != 1 else ''} behind the top score, so on this task "
                     "training data buys surprisingly little.</p>" if gap <= 2 else
                     f"That is {gap} points behind the top score: roughly the price of skipping training data on "
                     "this task.</p>"))
        if majority:
            above = sum(1 for n_ in models if acc(n_) > acc(majority) + 0.10)
            beat = ("Every classifier beat that" if above == len(models)
                    else f"{above} of the {len(models)} classifiers beat that")
            w(f"<p>For a sense of scale, a rule that ignores the text and always picks the most common label gets "
              f"{pct(acc(majority))} right. {beat} by more than ten points.</p>")

    # ---- how to read
    w("<h2>How to read the numbers</h2>")
    w("<dl class='terms'>"
      f"<dt>Accuracy</dt><dd>The share of {esc(nouns)} the classifier labeled correctly. 90% means 9 out of 10 "
      "were right.</dd>"
      f"<dt>Likely range</dt><dd>A test on {n:,} {esc(nouns)} is a sample, not the final word. If the same "
      "classifier were tested on fresh items, its accuracy would usually land inside this range. When two "
      "classifiers' ranges overlap a lot, treat them as roughly equal.</dd>"
      f"<dt>Time per {esc(noun)}</dt><dd>How long the classifier took for each {esc(noun)} on our test machine, "
      "working at its normal pace. Smaller is faster.</dd>"
      "<dt>Memory</dt><dd>The most memory the classifier used while running, including any separate model server. "
      "A dash means it was not measured, for example for an online service.</dd>"
      "</dl>")

    # ---- results table
    w("<h2>All results</h2>")
    n_base = sum(1 for r in ranked if r in virtual)
    w("<p>Every classifier, from most to least accurate."
      + (f" The {'grey row' if n_base == 1 else count_word(n_base) + ' grey rows'} at the bottom "
         f"{'is a simple rule' if n_base == 1 else 'are simple rules'}, not real models; they mark the floor any "
         "classifier should clear." if n_base else "") + "</p>")
    w("<div class='table'><table><thead><tr><th class='l'>#</th><th class='l'>Classifier</th><th class='l'>Learned from</th>"
      f"<th>Accuracy</th><th>Likely range</th><th>Time per {esc(noun)}</th><th>Memory</th></tr></thead><tbody>")
    rank = 0
    for n_ in [r for r in ranked if r not in virtual] + [r for r in ranked if r in virtual]:
        m = metrics[n_]
        base = n_ in virtual
        if not base:
            rank += 1
        cls = "base" if base else ("top" if n_ in lead or n_ == best else "")
        w(f"<tr class='{cls}'><td class='l'>{'' if base else rank}</td><td class='l name'>{link(n_)}</td>"
          f"<td class='l kind'>{esc(KIND_TEXT.get(supervision(n_), supervision(n_)))}</td>"
          f"<td><div class='bar'><span class='track'><span class='fill' style='width:{max(m['accuracy'], 0) * 100:.1f}%'>"
          f"</span></span>{pct(m['accuracy'])}</div></td>"
          f"<td>{pct(m['ci'][0])}–{pct(m['ci'][1])}</td>"
          f"<td>{'–' if base else secs(time_of(n_))}</td><td>{'–' if base else size(mem_of(n_))}</td></tr>")
    w("</tbody></table></div>")
    missing = [title(a) for a, e in apps.items() if not e.get("virtual") and a not in metrics]
    if missing:
        w(f"<p class='small'>Not included because they have no usable results for this test: {esc(join(missing))}.</p>")

    # ---- speed and accuracy
    timed = {n_: (acc(n_), time_of(n_)) for n_ in models if time_of(n_)}
    if len(timed) > 2:
        from stats import pareto_front
        front = pareto_front(timed)
        times = sorted(t for _, t in timed.values())
        w("<h2>Speed versus accuracy</h2>")
        w(f"<p>Accuracy is only half the story. The fastest classifier here needs {secs(times[0])} per {esc(noun)}; "
          f"the slowest needs {secs(times[-1])}, roughly {rough(times[-1] / times[0])} times longer. As a rule, the "
          "larger and more accurate models are also the slower ones, so picking a classifier means deciding how much "
          "speed you are willing to trade for each extra point of accuracy.</p>")
        w(f"<p>The {count_word(len(front))} classifiers below are the best trades on offer: for each of them, no other "
          "classifier is both faster and more accurate. Any classifier not on this list is beaten on both counts by "
          "at least one that is.</p>")
        w("<ul class='plain'>" + "".join(
            f"<li>{link(n_)}: {pct(acc(n_))} correct, {secs(time_of(n_))} per {esc(noun)}</li>" for n_ in front) + "</ul>")
        w(f"<p class='small'>Times were measured on one machine and depend on hardware; treat them as a guide to the "
          "relative speed of the classifiers rather than as exact figures.</p>")

    # ---- hard labels
    real = [n_ for n_ in models]
    if real:
        w(f"<h2>Where classifiers go wrong</h2>")
        if per_label >= 10:
            recall = {lab: sum(metrics[n_]["per_class"][lab]["recall"] for n_ in real) / len(real) for lab in labels}
            order = sorted(labels, key=lambda lab: recall[lab])
            if len(labels) <= 10:
                w(f"<p>Some labels are harder to recognise than others. Averaged over all classifiers, this is how "
                  f"often the {esc(nouns)} under each label were labeled correctly:</p>")
                w("<div class='table'><table><thead><tr><th class='l'>Label</th><th>Labeled correctly</th></tr></thead><tbody>"
                  + "".join(f"<tr><td class='l'>{esc(label_names[lab])}</td><td><div class='bar'><span class='track'>"
                            f"<span class='fill' style='width:{recall[lab] * 100:.1f}%'></span></span>{pct(recall[lab])}"
                            "</div></td></tr>" for lab in reversed(order)) + "</tbody></table></div>")
            else:
                w(f"<p>Some labels are much harder than others. Averaged over all classifiers, the hardest to "
                  f"recognise were {join([f'<i>{esc(label_names[lab])}</i> ({pct(recall[lab])})' for lab in order[:5]])}, "
                  f"while {join([f'<i>{esc(label_names[lab])}</i>' for lab in order[-3:]])} were almost always "
                  "spotted.</p>")
        else:
            w(f"<p>With {len(labels)} labels and {n:,} {esc(nouns)}, each label appears only about "
              f"{per_label:.0f} time{'s' if round(per_label) != 1 else ''} in this test, too few to say reliably which "
              "labels are hardest. The mix-ups below are more telling.</p>")
        # Most common mix-ups, summed over every classifier's confusion table.
        total = None
        for n_ in real:
            cm = metrics[n_]["confusion"]
            total = cm if total is None else total.add(cm, fill_value=0)
        pairs = sorted(((int(total.at[t, p]), t, p) for t in labels for p in labels if t != p and total.at[t, p] > 0),
                       reverse=True)[:3]
        if pairs:
            w(f"<p>The most common mix-ups, counted across all {len(real)} classifiers:</p><ul class='plain'>"
              + "".join(f"<li><i>{esc(label_names[t])}</i> {esc(nouns)} labeled as <i>{esc(label_names[p])}</i>: "
                        f"{k:,} times, about {k / len(real):.1f} per classifier</li>" for k, t, p in pairs) + "</ul>")
            w("<p>Mistakes like these tend to happen where the labels themselves overlap, and a person might "
              "hesitate too.</p>")

    # ---- trust
    cert = {n_: metrics[n_]["at_target"][0.95]["certified"] for n_ in models
            if (metrics[n_].get("at_target") or {}).get(0.95)}
    top_gate = sorted((n_ for n_ in cert if cert[n_]["coverage"] > 0), key=lambda n_: -cert[n_]["coverage"])[:3]
    if top_gate:
        w("<h2>Knowing when to trust an answer</h2>")
        w(f"<p>Most classifiers also say how sure they are of each answer. That makes a useful working pattern "
          f"possible: let the classifier handle the {esc(nouns)} it is confident about, and pass the rest to a person. "
          "The question is how much work it can take on before mistakes creep in.</p>")
        g = cert[top_gate[0]]
        w(f"<p>Set the bar at 95% correct. {link(top_gate[0])} could handle {pct(g['coverage'])} of the {esc(nouns)} on "
          f"its own and still meet that bar"
          + (", followed by " + join([f"{link(n_)} ({pct(cert[n_]['coverage'])})" for n_ in top_gate[1:]])
             if len(top_gate) > 1 else "")
          + ". Note that a classifier's raw accuracy and its ability to judge its own answers are different skills: "
          "the most accurate classifier is not always the best at knowing when it is right.</p>")
        w("<p class='small'>The confidence cut-offs behind these figures were chosen on the same test "
          f"{esc(nouns)}, so check them on new data before relying on them.</p>")

    # ---- caution
    w("<h2>Before you draw conclusions</h2><ul class='plain'>"
      f"<li><b>The test is a sample.</b> {n:,} {esc(nouns)} is enough to separate good classifiers from poor ones, "
      "but not to rank classifiers that are within a few points of each other.</li>"
      f"<li><b>Results belong to this task.</b> A classifier that does well on {esc(name)} may do worse on "
      "different text or different labels; the other benchmark task can tell quite a different story.</li>"
      "<li><b>Speed depends on hardware.</b> All timings come from one machine; on other hardware the order of the "
      "faster classifiers may change.</li></ul>")

    # ---- full report
    w("<section class='full'><h2>The full raw report</h2>"
      "<p>This page leaves out most of the detail. The full report is generated automatically from the same run and "
      "contains every score, statistical test and chart, plus the settings needed to reproduce the results.</p>"
      f"<a class='btn' href='{FULL_HREF}'>Open the full raw generated report</a></section>")
    w(f"<p class='meta' style='margin-top:32px'>Model guides: <a href='{MODELS_HREF}/index.html'>all classifiers</a>. "
      f"Classifier names above link to their own pages.</p>")
    w("</main></body></html>")
    return "".join(out)

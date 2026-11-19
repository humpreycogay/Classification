"""Registry of the classifier apps the runner knows about, in run order.

Each app is a folder in classifiers/ with its own .venv, classifier.py and evaluate.py.
`needs` names the pre-flight check the runner performs before starting the app.
Several registry entries may share one folder (`folder`), each run with its own extra
`args` and writing to its own results folder (`results`, relative to the app folder).

`confidence_rule` tells the result validator how the app defines `confidence`: "max_p" means it
must equal the largest p_* value; "own" means the app reports its own score (only checked to be in
[0, 1]). `pred` must always be a label with the largest p_*. `cost` = "paid" marks apps that bill
per request; the runner skips them above PAID_MAX_N items unless --allow-paid is given.
`model_arg` makes the runner pass the resolved `ollama_model` tag as `--model` (and its server as `--host`),
for apps that serve several models from one folder (llm/: each model is its own entry, with its own results
folder, results/{model}/ = the resolved tag). `ollama_host` names an env var for the app's own Ollama server (llm/: LLM_HOST, e.g. a GPU box on
the network); unset, it is OLLAMA_HOST. The shortlist's embedding model (extra_ollama) stays on OLLAMA_HOST.
`max_labels` and `remote` let the runner skip an app, with the reason, for a task it cannot or may
not run (too many labels; private data that must not leave the machine).
`trained_on_tasks` names tasks whose data the model saw in training (e.g. a model trained on jev-bench, which
includes Banking77): the report labels it "in-domain trained" there instead of zero-shot, and the runner skips it
on such a task when the app's setup-time overlap check (results/overlap.json, shared/overlap.py) found test texts
of ours in its training data beyond the task's own train split, or has not run. Test texts that are also in the
train split (the dataset's own duplicates) are seen by every supervised baseline too, so they do not count.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLASSIFIERS_DIR = ROOT / "classifiers"  # one folder per classifier


def load_root_env() -> dict[str, str]:
    """Key/value pairs from the root .env (values are never logged)."""
    env = {}
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip().strip('"').strip("'")
    return env


ROOT_ENV = load_root_env()


def setting(key: str, default: str) -> str:
    """A setting: the shell environment, else the root .env, else the default."""
    return os.environ.get(key) or ROOT_ENV.get(key) or default


def model_slug(tag: str) -> str:
    """A model tag as a folder name, the same as llm/evaluate.py's default --results (gemma4:e4b -> gemma4_e4b)."""
    return re.sub(r"[^A-Za-z0-9.-]+", "_", tag)


# How each app learned the task, for grouping results (App.approach_on). Remote execution is a separate flag (App.remote).
APPROACHES = {
    "zero-shot": "No labeled examples of the task; works from the label names or descriptions",
    "in-domain checkpoint": "A released model whose own training already included this task's data",
    "classical supervised": "TF-IDF or embedding features with a linear or centroid classifier, fitted on the train split",
    "fine-tuned encoder": "A pre-trained encoder fine-tuned on the train split",
    "from scratch": "A neural network trained from random weights on the train split",
}


@dataclass(frozen=True)
class App:
    name: str
    title: str
    runs_on: str
    confidence: str
    needs: str | None = None  # jev_key | ollama | openjev_server | winnow_server | djev_server
    ollama_model: tuple[str, str] | None = None  # (env var, default tag)
    supervision: str = "zero-shot"  # zero-shot | supervised (trained on the task's train split)
    folder: str | None = None  # app folder, if not the same as name
    args: tuple[str, ...] = ()  # extra evaluate.py arguments
    results: str = "results"  # results folder, relative to the app folder
    confidence_rule: str = "max_p"  # max_p | own
    cost: str = "free"  # free | paid
    max_labels: int | None = None  # largest label set the app supports (None = no fixed limit)
    remote: bool = False  # sends text to a remote service (skipped for tasks that do not allow it)
    prob_sum_tol: float = 1e-3  # allowed |sum(p_*) - 1| per row; larger for apps that round their probabilities
    shortlist: int | None = None  # two-stage variant: classifies among a zero-shot shortlist of this many labels
    extra_ollama: tuple[str, str] | None = None  # a second Ollama model the app needs (env var, default tag)
    model_arg: bool = False  # pass the resolved ollama_model tag (and its server) to evaluate.py as --model/--host
    ollama_host: str | None = None  # env var naming this app's own Ollama server (default: OLLAMA_HOST)
    trained_on_tasks: tuple[str, ...] = ()  # tasks whose data the model saw in training (see the module docstring)

    @property
    def path(self) -> Path:
        return CLASSIFIERS_DIR / (self.folder or self.name)

    @property
    def model_tag(self) -> str:
        """The resolved Ollama model tag (its env var, else the default)."""
        return setting(*self.ollama_model)

    @property
    def display_title(self) -> str:
        return self.title.format(model=self.model_tag) if self.ollama_model else self.title

    def supervision_on(self, task: str) -> str:
        """The app's type on a task: "in-domain trained" where the model saw the task's data in training."""
        return "in-domain trained" if task in self.trained_on_tasks else self.supervision

    def approach_on(self, task: str) -> str:
        """How the app came to know the task, finer than supervision_on (one of APPROACHES)."""
        kind = self.supervision_on(task)
        if kind == "in-domain trained":
            return "in-domain checkpoint"
        if kind == "zero-shot":
            return "zero-shot"
        folder = self.folder or self.name
        if folder.startswith("scratch_"):
            return "from scratch"
        if folder.startswith("finetune_"):
            return "fine-tuned encoder"
        return "classical supervised"

    def training_overlap(self, task: str) -> str | None:
        """Why the app may not run a task it was trained on (test texts in its training data, or the overlap
        check has not run), else None. Reads results/overlap.json, written by the app's setup.sh."""
        if task not in self.trained_on_tasks:
            return None
        path = self.path / "results" / "overlap.json"
        try:
            info = json.loads(path.read_text()).get(task)
        except (OSError, ValueError):
            info = None
        if not isinstance(info, dict) or "overlap" not in info:
            return (f"the model was trained on {task} data and its overlap check has not run "
                    f"(run {self.path.name}/setup.sh)")
        if info["overlap"]:
            return (f"the model's training data contains {info['overlap']} of the {task} test texts that are not in "
                    f"its train split ({info.get('detail', 'exact text match')}); a test on this task would not be fair")
        return None

    @property
    def results_dir(self) -> Path:
        # "{model}" in results: one folder per model tag, so changing the model in .env never mixes results.
        return self.path / (self.results.format(model=model_slug(self.model_tag)) if self.ollama_model else self.results)


# Ollama models of the llm/ app: (env var, default tag). The env var overrides the tag for that entry.
LLM_QWEN = ("LLMQWEN_MODEL", "qwen3.8:27b-q4_K_M")

APPS: list[App] = [
    App("jev", "Jev (TypeSafe API)", "Remote API", "Jev's own score", needs="jev_key",
        confidence_rule="own", cost="paid", remote=True,
        prob_sum_tol=0.02),  # the API rounds probabilities to 2 decimals (rows can sum to 0.99)
    App("laya", "Laya", "Local, PyTorch MPS", "max p (Laya answer_confidence, uncalibrated)"),
    # One Laya checkpoint pinned instead of routing by language (only that one is loaded). Batch sizes: the fastest
    # in a --sweep (2026-09-30, AG News) that agreed with batch 1.
    App("laya_typed", "Laya typed-decisions (ModernBERT-large 421M)", "Local, PyTorch MPS",
        "max p (Laya answer_confidence, uncalibrated)", folder="laya", args=("--model", "typed-decisions", "--batch-size", "16"),
        results="results/typed-decisions"),
    App("laya_multilingual", "Laya multilingual (mmBERT-base 322M)", "Local, PyTorch MPS",
        "max p (Laya answer_confidence, uncalibrated)", folder="laya", args=("--model", "multilingual", "--batch-size", "8"),
        results="results/multilingual"),
    App("nli", "NLI zero-shot (DeBERTa-v3-large)", "Local, PyTorch MPS", "max p"),
    # The same NLI app with ModernBERT-large (float32; batch 8: fastest in a --sweep that agreed with batch 1).
    App("nli_modernbert", "NLI zero-shot (ModernBERT-large)", "Local, PyTorch MPS", "max p",
        folder="nli", args=("--model", "MoritzLaurer/ModernBERT-large-zeroshot-v2.0", "--batch-size", "8"),
        results="results/modernbert-large"),
    # ModernBERT-base + LoRA trained on jev-bench, which includes Banking77: its setup.sh checks our test texts
    # against that training data (results/overlap.json) and the runner skips banking77 when they overlap.
    App("modernbert_jev", "ModernBERT-JEV (ModernBERT-base + LoRA)", "Local, PyTorch MPS",
        "max p (shipped temperature)", trained_on_tasks=("banking77",)),
    App("gliclass", "GLiClass base v3.0", "Local, PyTorch MPS", "max p"),
    App("verdict", "Verdict (ModernBERT 151M)", "Local, PyTorch MPS", "max p (shipped temperature)",
        max_labels=24),  # the checkpoint scores 2-24 options
    App("embedsim", "Embedding similarity (bge-m3)", "Local, Ollama", "max p (fitted temperature)",
        needs="ollama", ollama_model=("EMBEDSIM_MODEL", "bge-m3:567m")),
    App("SemIf", "SemIf (Qwen3.5-4B)", "Local, MLX", "max p", max_labels=16),  # the SemIf library takes 2-16 options
    # Open Jev-style decision model (Bespoke Labs): one enum field, logits of one-token answer codes, 1-255 choices.
    App("nimble", "Nimble (Bespoke-Nimble-9B)", "Local, MLX", "max p (release temperature)", max_labels=255),
    # The chat-LLM app (llm/), one model: LLMQWEN_MODEL. Another model = another entry; each keeps its own results folder.
    App("llmqwen", "LLM Qwen ({model})", "Ollama (LLM_HOST)", "max p",
        needs="ollama", ollama_model=LLM_QWEN, model_arg=True, ollama_host="LLM_HOST", max_labels=16,  # lettered options A-P
        folder="llm", results="results/{model}"),
    # Decider-4B v3: trained on the train splits of AG News and Banking77 (its mixture takes every registered
    # task's train split), so it is in-domain trained on both; setup.sh's overlap check (shared/overlap.py).
    App("decider", "Decider-4B v3 (Qwen3.5-4B)", "Local, PyTorch MPS + MLX kernels", "max p (release temperature)",
        max_labels=255, trained_on_tasks=("ag_news", "banking77")),
    # Strands Decider (Strands Labs, AWS): another model than `decider`, with its own code and weights. Trained on the
    # train splits of AG News and Banking77 (its data/sources.md), so in-domain trained on both; overlap check as decider.
    App("strands_decider", "Strands Decider 2B hobson-v19 (Qwen3.5-2B)", "Local, MLX", "max p (release temperature)",
        max_labels=255, trained_on_tasks=("ag_news", "banking77"),
        prob_sum_tol=5e-3),  # the engine rounds probabilities to 4 decimals (77 options: up to 0.004 off)
    # LiquidAI d1-3B: native System One choice probabilities, packed without padding; no generated tokens.
    App("d1", "LiquidAI d1-3B (LFM2.5-VL-3B)", "Local, PyTorch MPS", "max p (native)"),
    App("d1_omni", "LiquidAI d1-omni-600M", "Local, PyTorch MPS", "max p (native)"),
    # Naluri's release uses a 192-token question/options budget. More than about 20 options destructively truncates
    # each option, so Banking77 is intentionally refused rather than producing a misleading direct benchmark.
    App("naluri_279m", "Naluri-279M encoder (XLM-R)", "Local, PyTorch MPS", "max p (release temperature)",
        folder="naluri", args=("--model", "279m"), results="results/279m", max_labels=20),
    App("naluri_308m", "Naluri-308M encoder (mmBERT)", "Local, PyTorch MPS", "max p (release temperature)",
        folder="naluri", args=("--model", "308m"), results="results/308m", max_labels=20),
    # Decision 3.0's portable runtime is CUDA/CPU-first. MPS is opt-in and remains experimental.
    App("d3_edge", "Decision 3.0 edge (0.59B)", "Local, PyTorch CPU (MPS experimental)", "max p (native)",
        folder="d3", args=("--model", "edge"), results="results/edge", max_labels=255),
    App("d3_lite", "Decision 3.0 lite", "Local, PyTorch CPU (MPS experimental)", "max p (native)",
        folder="d3", args=("--model", "lite"), results="results/lite", max_labels=255),
    App("d3_nano", "Decision 3.0 nano", "Local, PyTorch CPU (MPS experimental)", "max p (native)",
        folder="d3", args=("--model", "nano"), results="results/nano", max_labels=255),
    App("kev", "Kev-0.8B (Qwen3.5 + pointer head)", "Local, MLX", "max p (release temperature)",
        max_labels=255, trained_on_tasks=("ag_news", "banking77")),
    App("dexy", "Dexy-Decision-9B (experimental)", "Local CUDA; MPS unverified", "max p (release temperature)",
        max_labels=256),
    # Plumb-4B: jevk5's prompt and temperature, forward pass on MLX; 2-16 options (letters A-P).
    App("plumb", "Plumb-4B (Qwen3.5-4B)", "Local, MLX", "max p (release temperature)", max_labels=16),
    # Winnow-12B (Gemma 4 12B fine-tune) behind its llama.cpp server, which the runner starts and stops.
    App("winnow", "Winnow-12B (Gemma 4 12B, Q8_0)", "Local server, llama.cpp Metal", "max p",
        needs="winnow_server", max_labels=64),  # the server takes 2-64 options per question
    # CLM-v0.1-8B: contrastive heads on a local Qwen3-8B encoder (trained on agent actions, not topics).
    App("clm", "CLM-v0.1-8B (Qwen3-8B + contrastive heads)", "Local, PyTorch MPS", "max p (fitted temperature)"),
    App("OpenJev", "OpenJev (DiffusionGemma 26B-A4B)", "Local server, MLX",
        "server's 1 - normalized entropy", needs="openjev_server", confidence_rule="own"),
    App("djev", "djev (DiffusionGemma 26B-A4B)", "Remote GPU server (B200)", "server's own score",
        needs="djev_server", confidence_rule="own", remote=True),
    # Supervised baselines: trained on the task's train split, never on test data.
    App("tfidf_logreg", "TF-IDF + LogReg (supervised)", "Local, CPU (scikit-learn)", "max p (native)",
        supervision="supervised"),
    # Same train sample as the embedding baselines below (the task's supervised_sample_per_label per label;
    # 4,000 articles for AG News), so the two are compared on equal data.
    App("tfidf_logreg_4k", "TF-IDF + LogReg, embedding baselines' train sample (supervised)", "Local, CPU (scikit-learn)",
        "max p (native)", supervision="supervised", folder="tfidf_logreg", args=("--train-per-class", "task"),
        results="results/matched_train"),
    App("tfidf_svm", "TF-IDF + Linear SVM, calibrated (supervised)", "Local, CPU (scikit-learn)",
        "max p (sigmoid-calibrated on train, 5-fold CV)", supervision="supervised"),
    App("tfidf_svm_raw", "TF-IDF + Linear SVM, uncalibrated (supervised)", "Local, CPU (scikit-learn)",
        "max softmax(SVM margins), uncalibrated", supervision="supervised",
        folder="tfidf_svm", args=("--calibration", "none"), results="results/uncalibrated"),
    App("tfidf_cnb", "TF-IDF + Complement NB (supervised)", "Local, CPU (scikit-learn)", "max p (native, uncalibrated NB)",
        supervision="supervised"),
    App("embedsup_centroid", "bge-m3 + class centroids (supervised)", "Local, Ollama + CPU",
        "max p (temperature fitted on held-out train)", needs="ollama", ollama_model=("EMBEDSUP_MODEL", "bge-m3:567m"),
        supervision="supervised", folder="embedsup", args=("--method", "centroid"), results="results/centroid"),
    App("embedsup_logreg", "bge-m3 + LogReg (supervised)", "Local, Ollama + CPU", "max p (native)",
        needs="ollama", ollama_model=("EMBEDSUP_MODEL", "bge-m3:567m"),
        supervision="supervised", folder="embedsup", args=("--method", "logreg"), results="results/logreg"),
    # Neural networks trained from scratch on the task's train split (shared/textnn.py).
    App("scratch_fasttext", "fastText-style network, trained from scratch (supervised)", "Local, PyTorch CPU",
        "max p (temperature fitted on held-out train)", supervision="supervised"),
    App("scratch_cnn", "TextCNN, trained from scratch (supervised)", "Local, PyTorch MPS",
        "max p (temperature fitted on held-out train)", supervision="supervised"),
    App("scratch_bilstm", "BiLSTM, trained from scratch (supervised)", "Local, PyTorch CPU",
        "max p (temperature fitted on held-out train)", supervision="supervised"),
    # Pre-trained encoders fine-tuned on the task's train split (shared/finetune.py).
    App("finetune_modernbert", "ModernBERT-base, fine-tuned (supervised)", "Local, PyTorch MPS",
        "max p (temperature fitted on held-out train)", supervision="supervised"),
    App("finetune_deberta", "DeBERTa-v3-base, fine-tuned (supervised)", "Local, PyTorch MPS",
        "max p (temperature fitted on held-out train)", supervision="supervised"),
]

# Two-stage variants for tasks with more labels than the app takes (e.g. banking77's 77): a zero-shot
# shortlist (bge-m3 similarity to the label texts, as embedsim) keeps the K most likely labels per text,
# and the app chooses among those. Still zero-shot; accuracy is capped by the shortlist's recall.
SHORTLIST_MODEL = ("SHORTLIST_MODEL", "bge-m3:567m")
APPS += [
    App("llmqwen_shortlist", "LLM Qwen ({model}) + shortlist of 16", "Ollama (LLM_HOST + OLLAMA_HOST)",
        "max p over the shortlist", needs="ollama", ollama_model=LLM_QWEN, model_arg=True, ollama_host="LLM_HOST",
        extra_ollama=SHORTLIST_MODEL, folder="llm", args=("--shortlist", "16"), results="results/{model}/shortlist16",
        shortlist=16),
    App("SemIf_shortlist", "SemIf (Qwen3.5-4B) + shortlist of 16", "Local, MLX + Ollama", "max p over the shortlist",
        needs="ollama", ollama_model=SHORTLIST_MODEL, folder="SemIf", args=("--shortlist", "16"),
        results="results/shortlist16", shortlist=16),
    App("verdict_shortlist", "Verdict (ModernBERT 151M) + shortlist of 24", "Local, PyTorch MPS + Ollama",
        "max p over the shortlist", needs="ollama", ollama_model=SHORTLIST_MODEL, folder="verdict",
        args=("--shortlist", "24"), results="results/shortlist24", shortlist=24),
    App("plumb_shortlist", "Plumb-4B (Qwen3.5-4B) + shortlist of 16", "Local, MLX + Ollama", "max p over the shortlist",
        needs="ollama", ollama_model=SHORTLIST_MODEL, folder="plumb", args=("--shortlist", "16"),
        results="results/shortlist16", shortlist=16),
    # Winnow's server is started by the runner (winnow_server); the shortlist's bge-m3 runs on OLLAMA_HOST.
    App("winnow_shortlist", "Winnow-12B (Gemma 4 12B, Q8_0) + shortlist of 64", "Local server, llama.cpp Metal + Ollama",
        "max p over the shortlist", needs="winnow_server", extra_ollama=SHORTLIST_MODEL, folder="winnow",
        args=("--shortlist", "64"), results="results/shortlist64", shortlist=64),
]

APPS_BY_NAME = {a.name: a for a in APPS}

# What an app's `confidence` (and its p_* columns) means, derived from its confidence rule. Shared by the score
# reports and the model pages so both use the same terms: (label, one-sentence definition).
CONFIDENCE_KINDS = {
    "native": ("Native probability",
               "The model's own predicted probability, used as is: no temperature or calibrator was fitted for "
               "this benchmark."),
    "task": ("Task-calibrated",
             "A temperature or calibrator was fitted on labeled rows of this task's train split, so the confidence "
             "and calibration metrics are task-calibrated."),
    "release": ("Release temperature",
                "Scores divided by a temperature shipped with the model release, then normalized; nothing was fitted "
                "on this task."),
    "normalized": ("Normalized score",
                   "Softmax or normalized scores that sum to 1 but have no calibration step; useful for ranking labels, "
                   "not as calibrated probabilities."),
    "separate": ("Separate score",
                 "A confidence score defined by the service; it is not the selected label's probability."),
}


def confidence_kind(app: App) -> str:
    """Key into CONFIDENCE_KINDS for an app. Shortlist variants take the kind of the app they wrap."""
    if app.shortlist:
        base = next((a for a in APPS if a.path == app.path and not a.shortlist and not a.args), None)
        return confidence_kind(base) if base else "normalized"
    rule = app.confidence.lower()
    if app.confidence_rule == "own":
        return "separate"
    if "uncalibrated" in rule:
        return "normalized"
    if "fitted" in rule or "calibrated on train" in rule:
        return "task"
    if "release temperature" in rule or "shipped temperature" in rule:
        return "release"
    if "native" in rule:
        return "native"
    return "normalized"

# The model files each app needs on disk, checked by run_all.preflight so `./run list` does not report an app as
# ready when its weights were deleted (e.g. a cleaned Hugging Face cache). Keyed by app name, else by folder.
# "hf:<repo>" = a repo in the local HF cache with at least one weight file and no broken file links;
# "hf-setting:<VAR>:<default>" = the same, for a repo named by a setting; "path:<glob>" = a file under the app folder
# (symlinks must resolve). Apps served by a running server are not checked (the server already holds the model).
WEIGHTS: dict[str, tuple[str, ...]] = {
    "laya": ("hf:convaiinnovations/laya",),
    "nli": ("hf:MoritzLaurer/deberta-v3-large-zeroshot-v2.0",),
    "nli_modernbert": ("hf:MoritzLaurer/ModernBERT-large-zeroshot-v2.0",),
    "modernbert_jev": ("hf:ali-rehman-ML/modern-bert-jev",),
    "gliclass": ("hf:knowledgator/gliclass-base-v3.0",),
    "verdict": ("hf:heman10x/rlcd-modernbert-151m",),
    "SemIf": ("hf:Qwen/Qwen3.5-4B",),
    "nimble": ("path:models/*/READY.json",),
    "decider": ("hf:Mapika/decider-4b",),
    "strands_decider": ("hf:StrandsAgents/strands-decider-2B-hobson-v19", "hf:Qwen/Qwen3.5-2B-Base"),
    "d1": ("hf:LiquidAI/d1-3B",),
    "d1_omni": ("hf:LiquidAI/d1-omni-600M",),
    "naluri_279m": ("hf:jerichosiahaya/Naluri-279M-encoder",),
    "naluri_308m": ("hf:jerichosiahaya/Naluri-308M-encoder",),
    "d3_edge": ("hf:vllm-sr/d3-edge",),
    "d3_lite": ("hf:vllm-sr/d3-lite",),
    "d3_nano": ("hf:vllm-sr/d3-nano",),
    "kev": ("hf:jaredpalmer/kev-0.8b", "hf:Qwen/Qwen3.5-0.8B-Base"),
    "dexy": ("hf:Patrick-18/Dexy-Decision-9B",),
    "plumb": ("hf:crh225/plumb-4b",),
    "winnow": ("path:server/models/gguf/Winnow-12B-Q8_0.gguf",),
    "clm": ("hf:Qwen/Qwen3-8B", "hf:Contrastive-LM/CLM-v0.1-8B"),
    "OpenJev": ("hf-setting:OPENJEV_MLX_MODEL:mlx-community/diffusiongemma-26B-A4B-it-4bit",),
}
WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin", ".pt", ".pth", ".onnx", ".npz")


def weights_of(app: App) -> tuple[str, ...]:
    """The model files an app needs (see WEIGHTS)."""
    return WEIGHTS.get(app.name) or WEIGHTS.get(app.folder or app.name, ())


def hf_cache_dir() -> Path:
    """The Hugging Face hub cache, as huggingface_hub resolves it (HF_HUB_CACHE, else HF_HOME/hub)."""
    if cache := setting("HF_HUB_CACHE", ""):
        return Path(cache).expanduser()
    return Path(setting("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"


def hf_repo_problem(repo: str, cache: Path | None = None) -> str | None:
    """None if some cached snapshot of `repo` has a weight file and no broken links, else what is wrong."""
    snapshots = (cache or hf_cache_dir()) / f"models--{repo.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir() or not any(snapshots.iterdir()):
        return f"{repo} is not in the Hugging Face cache"
    problems = []
    for snapshot in sorted(snapshots.iterdir()):
        files = [p for p in snapshot.rglob("*") if p.is_symlink() or p.is_file()]
        broken = [p for p in files if not p.exists()]
        if not broken and any(p.name.endswith(WEIGHT_SUFFIXES) for p in files):
            return None
        problems.append(f"{len(broken)} missing file(s)" if broken else "no weight file")
    return f"{repo} in the Hugging Face cache is incomplete ({problems[-1]})"


def missing_weights(app: App, cache: Path | None = None) -> str | None:
    """None if every model file the app needs is on disk, else what is missing (see WEIGHTS)."""
    for spec in weights_of(app):
        kind, _, value = spec.partition(":")
        if kind == "path":
            if not any(p.exists() for p in app.path.glob(value)):  # exists() is False for a broken symlink
                return f"model file missing: {app.path.name}/{value}"
            continue
        if kind == "hf-setting":
            var, _, default = value.partition(":")
            value = setting(var, default)
        if problem := hf_repo_problem(value, cache):
            return problem
    return None

# Largest sample a paid app runs without --allow-paid (the "common" tier).
PAID_MAX_N = 1000

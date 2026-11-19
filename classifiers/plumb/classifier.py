"""Plumb Classifier: local text classification with Plumb-4B (crh225/plumb-4b) on Apple Silicon MLX.

Plumb-4B is a decision model fine-tuned from JevK5 v0.2 (Qwen3.5-4B + a merged LoRA). Its runtime, jevk5
(github.com/allebee/jevk5, pinned at v0.2.0 as the Plumb card specifies), builds a SemIf-style prompt: a fixed system
instruction, the decision as JSON (evidence, criterion, lettered options), the chat template with thinking off;
the answer letters' next-token logits are divided by T (jevk5_config.json: 2.07) and softmaxed. Nothing is generated.

jevk5's own forward pass targets CUDA (CUDA graphs), so here the prompt, the option texts, the answer-letter tokens
and the temperature come from jevk5, and the forward pass runs on MLX. setup.sh checks the result against jevk5's
PyTorch path on CPU (parity gate). Each option is "<name>: <description>" (jevk5's own "id: description" mapping,
with the label's name as id). 2-16 options (letters A-P); Banking77 runs through the shortlist variant.

Speed: when the prompt start every text shares (system prompt, chat header) is a large enough share of the prompt,
it is computed once and reused, but only if it reproduces the plain logits on probe texts (shared/mlx_prefix.py).

Usage:
    python classifier.py "Stocks rallied after the Fed held rates steady."

Environment (optional, read from the root .env):
    PLUMB_BITS   4 | 8   in-memory quantization (default: unquantized BF16, ~8.4 GB)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

# Xet transfers stall on some networks; plain HTTPS downloads are reliable.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from jevk5.runtime import LETTERS, JevK5, _load_temperature, decision_options  # noqa: E402
from mlx_lm import load  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import mlx_prefix  # noqa: E402  (shared/: the shared-prompt-start cache)
from task import current_task  # noqa: E402

# .env lives in the repo root (classifiers/<app>/ -> two levels up).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "crh225/plumb-4b"
REVISION = "55de037801a8a9b9de3db5c0e16cef86210c2186"  # Plumb-4B v5 weights (later commits change only the card)
MAX_OPTIONS = len(LETTERS)  # 16
# Reuse the shared prompt start only when it is at least this share of a prompt (as Nimble).
MIN_PREFIX_SHARE = 0.10
PREFIX_PROBES = ("Alpha probe text about markets.", "Zebra probe sentence on sports.", "123 probe of a third kind.")

# Labels, descriptions and instruction come from the task manifest (data/tasks/<task>.json, see task.py),
# chosen with --task on evaluate.py (default: ag_news).
TASK = current_task()
_LABEL_SPECS = TASK.labels
CLASS_TO_LABEL = {spec["raw"]: spec["id"] for spec in _LABEL_SPECS}
TASK_LABELS: dict[str, str] = {spec["id"]: spec["description"] for spec in _LABEL_SPECS}
LABEL_NAMES = {spec["id"]: spec["name"] for spec in _LABEL_SPECS}


def model_dir() -> str:
    return snapshot_download(MODEL, revision=REVISION)


def question(instruction: str, labels: dict[str, str]) -> dict:
    """The jevk5 choice question for these labels, keyed by the label's name."""
    return {"type": "choice", "instructions": instruction,
            "criteria": {LABEL_NAMES.get(l, l): d for l, d in labels.items()}}


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class PlumbClassifier:
    """Single-label text classifier: one Plumb choice decision per text, scored on MLX."""

    def __init__(self, labels: dict[str, str], instruction: str, bits: int | None = None,
                 prefix: bool = True) -> None:
        if not 2 <= len(labels) <= MAX_OPTIONS:
            raise ValueError(f"Plumb takes 2-{MAX_OPTIONS} options (use --shortlist)")
        self.labels = labels
        self.instruction = instruction
        self.path = model_dir()
        env_bits = os.getenv("PLUMB_BITS")
        self.bits = bits or (int(env_bits) if env_bits else None)
        self.temperature = _load_temperature(self.path)  # jevk5_config.json
        # jevk5's tokenizer and prompt builder (JevK5.encode needs only .tok).
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self._jevk5 = SimpleNamespace(tok=self.tok)
        self.slots = [self.tok.encode(letter, add_special_tokens=False)[0] for letter in LETTERS]
        mx.set_default_device(mx.gpu)
        # The release's config is text-only (model_type qwen3_5_text, Qwen3_5ForCausalLM); mlx-lm's qwen3_5 model
        # takes such a flat config as its text config, and maps the model.language_model.* weights itself.
        self.model, _ = load(self.path, model_config={"model_type": "qwen3_5"})
        language_model = getattr(self.model, "language_model", self.model)
        if self.bits:  # the body only: the answer-letter rows of the head stay unquantized
            nn.quantize(language_model.model, bits=self.bits, group_size=64,
                        class_predicate=lambda _, m: isinstance(m, nn.Linear))
        self.backbone = language_model.model
        self.head_weight = (language_model.lm_head.weight if hasattr(language_model, "lm_head")
                            else self.backbone.embed_tokens.weight)
        self.prefix = self._shared_prefix() if prefix else None

    @property
    def scoring(self) -> str:
        if self.prefix is None:
            return "one forward pass per text"
        return f"one forward pass per text, with the shared prompt start ({len(self.prefix)} tokens) computed once"

    def encode(self, text: str, labels: dict[str, str]) -> tuple[list[int], list[str]]:
        """Prompt token ids (jevk5's exact prompt) and the option ids, in letter order."""
        q = question(self.instruction, labels)
        options = decision_options(q)
        ids = JevK5.encode(self._jevk5, text, q["instructions"], [t for _, t in options])
        return ids, [k for k, _ in options]

    def _head(self, hidden: mx.array, count: int) -> mx.array:
        weight = self.head_weight[mx.array(self.slots[:count])].astype(mx.float32)
        return (hidden.astype(mx.float32) @ weight.T)[0]

    def _plain(self, ids: list[int], count: int) -> mx.array:
        return self._head(self.backbone(mx.array([ids]))[:, -1, :], count)

    def _shared_prefix(self) -> mlx_prefix.SharedPrefix | None:
        probes = [self.encode(t, self.labels)[0] for t in PREFIX_PROBES]
        k = len(self.labels)
        return mlx_prefix.build(self.backbone, self.model.make_cache, probes,
                                plain_logits=lambda i: self._plain(probes[i], k),
                                head=lambda hidden, i: self._head(hidden, k), min_share=MIN_PREFIX_SHARE)

    def letter_logits(self, text: str, labels: dict[str, str]) -> tuple[list[float], list[str]]:
        ids, keys = self.encode(text, labels)
        if self.prefix is not None and self.prefix.matches(ids):
            logits = self._head(self.prefix.hidden(ids), len(keys))
        else:
            logits = self._plain(ids, len(keys))
        mx.eval(logits)
        return logits.tolist(), keys

    def _predict(self, text: str, labels: dict[str, str]) -> Prediction:
        logits, keys = self.letter_logits(text, labels)
        p = mx.softmax(mx.array(logits) / self.temperature).tolist()
        by_name = {LABEL_NAMES.get(l, l): l for l in labels}
        probabilities = {by_name[k]: v for k, v in zip(keys, p)}
        label = max(probabilities, key=probabilities.get)
        # Confidence = max p, the probability of the chosen label.
        return Prediction(label, probabilities[label], probabilities)

    def predict(self, text: str) -> Prediction:
        return self._predict(text, self.labels)

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        """Texts one after another (one forward pass each; batching gives no speed-up on the Mac GPU)."""
        return [self._predict(t, self.labels) for t in texts]

    def predict_shortlisted(self, texts: list[str], label_sets: list[dict[str, str]], _=None) -> list[Prediction]:
        """Classify each text among its own labels (a shortlist); other labels get probability 0."""
        return [self._predict(t, s) for t, s in zip(texts, label_sets)]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    text = " ".join(sys.argv[1:])
    pred = PlumbClassifier(TASK_LABELS, TASK.instruction).predict(text)
    print(f"label:      {pred.label}")
    print(f"confidence: {pred.confidence:.2f}")
    for label, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<10} {p:.3f}")


if __name__ == "__main__":
    main()

"""Nimble Classifier: local text classification with Bespoke-Nimble-9B (Qwen3.5-9B + LoRA) on Apple Silicon MLX.

Nimble (https://github.com/bespokelabsai/nimble) is an open, Jev-style "System One" decision model: it
takes a text and a flat schema of enum/boolean fields and, for each field, reads the logits of one-token
answer codes and softmaxes them. No text is generated. It is trained from Qwen3.5-9B with LoRA on
contrastive, synthetic decision data (not distilled from Jev). Runs locally on the Mac GPU via MLX.

Each text is the `context`, scored against one enum field whose choices are the class labels (up to 255;
more than 26 use Nimble's wide candidate codes). Each choice's description starts with the label's name.

Speed: Nimble's prompts and candidate logits are used unchanged; only how they are computed is chosen for a
single question. One forward pass per text ("independent": ~17% faster than Nimble's default multi-field
"parallel" mode), and, when the prompt start shared by every text (system prompt and chat header, before the
text) is a large enough share of the prompt, that start is computed once and reused, but only if it gives the
same logits as the plain pass on probe texts (AG News: ~1.4x faster; Banking77's long schema: not used).
Batching texts was measured and left out: no faster on the Mac GPU, and it shifts logits slightly.

Usage:
    python classifier.py "Stocks rallied after the Fed held rates steady."

Environment (optional, read from the root .env):
    NIMBLE_MODEL_PATH   merged checkpoint folder (default: models/<pinned revision>/, made by setup.sh)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
# Nimble is not a pip package: setup.sh checks it out into src/ at a pinned commit.
sys.path.insert(0, str(APP_DIR / "src"))

import mlx.core as mx  # noqa: E402
from nimble.scoring.parallel_scorer import ParallelScorer, candidate_projection  # noqa: E402

import mlx_prefix  # noqa: E402  (shared/: the shared-prompt-start cache)
from task import current_task, load_split  # noqa: E402

# .env lives in the repo root (classifiers/<app>/ -> two levels up).
load_dotenv(APP_DIR.parents[1] / ".env")

# The release setup.sh downloads and merges (keep in step with setup.sh).
MODEL = "bespokelabs/Bespoke-Nimble-9B"
REVISION = "bd792f44ec8e265be861bfcdf4e05967ffe0e858"
BASE_MODEL = "Qwen/Qwen3.5-9B"
MAX_INPUT_TOKENS = 8192  # the release's prompt limit (schema included)
# Reuse the shared prompt start only when it is at least this share of a prompt: below that, the saving is
# smaller than the cost of forking its cache (measured: 84 of ~266 tokens for AG News pays, 82 of ~2,290 does not).
MIN_PREFIX_SHARE = 0.10
PREFIX_PROBES = ("Alpha probe text about markets.", "Zebra probe sentence on sports.", "123 probe of a third kind.")

# Labels, descriptions and instruction come from the task manifest (data/tasks/<task>.json, see task.py),
# chosen with --task on evaluate.py (default: ag_news).
TASK = current_task()
_LABEL_SPECS = TASK.labels
CLASS_TO_LABEL = {spec["raw"]: spec["id"] for spec in _LABEL_SPECS}
TASK_LABELS: dict[str, str] = {spec["id"]: spec["description"] for spec in _LABEL_SPECS}
LABEL_NAMES = {spec["id"]: spec["name"] for spec in _LABEL_SPECS}


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class NimbleClassifier:
    """Single-label text classifier backed by one Nimble enum field."""

    def __init__(
        self,
        labels: dict[str, str],
        instructions: str = "Which category does this text belong to?",
        model_path: str | None = None,
    ) -> None:
        self.labels = labels
        self.instructions = instructions
        self.model_path = Path(model_path or os.getenv("NIMBLE_MODEL_PATH") or APP_DIR / "models" / REVISION)
        if not (self.model_path / "READY.json").is_file():
            raise FileNotFoundError(f"No merged Nimble checkpoint in {self.model_path}: run classifiers/nimble/setup.sh")
        # Temperature: the release's own default, chosen by Nimble from the adapter hash in READY.json
        # (T = 1.0 for this release, which has no separate fit).
        self.scorer = ParallelScorer(model_path=str(self.model_path), model_id=MODEL, revision=REVISION,
                                     max_input_tokens=MAX_INPUT_TOKENS)
        self.temperature = self.scorer.temperature
        self.schema = self._schema()
        self.prefix = self._shared_prefix()

    @property
    def scoring(self) -> str:
        """How texts are scored, for the run metadata."""
        if self.prefix is None:
            return "independent: one forward pass per text"
        return f"independent, with the shared prompt start ({len(self.prefix)} tokens) computed once"

    def _independent(self, prepared) -> mx.array:
        rows, _ = self.scorer.evaluate(prepared, mode="independent")
        return rows[0]

    def _head(self, hidden: mx.array, candidates: list[int]) -> mx.array:
        return candidate_projection(hidden, self.scorer.head_weight, candidates)[0]

    def _shared_prefix(self) -> mlx_prefix.SharedPrefix | None:
        """The prompt start every text shares (shared/mlx_prefix.py), computed once, or None when it is too
        small a share of the prompt or its cached path does not reproduce the plain logits."""
        prepared = [self.scorer.prepare(text, self.schema) for text in PREFIX_PROBES]
        return mlx_prefix.build(self.scorer.backbone, self.scorer.model.make_cache, [p.full_ids[0] for p in prepared],
                                plain_logits=lambda i: self._independent(prepared[i]),
                                head=lambda hidden, i: self._head(hidden, prepared[i].candidate_ids[0]),
                                min_share=MIN_PREFIX_SHARE)

    def _schema(self) -> dict:
        return {"label": {
            "type": "enum",
            "description": self.instructions,
            "choices": list(self.labels),
            "choice_descriptions": {label: f"{LABEL_NAMES.get(label, label)}: {description}"
                                    for label, description in self.labels.items()},
        }}

    def predict(self, text: str) -> Prediction:
        # The same prompt, candidates and softmax(logits / T) as ParallelScorer.score(); see the module docstring.
        prepared = self.scorer.prepare(text, self.schema)
        ids, candidates = prepared.full_ids[0], prepared.candidate_ids[0]
        if self.prefix is not None and self.prefix.matches(ids):
            logits = self._head(self.prefix.hidden(ids), candidates)
            mx.eval(logits)
        else:
            logits = self._independent(prepared)
        scores = mx.softmax(logits.astype(mx.float32) / self.temperature).tolist()
        probabilities = dict(zip(prepared.choices[0], scores))
        label = max(probabilities, key=probabilities.get)
        # Confidence = max p, the probability of the chosen label. Nimble reports no separate score.
        return Prediction(label, probabilities[label], probabilities)

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        """Score texts one after another (batching was measured: no faster on the Mac GPU; see the module docstring)."""
        return [self.predict(text) for text in texts]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    text = " ".join(sys.argv[1:])
    pred = NimbleClassifier(TASK_LABELS, TASK.instruction).predict(text)
    print(f"label:      {pred.label}")
    print(f"confidence: {pred.confidence:.2f}")
    for label, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<10} {p:.3f}")


if __name__ == "__main__":
    main()

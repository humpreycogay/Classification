"""ModernBERT-JEV Classifier: zero-shot choice scoring with ModernBERT-JEV (ali-rehman-ML/modern-bert-jev).

ModernBERT-JEV is a candidate-conditioned scorer: ModernBERT-base with a LoRA adapter (rank 16 on the attention
projections) and a one-unit head, trained on the jev-bench choice sources. Each option is scored on its own:

    segment A  "Context: <text>"                                     (the only part truncated, 512 tokens)
    segment B  "Question: <instruction>\\nCandidate: <description>"   (jev-bench's own format: the criteria
                                                                       descriptions, as bench_data.normalize renders them)

The masked mean of the last hidden state goes through the head to one logit per option; the logits are divided
by the shipped temperature (calibration.json, T = 1.2023) and softmaxed across the options.

The model code is the repo's own (src/, cloned at a pinned commit by setup.sh): `choice_model.ChoiceScorer` and
its tokenization rules. Two changes for this Mac: the repo's `Predictor` picks CUDA or CPU only, so the scorer is
built here on MPS; and the LoRA is folded into the base weights once at load (one matmul per layer less).

Speed: all (text, option) pairs of a batch of texts are flattened, sorted by length and scored in chunks of
`choice_chunk` pairs (config.json). Options are scored independently, so batching only changes padding.

Usage:
    python classifier.py "Stocks rallied after the Fed held rates steady."

Environment (optional, read from the root .env):
    MODERNBERT_JEV_DEVICE    mps | cuda | cpu    (default: auto-detect)
    MODERNBERT_JEV_BACKEND   torch | onnx        (default: torch; onnx = the shipped fp32 ONNX graph on CPU)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Xet transfers stall on some networks; plain HTTPS downloads are reliable.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from torch import nn  # noqa: E402

from batching import length_order, restore
from task import current_task

APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR / "src"  # github.com/ali-rehman-ML/modern-bert-jev at REPO_COMMIT (setup.sh)
sys.path.insert(0, str(SRC_DIR))
import choice_model  # noqa: E402

# .env lives in the repo root (classifiers/<app>/ -> two levels up).
load_dotenv(APP_DIR.parents[1] / ".env")

REPO_COMMIT = "c74ba60a40f287085c77547078e374b6526fe645"
ADAPTER_REPO = "ali-rehman-ML/modern-bert-jev"
ADAPTER_REVISION = "133d6c8487e1ae2b2e4b4854f0412c2617c8d95e"
BACKBONE = "answerdotai/ModernBERT-base"  # revision pinned in src/choice_model.py MODELS (8949b90)

# The repo resolves the backbone relative to the working directory (models/<name>); setup.sh puts it in src/models/.
choice_model.model_dir = lambda model=BACKBONE: SRC_DIR / "models" / model.rsplit("/", 1)[-1]

# Labels, descriptions and instruction come from the task manifest (data/tasks/<task>.json, see task.py),
# chosen with --task on evaluate.py (default: ag_news).
TASK = current_task()
_LABEL_SPECS = TASK.labels
CLASS_TO_LABEL = {spec["raw"]: spec["id"] for spec in _LABEL_SPECS}
TASK_LABELS: dict[str, str] = {spec["id"]: spec["description"] for spec in _LABEL_SPECS}
LABEL_NAMES = {spec["id"]: spec["name"] for spec in _LABEL_SPECS}


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def release_dir(onnx: bool = False) -> Path:
    """The adapter release (adapter.pt, calibration.json, config.json, ...) at the pinned revision."""
    return Path(snapshot_download(ADAPTER_REPO, revision=ADAPTER_REVISION,
                                  allow_patterns=None if onnx else ["*.json", "adapter.pt"]))


def fold_lora(model: nn.Module) -> int:
    """Replace every LoRALinear by its base Linear with W + scale * B @ A folded in; returns the count."""
    folded = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, choice_model.LoRALinear):
            with torch.no_grad():
                module.base.weight += module.scale * (module.B @ module.A)
            parent, attribute = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), attribute, module.base)
            folded += 1
    return folded


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class ModernBertJevClassifier:
    """Single-label text classifier: one ModernBERT-JEV score per (text, option) pair, softmaxed over options."""

    def __init__(self, labels: dict[str, str], instruction: str, device: str | None = None,
                 backend: str | None = None, fold: bool = True) -> None:
        self.labels = labels
        self.instruction = instruction
        self.backend = backend or os.getenv("MODERNBERT_JEV_BACKEND") or "torch"
        if self.backend not in ("torch", "onnx"):
            raise ValueError(f"MODERNBERT_JEV_BACKEND must be torch or onnx, not {self.backend!r}")
        run = release_dir(onnx=self.backend == "onnx")
        self.config = json.loads((run / "config.json").read_text())
        self.temperature = float(json.loads((run / "calibration.json").read_text())["temperature"])
        self.max_length = int(self.config["max_length"])
        self.chunk = int(self.config["choice_chunk"])
        # Candidate text per option: its description, as in training (jev-bench criteria map ids to descriptions).
        # On 200 AG News calibration texts (carved from train): description 63%, "<name>: <description>" 47%.
        self.candidates = {l: d if d.strip() else LABEL_NAMES.get(l, l) for l, d in labels.items()}
        self.tok = choice_model.tokenizer(self.config["model"])
        self.overhead = self.tok.num_special_tokens_to_add(pair=True)
        if self.backend == "onnx":
            from onnx_scorer import Scorer

            self.device = "cpu"
            self.scorer = Scorer(run / "onnx" / "model.onnx", choice_model.model_dir(self.config["model"]) /
                                 "tokenizer.json", self.temperature, self.max_length)
            self.folded = 0
            return
        self.device = device or os.getenv("MODERNBERT_JEV_DEVICE") or default_device()
        self.model = choice_model.ChoiceScorer(self.config["lora_rank"], self.config["lora_alpha"], "cpu",
                                               self.config["model"])
        self.model.load_adapter(run / "adapter.pt")
        self.folded = fold_lora(self.model) if fold else 0
        self.model.to(self.device).eval()

    def _right(self, candidate: str) -> str:
        return f"Question: {self.instruction}\nCandidate: {candidate}"

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """One raw logit per (segment A, segment B) pair; length-sorted chunks, results in input order."""
        enc = self.tok([a for a, _ in pairs], [b for _, b in pairs], truncation="only_first",
                       max_length=self.max_length)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        order = length_order(ids)
        scores: list[float] = []
        for start in range(0, len(order), self.chunk):
            part = order[start:start + self.chunk]
            batch = self.tok.pad({"input_ids": [ids[i] for i in part], "attention_mask": [mask[i] for i in part]},
                                 return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.inference_mode():
                scores += self.model(batch).float().cpu().tolist()
        return restore(order, scores)

    def raw_logits(self, texts: list[str], candidates: list[dict[str, str]], question: list[str] | None = None
                   ) -> list[list[float]]:
        """Raw (untempered) logits per text over its own candidate texts (the parity check uses jev-bench rows)."""
        questions = question or [self.instruction] * len(texts)
        pairs, spans = [], []
        for text, cands, q in zip(texts, candidates, questions):
            context = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, sort_keys=True)
            spans.append((len(pairs), len(cands)))
            pairs += [(f"Context: {context}", f"Question: {q}\nCandidate: {c}") for c in cands.values()]
        flat = self._score_pairs(pairs)
        return [flat[s:s + n] for s, n in spans]

    def _to_prediction(self, logits: list[float]) -> Prediction:
        p = (torch.tensor(logits) / self.temperature).softmax(0).tolist()
        probabilities = dict(zip(self.labels, p))
        label = max(probabilities, key=probabilities.get)
        # Confidence = max p, the probability of the chosen label.
        return Prediction(label, probabilities[label], probabilities)

    def predict_many(self, texts: list[str], batch_size: int = 1) -> list[Prediction]:
        if self.backend == "onnx":
            out = []
            for text in texts:
                r = self.scorer(text, self.instruction, self.candidates)
                out.append(Prediction(r["choice"], r["max_probability"], r["probabilities"]))
            return out
        preds: list[Prediction] = []
        for start in range(0, len(texts), max(1, batch_size)):
            chunk = texts[start:start + batch_size]
            preds += [self._to_prediction(l) for l in self.raw_logits(chunk, [self.candidates] * len(chunk))]
        return preds

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    text = " ".join(sys.argv[1:])
    pred = ModernBertJevClassifier(TASK_LABELS, TASK.instruction).predict(text)
    print(f"label:      {pred.label}")
    print(f"confidence: {pred.confidence:.2f}")
    for label, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<10} {p:.3f}")


if __name__ == "__main__":
    main()

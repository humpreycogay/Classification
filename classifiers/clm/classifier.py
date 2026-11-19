"""CLM Classifier: zero-shot classification with CLM-v0.1-8B (Contrastive-LM/CLM-v0.1-8B) on a local Qwen3-8B.

CLM is a contrastive "System One" model: a frozen Qwen3-8B encoder (last-token pooling, L2-normalized) and two
small projection heads (~75 MB), one for states and one for actions. A (state, action) pair scores
exp(logit_scale) * cos(state_head(s), action_head(a)); a softmax over the options is the answer.

The reference stack serves the encoder with vLLM (CUDA only). Here Qwen3-8B runs in-process with transformers on
MPS in BF16, and only that encoder call is replaced: the state and action texts (clm.schema), the heads (clm.heads)
and the score come from the contrastive-lm package, unchanged. The state is the text with the task's question
appended (clm.schema.state_text); each action is "<name>: <description>". Pooling as vLLM's: the hidden state of
the last token (inputs left-padded, so it is the last position), states truncated to their last 2,048 tokens.

Speed: the label embeddings are computed once at load (they are fixed per task); each text then costs one encoder
pass. Texts can be batched longest first (shared/batching.py; left padding with position ids from the attention
mask), but in BF16 padding moves the scores slightly, so the evaluate.py default is one text per pass.

Calibration: CLM's scores are relative, not calibrated. The softmax temperature is fitted on the task's
calibration split (carved from train by task.py, never the test set) by minimum log loss, as embedsim does, and
cached. CLM was trained on agent actions, not topics: expect lower accuracy than the topic-trained apps.

Usage:
    python classifier.py "Stocks rallied after the Fed held rates steady."

Environment (optional, read from the root .env):
    CLM_DEVICE        mps | cuda | cpu   (default: auto-detect)
    CLM_TEMPERATURE   fixed softmax temperature (default: unset = fitted on the calibration split)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Xet transfers stall on some networks; plain HTTPS downloads are reliable.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch  # noqa: E402
from clm.heads import HF_FILE, HeadPair  # noqa: E402
from clm.schema import candidates, state_text  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from batching import sorted_batches
from task import current_task

# .env lives in the repo root (classifiers/<app>/ -> two levels up).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

ENCODER = "Qwen/Qwen3-8B"
ENCODER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
HEADS = "Contrastive-LM/CLM-v0.1-8B"
HEADS_REVISION = "e939398d4556fcd9400c76fa8c5a513202f42b0a"
MAX_TOKENS = 2048  # the reference server's limit (vLLM truncate_prompt_tokens: the last 2,048 tokens are kept)
TEMPERATURE_GRID = [0.05 * (20 / 0.05) ** (i / 199) for i in range(200)]  # geometric, 0.05 to 20
CACHE_DIR = Path(__file__).resolve().parent / "results" / "cache"
ACTION_TEMPLATE = "{name}: {description}"

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


def encoder_dir() -> str:
    return snapshot_download(ENCODER, revision=ENCODER_REVISION,
                             allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"])


def heads_path() -> str:
    return hf_hub_download(HEADS, HF_FILE, revision=HEADS_REVISION)


def softmax(values: list[float]) -> list[float]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class CLMClassifier:
    """Single-label text classifier: CLM scores of the text (state) against each label (action)."""

    def __init__(self, labels: dict[str, str], instruction: str, device: str | None = None,
                 dtype: torch.dtype = torch.bfloat16) -> None:
        self.labels = labels
        self.instruction = instruction
        self.device = device or os.getenv("CLM_DEVICE") or default_device()
        self.dtype = dtype
        path = encoder_dir()
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"  # the last position is every row's last real token
        self.encoder = AutoModel.from_pretrained(path, dtype=dtype).to(self.device).eval()
        self.heads = HeadPair("clm-latest", heads_path(), self.device).ensure()
        self.scale = self.heads.scale
        # Actions: clm.schema.candidates() of a choice question = each option's own text, unprefixed.
        question = {"type": "choice", "instructions": instruction,
                    "criteria": {l: ACTION_TEMPLATE.format(name=LABEL_NAMES.get(l, l), description=d)
                                 for l, d in labels.items()}}
        self.label_ids, self.action_texts = candidates(question)
        self.action_vectors = self.heads.project_actions(self.embed(self.action_texts).numpy())  # once, at load
        self.temperature = float(os.getenv("CLM_TEMPERATURE") or 1.0)
        self.temperature_source = "CLM_TEMPERATURE" if os.getenv("CLM_TEMPERATURE") else "1.0 (not calibrated)"

    def embed(self, texts: list[str], batch_size: int = 1) -> torch.Tensor:
        """[n, 4096] L2-normalized last-token hidden states (float32, CPU), as the reference vLLM pooling."""
        def run(chunk: list[str]) -> list[torch.Tensor]:
            ids = [self.tok.encode(t, add_special_tokens=True)[-MAX_TOKENS:] for t in chunk]
            batch = self.tok.pad({"input_ids": ids}, return_tensors="pt").to(self.device)
            # Left padding: every text's positions start at 0, as when it is encoded alone.
            positions = (batch["attention_mask"].cumsum(-1) - 1).clamp(min=0)
            with torch.inference_mode():
                hidden = self.encoder(**batch, position_ids=positions).last_hidden_state[:, -1, :].float()
            return list(torch.nn.functional.normalize(hidden, dim=-1).cpu())

        return torch.stack(sorted_batches(texts, batch_size, run))

    def scores(self, texts: list[str], batch_size: int = 1) -> torch.Tensor:
        """[n, labels] exp(logit_scale) * cosine, before the temperature."""
        states = [state_text(t, self.instruction) for t in texts]
        zs = self.heads.project_states(self.embed(states, batch_size).numpy())
        return (self.scale * (zs @ self.action_vectors.T)).float().cpu()

    def calibrate(self, texts: list[str], labels: list[str], batch_size: int = 1) -> dict:
        """Fit the temperature on labeled calibration texts (minimum log loss), cached per task, encoder, heads
        and label texts; a CLM_TEMPERATURE setting wins. Returns where the temperature came from."""
        if os.getenv("CLM_TEMPERATURE"):
            return {"temperature": self.temperature, "source": "CLM_TEMPERATURE"}
        key = hashlib.sha1(json.dumps([TASK.name, ENCODER_REVISION, HEADS_REVISION, str(self.dtype), self.instruction,
                                       self.action_texts, texts, labels]).encode()).hexdigest()[:16]
        path = CACHE_DIR / f"temperature_{TASK.name}_{key}.json"
        if path.exists():
            info = json.loads(path.read_text())
        else:
            rows = self.scores(texts, batch_size).tolist()
            true_idx = [self.label_ids.index(label) for label in labels]
            best_t, best_loss = 1.0, float("inf")
            for t in TEMPERATURE_GRID:
                loss = -sum(math.log(max(softmax([s / t for s in row])[j], 1e-12))
                            for row, j in zip(rows, true_idx)) / len(rows)
                if loss < best_loss:
                    best_t, best_loss = t, loss
            info = {"temperature": round(best_t, 5), "log_loss": round(best_loss, 5), "n_calibration": len(texts),
                    "grid": [round(TEMPERATURE_GRID[0], 4), round(TEMPERATURE_GRID[-1], 4)],
                    "source": f"fitted on the {TASK.name} calibration split ({len(texts)} train texts)"}
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(info, indent=2) + "\n")
        self.temperature, self.temperature_source = info["temperature"], info["source"]
        return info

    def predict_many(self, texts: list[str], batch_size: int = 1) -> list[Prediction]:
        preds = []
        for row in self.scores(texts, batch_size).tolist():
            probabilities = dict(zip(self.label_ids, softmax([s / self.temperature for s in row])))
            label = max(probabilities, key=probabilities.get)
            # Confidence = max p, the probability of the chosen label (temperature fitted; see module doc).
            preds.append(Prediction(label, probabilities[label], probabilities))
        return preds

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    text = " ".join(sys.argv[1:])
    pred = CLMClassifier(TASK_LABELS, TASK.instruction).predict(text)
    print(f"label:      {pred.label}")
    print(f"confidence: {pred.confidence:.2f}")
    for label, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<10} {p:.3f}")


if __name__ == "__main__":
    main()

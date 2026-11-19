"""Naluri encoder decision classifiers using each checkpoint's bundled custom head runtime."""
from __future__ import annotations
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.environ.get("BENCHMARK_SHARED_DIR") or str(Path(__file__).resolve().parents[2] / "shared"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from task import current_task  # noqa: E402

MODELS = {
    "279m": ("jerichosiahaya/Naluri-279M-encoder", "ba8c4c17c1af0790bc00ad9fbf407eb8ea539821"),
    "308m": ("jerichosiahaya/Naluri-308M-encoder", "f5d1a9feb1ab8ef3a5fd3a287258182110559416"),
}
MAX_OPTIONS = 20
MAX_LENGTH = 512
HEAD_MAX_LENGTH = 192
TASK = current_task()
TASK_LABELS = {s["id"]: s["description"] for s in TASK.labels}
LABEL_NAMES = {s["id"]: s["name"] for s in TASK.labels}

@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]

def default_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def load_runtime(variant: str, device: str):
    repo, revision = MODELS[variant]
    root = Path(snapshot_download(repo, revision=revision))
    spec = importlib.util.spec_from_file_location(f"naluri_{variant}_{revision[:7]}", root / "naluri.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    tok, model = module.load(root, device=device)
    return module, tok, model

class NaluriClassifier:
    def __init__(self, labels: dict[str, str], instruction: str, variant="308m", device=None, runtime=None) -> None:
        if variant not in MODELS: raise ValueError(f"unknown Naluri variant {variant!r}")
        if not 2 <= len(labels) <= MAX_OPTIONS:
            raise ValueError(f"Naluri supports 2-{MAX_OPTIONS} benchmark options without destructive option truncation")
        self.variant = variant
        self.device = device or os.getenv("NALURI_DEVICE") or default_device()
        if self.device not in {"cuda", "mps", "cpu"}: raise ValueError("NALURI_DEVICE must be cuda, mps or cpu")
        self.labels = labels
        self.label_of = {LABEL_NAMES.get(k, k): k for k in labels}
        if len(self.label_of) != len(labels): raise ValueError("label names must be unique")
        self.question = {"label": {"type": "choice", "instructions": instruction,
                                    "criteria": {LABEL_NAMES.get(k, k): v for k, v in labels.items()}}}
        self.runtime = runtime or load_runtime(variant, self.device)

    def _predict(self, text):
        module, tok, model = self.runtime
        native = module.predict(tok, model, text, self.question, device=self.device)["label"]
        if set(native) != set(self.label_of): raise RuntimeError("Naluri returned unexpected options")
        total = sum(float(v) for v in native.values())
        if not 0.999 <= total <= 1.001: raise RuntimeError(f"Naluri probabilities sum to {total}")
        probabilities = {self.label_of[k]: float(v) for k, v in native.items()}
        label = max(probabilities, key=probabilities.get)
        return Prediction(label, probabilities[label], probabilities)

    def predict(self, text): return self._predict(text)
    def predict_many(self, texts): return [self._predict(t) for t in texts]

if __name__ == "__main__": print(NaluriClassifier(TASK_LABELS, TASK.instruction).predict(" ".join(sys.argv[1:])))
"""Small neural text classifiers trained from scratch: tokenizer, vocabulary, batching, training loop,
temperature scaling and a model cache.

Part of the shared benchmark library (shared/); used by scratch_fasttext, scratch_cnn and scratch_bilstm. Only the
task's train split is used: a stratified 90% trains the network, the other 10% (seed 0) chooses the best epoch and
fits the temperature. The test split is never read here.

Each app supplies only its architecture (an nn.Module whose forward(x) maps a LongTensor (batch, length) of token
ids, 0 = padding, to logits (batch, labels)) and its per-task Config.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

VERSION = 1  # bump when tokenization, training or caching changes: old models are then retrained
PAD, UNK = 0, 1
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")  # words (with 's, n't) and single punctuation marks


# --------------------------------------------------------------------------------------------------------------
# Text to ids
# --------------------------------------------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Lowercase words and punctuation: "My card hasn't arrived!" -> my card hasn't arrived !"""
    return TOKEN.findall(text.lower())


class Vocab:
    """Word <-> id. Ids 0 and 1 are padding and unknown; the rest are train words by frequency."""

    def __init__(self, itos: list[str]) -> None:
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    @classmethod
    def build(cls, texts, min_freq: int, max_size: int) -> "Vocab":
        counts = Counter(tok for text in texts for tok in tokenize(text))
        words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
        return cls(["<pad>", "<unk>"] + words)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(tok, UNK) for tok in tokenize(text)][:max_len]
        return ids or [UNK]  # never empty: every model needs at least one position

    def __len__(self) -> int:
        return len(self.itos)


def bigram_ids(ids: list[int], buckets: int, offset: int) -> list[int]:
    """fastText's hashing trick: each adjacent word pair -> one of `buckets` extra embedding rows."""
    return [offset + (a * 1_000_003 + b) % buckets for a, b in zip(ids, ids[1:])]


@dataclass
class Config:
    max_len: int = 100            # tokens kept per text (AG News ~45 on average, Banking77 ~12)
    min_freq: int = 2             # a word needs this many train occurrences to get its own vector
    max_vocab: int = 100_000
    bigrams: int = 0              # hash buckets for word pairs (fastText-style only); 0 = off
    batch_size: int = 64          # training batch
    lr: float = 1e-3              # peak learning rate (one-cycle schedule)
    weight_decay: float = 0.0
    max_epochs: int = 10
    patience: int = 2             # stop after this many epochs without a better validation accuracy
    val_frac: float = 0.1
    seed: int = 0
    model: dict = field(default_factory=dict)  # architecture settings passed to the model's constructor


def encode_all(vocab: Vocab, texts, cfg: Config) -> list[list[int]]:
    out = []
    for text in texts:
        ids = vocab.encode(text, cfg.max_len)
        if cfg.bigrams:
            ids = ids + bigram_ids(ids, cfg.bigrams, len(vocab))
        out.append(ids)
    return out


def batches(seqs, labels, batch_size: int, rng: random.Random | None):
    """(row indices, padded ids, labels). Shuffled for training (rng); sorted by length otherwise (less padding)."""
    order = list(range(len(seqs)))
    if rng:
        rng.shuffle(order)
    else:
        order.sort(key=lambda i: len(seqs[i]))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        x = torch.full((len(idx), max(len(seqs[i]) for i in idx)), PAD, dtype=torch.long)
        for row, i in enumerate(idx):
            x[row, :len(seqs[i])] = torch.tensor(seqs[i])
        y = torch.tensor([labels[i] for i in idx]) if labels is not None else None
        yield idx, x, y


def split_train_val(df: pd.DataFrame, frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split of the TRAIN split into fit and validation parts."""
    val = df.groupby("label", group_keys=False).sample(frac=frac, random_state=seed)
    return df.drop(val.index).reset_index(drop=True), val.reset_index(drop=True)


def data_sha1(texts, labels) -> str:
    h = hashlib.sha1()
    h.update("\n".join(texts).encode())
    h.update("\n".join(labels).encode())
    return h.hexdigest()


def pick_device(env_var: str, default: str = "auto") -> str:
    choice = os.getenv(env_var) or default
    if choice != "auto":
        return choice
    return "mps" if torch.backends.mps.is_available() else "cpu"


# --------------------------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def logits_for(model: nn.Module, seqs, batch_size: int, device: str) -> torch.Tensor:
    """Logits for every sequence, in input order (float32, on the CPU)."""
    model.eval()
    out = torch.empty(len(seqs), model.n_labels)
    for idx, x, _ in batches(seqs, None, batch_size, None):
        out[idx] = model(x.to(device)).float().cpu()
    return out


def train_loop(model, fit_x, fit_y, val_x, val_y, cfg: Config, device: str, log=print) -> tuple[int, list[dict]]:
    """AdamW + one-cycle learning rate + gradient clipping; keeps the epoch with the best validation accuracy.
    Returns (best epoch, per-epoch history). The model is left holding the best epoch's weights."""
    rng = random.Random(cfg.seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.max_epochs * math.ceil(len(fit_x) / cfg.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps, pct_start=0.1)
    vy = torch.tensor(val_y)
    best_acc, best_state, best_epoch, stale, history = -1.0, None, 0, 0, []
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        start, total = time.perf_counter(), torch.zeros((), device=device)
        for _, x, y in batches(fit_x, fit_y, cfg.batch_size, rng):
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.detach() * len(y)  # no .item() per step: avoids a GPU sync on every batch
        val_logits = logits_for(model, val_x, 512, device)
        row = {"epoch": epoch, "train_loss": round(total.item() / len(fit_x), 4),
               "val_loss": round(F.cross_entropy(val_logits, vy).item(), 4),
               "val_acc": round((val_logits.argmax(1) == vy).float().mean().item(), 4),
               "seconds": round(time.perf_counter() - start, 1)}
        history.append(row)
        log(f"  epoch {epoch:>2}: train loss {row['train_loss']:.4f}  val loss {row['val_loss']:.4f}  "
            f"val acc {row['val_acc']:.4f}  ({row['seconds']}s)")
        if row["val_acc"] > best_acc:
            best_acc, best_epoch, stale = row["val_acc"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                log(f"  early stop: no better validation accuracy for {cfg.patience} epochs")
                break
    model.load_state_dict(best_state)
    return best_epoch, history


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    """One T > 0 minimising validation cross-entropy of softmax(logits / T). Changes confidence, never the label."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


# --------------------------------------------------------------------------------------------------------------
# The classifier every scratch app wraps
# --------------------------------------------------------------------------------------------------------------
@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class ScratchClassifier:
    """Train (or load) a from-scratch network on a train DataFrame (text, label); classify texts."""

    def __init__(self, app: str, make_model: Callable[..., nn.Module], cfg: Config, labels: list[str], device: str,
                 description: str) -> None:
        self.app, self.make_model, self.cfg, self.labels, self.device = app, make_model, cfg, labels, device
        self.index = {label: i for i, label in enumerate(labels)}
        self.model: nn.Module | None = None
        self.vocab: Vocab | None = None
        self.temperature = 1.0
        self.model_description = f"{description}, trained from scratch ({device})"
        self.calibration = "softmax(logits / T), T fitted on a stratified 10% validation part of the train split"

    def _new_model(self) -> nn.Module:
        rows = len(self.vocab) + self.cfg.bigrams  # word vectors + hashed bigram vectors
        return self.make_model(rows, len(self.labels), **self.cfg.model)

    def fit_cached(self, train: pd.DataFrame, models_dir: Path, log=print) -> dict:
        """Train on `train`, or load the identical earlier model from models_dir. Returns training metadata;
        '_load_seconds' is the time to load a cached model (load time, not training time)."""
        texts, labels = train["text"].tolist(), train["label"].tolist()
        ident = {"app": self.app, "config": asdict(self.cfg), "labels": self.labels,
                 "train_sha1": data_sha1(texts, labels), "torch": torch.__version__, "textnn": VERSION}
        digest = hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
        weights, meta_path = models_dir / f"model_{digest}.pt", models_dir / f"model_{digest}.json"
        if weights.exists() and meta_path.exists():
            start = time.perf_counter()
            meta = json.loads(meta_path.read_text())
            self.vocab, self.temperature = Vocab(meta["vocab"]), meta["temperature"]
            self.model = self._new_model()
            self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            self.model.to(self.device).eval()
            log(f"Loaded the trained model {weights.name}")
            return {**self._summary(meta), "train_cached": True, "_load_seconds": time.perf_counter() - start}

        start = time.perf_counter()
        fit, val = split_train_val(train, self.cfg.val_frac, self.cfg.seed)
        self.vocab = Vocab.build(fit["text"], self.cfg.min_freq, self.cfg.max_vocab)  # from the fit part only
        fit_x, val_x = encode_all(self.vocab, fit["text"], self.cfg), encode_all(self.vocab, val["text"], self.cfg)
        fit_y, val_y = [self.index[l] for l in fit["label"]], [self.index[l] for l in val["label"]]
        torch.manual_seed(self.cfg.seed)
        self.model = self._new_model()
        n_params = sum(p.numel() for p in self.model.parameters())
        log(f"Training {self.app} from scratch: {len(fit)} texts (+{len(val)} validation), vocabulary "
            f"{len(self.vocab):,}, {n_params:,} parameters, device {self.device}")
        best_epoch, history = train_loop(self.model, fit_x, fit_y, val_x, val_y, self.cfg, self.device, log)
        # Rounded as stored, so a fresh fit and a loaded one give identical probabilities.
        self.temperature = round(fit_temperature(logits_for(self.model, val_x, 512, self.device), torch.tensor(val_y)), 5)
        seconds = time.perf_counter() - start
        models_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), weights)
        meta = {**ident, "n_fit": len(fit), "n_val": len(val), "vocab_size": len(self.vocab), "parameters": n_params,
                "best_epoch": best_epoch, "history": history, "temperature": round(self.temperature, 5),
                "train_seconds": round(seconds, 1), "device": self.device, "vocab": self.vocab.itos}
        meta_path.write_text(json.dumps(meta) + "\n")
        log(f"Trained in {seconds:.0f}s; best epoch {best_epoch}; T = {self.temperature:.3f}; saved {weights.name}")
        return {**self._summary(meta), "train_cached": False, "_load_seconds": 0.0}

    def _summary(self, meta: dict) -> dict:
        keep = ("config", "n_fit", "n_val", "vocab_size", "parameters", "best_epoch", "history", "temperature",
                "train_seconds", "device", "torch", "train_sha1")
        return {"supervision": "supervised", "from_scratch": True, "calibration": self.calibration,
                **{k: meta[k] for k in keep}}

    def predict_many(self, texts: list[str], batch_size: int = 256) -> list[Prediction]:
        seqs = encode_all(self.vocab, texts, self.cfg)
        probs = torch.softmax(logits_for(self.model, seqs, batch_size, self.device) / self.temperature, dim=1)
        preds = []
        for row in probs.tolist():
            p = {label: row[i] for i, label in enumerate(self.labels)}
            label = max(p, key=p.get)
            preds.append(Prediction(label, p[label], p))  # confidence = max p
        return preds

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]


def add_training_arguments(parser) -> None:
    """evaluate.py options shared by the scratch apps. Any override is part of the model digest."""
    parser.add_argument("--epochs", type=int, help="maximum training epochs (default: the task's recipe)")
    parser.add_argument("--lr", type=float, help="peak learning rate")
    parser.add_argument("--train-batch-size", type=int, help="training batch size")
    parser.add_argument("--train-seed", type=int, help="seed for the validation split, initialisation and order")
    parser.add_argument("--batch-size", type=int, default=256, help="texts per forward pass when classifying")
    parser.add_argument("--device", help="cpu | mps | cuda (default: the app's SCRATCH_*_DEVICE, else its default)")


def config_from_args(cfg: Config, args) -> Config:
    overrides = {"max_epochs": args.epochs, "lr": args.lr, "batch_size": args.train_batch_size,
                 "seed": args.train_seed}
    return replace(cfg, **{k: v for k, v in overrides.items() if v is not None})

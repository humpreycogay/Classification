"""Fine-tuning a pre-trained encoder for single-label classification: tokenization, length-bucketed batches, the
training loop, a model cache and the classifier the fine-tune apps wrap.

Part of the shared benchmark library (shared/). Splits, temperature scaling and the Prediction type come from
textnn.py, so the fine-tuned and from-scratch models are trained and calibrated the same way: a stratified 90% of
the train split trains, the other 10% (seed 0) picks the best epoch and fits the temperature, the test split is
never read.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from textnn import Prediction, data_sha1, fit_temperature, split_train_val

# The load report on every from_pretrained() says the new classification head is "newly initialized": expected
# here, since the head is what fine-tuning trains. Keep transformers to errors only.
transformers.logging.set_verbosity_error()

VERSION = 1  # bump when the loop, batching or caching changes: old models are then retrained


@dataclass
class FTConfig:
    max_len: int = 160            # tokens per text (AG News: mean 55, 99th percentile 126; Banking77: 16 / 51)
    batch_size: int = 32
    lr: float = 3e-5              # peak learning rate
    weight_decay: float = 0.01
    warmup: float = 0.06          # share of steps with a rising learning rate
    max_epochs: int = 3
    patience: int = 1             # epochs without a better validation accuracy before stopping
    val_frac: float = 0.1
    seed: int = 0
    bf16: bool = True             # bfloat16 autocast while training (MPS/CUDA); classification is float32
    train_limit: int = 0          # 0 = the whole fit split; N = a stratified N-text subset (a quick pilot)


def tokenize_all(tok, texts, max_len: int) -> list[list[int]]:
    return tok(list(texts), truncation=True, max_length=max_len)["input_ids"]


def batches(seqs, labels, batch_size: int, pad_id: int, rng: random.Random | None):
    """(row indices, input_ids, attention_mask, labels).

    Training (rng): shuffle, sort by length inside chunks of 50 batches, then shuffle the batches: little padding,
    still random. Classifying (no rng): sorted by length, longest first.
    """
    order = list(range(len(seqs)))
    if rng:
        rng.shuffle(order)
        chunk = batch_size * 50
        order = [i for s in range(0, len(order), chunk)
                 for i in sorted(order[s:s + chunk], key=lambda i: len(seqs[i]))]
        groups = [order[s:s + batch_size] for s in range(0, len(order), batch_size)]
        rng.shuffle(groups)
    else:
        order.sort(key=lambda i: -len(seqs[i]))
        groups = [order[s:s + batch_size] for s in range(0, len(order), batch_size)]
    for idx in groups:
        width = max(len(seqs[i]) for i in idx)
        ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        for row, i in enumerate(idx):
            ids[row, :len(seqs[i])] = torch.tensor(seqs[i])
            mask[row, :len(seqs[i])] = 1
        y = torch.tensor([labels[i] for i in idx]) if labels is not None else None
        yield idx, ids, mask, y


@torch.no_grad()
def logits_for(model, seqs, batch_size: int, pad_id: int, device: str) -> torch.Tensor:
    """float32 logits for every sequence, in input order."""
    model.eval()
    out = torch.empty(len(seqs), model.config.num_labels)
    for idx, ids, mask, _ in batches(seqs, None, batch_size, pad_id, None):
        out[idx] = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float().cpu()
    return out


def train_loop(model, fit_x, fit_y, val_x, val_y, cfg: FTConfig, pad_id: int, device: str, log=print):
    """AdamW (no decay on biases and norms), linear warm-up and decay, gradient clipping, bfloat16 autocast; keeps
    the epoch with the best validation accuracy. Returns (best epoch, history); the model holds the best weights."""
    rng = random.Random(cfg.seed)
    model.to(device)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr)
    steps = cfg.max_epochs * math.ceil(len(fit_x) / cfg.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup * steps), steps)
    vy = torch.tensor(val_y)
    best_acc, best_state, best_epoch, stale, history = -1.0, None, 0, 0, []
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        start, total, done = time.perf_counter(), torch.zeros((), device=device), 0
        for step, (_, ids, mask, y) in enumerate(batches(fit_x, fit_y, cfg.batch_size, pad_id, rng), 1):
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=cfg.bf16 and device != "cpu"):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            loss = F.cross_entropy(logits.float(), y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.detach() * len(y)
            done += len(y)
            if step % 200 == 0:  # a progress line every 200 steps (about 3 minutes on AG News)
                log(f"    epoch {epoch} step {step}: {done}/{len(fit_x)} texts, "
                    f"{done / (time.perf_counter() - start):.0f} texts/s, loss {total.item() / done:.4f}")
        val_logits = logits_for(model, val_x, 64, pad_id, device)
        row = {"epoch": epoch, "train_loss": round(total.item() / len(fit_x), 4),
               "val_loss": round(F.cross_entropy(val_logits, vy).item(), 4),
               "val_acc": round((val_logits.argmax(1) == vy).float().mean().item(), 4),
               "seconds": round(time.perf_counter() - start, 1)}
        history.append(row)
        log(f"  epoch {epoch}: train loss {row['train_loss']:.4f}  val loss {row['val_loss']:.4f}  "
            f"val acc {row['val_acc']:.4f}  ({row['seconds']}s)")
        if row["val_acc"] > best_acc:
            best_acc, best_epoch, stale = row["val_acc"], epoch, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                log(f"  early stop: no better validation accuracy for {cfg.patience} epoch(s)")
                break
    model.load_state_dict(best_state)
    return best_epoch, history


class FineTuneClassifier:
    """Fine-tune (or load) a pre-trained encoder on a train DataFrame (text, label); classify texts."""

    def __init__(self, app: str, base_dir: Path, base_ref: str, cfg: FTConfig, labels: list[str], device: str,
                 description: str) -> None:
        self.app, self.base_dir, self.base_ref, self.cfg = app, base_dir, base_ref, cfg
        self.labels, self.device = labels, device
        self.index = {label: i for i, label in enumerate(labels)}
        self.tok = AutoTokenizer.from_pretrained(base_dir)
        self.pad_id = self.tok.pad_token_id
        self.model = None
        self.temperature = 1.0
        self.model_description = f"{description}, fine-tuned ({device})"
        self.calibration = "softmax(logits / T), T fitted on a stratified 10% validation part of the train split"

    def _new_model(self, source: Path):
        return AutoModelForSequenceClassification.from_pretrained(
            source, num_labels=len(self.labels), id2label=dict(enumerate(self.labels)),
            label2id=self.index, dtype=torch.float32)

    def fit_cached(self, train: pd.DataFrame, models_dir: Path, log=print) -> dict:
        texts, labels = train["text"].tolist(), train["label"].tolist()
        ident = {"app": self.app, "base": self.base_ref, "config": asdict(self.cfg), "labels": self.labels,
                 "train_sha1": data_sha1(texts, labels), "torch": torch.__version__,
                 "transformers": transformers.__version__, "finetune": VERSION}
        digest = hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
        out = models_dir / digest
        if (out / "meta.json").exists() and (out / "model.safetensors").exists():
            start = time.perf_counter()
            meta = json.loads((out / "meta.json").read_text())
            self.model = self._new_model(out).to(self.device).eval()
            self.temperature = meta["temperature"]
            log(f"Loaded the fine-tuned model {out.name}")
            return {**self._summary(meta), "train_cached": True, "_load_seconds": time.perf_counter() - start}

        start = time.perf_counter()
        fit, val = split_train_val(train, self.cfg.val_frac, self.cfg.seed)
        if self.cfg.train_limit and self.cfg.train_limit < len(fit):  # a stratified pilot subset of the fit part
            fit = (fit.groupby("label", group_keys=False)
                   .sample(frac=self.cfg.train_limit / len(fit), random_state=self.cfg.seed).reset_index(drop=True))
        fit_x, val_x = tokenize_all(self.tok, fit["text"], self.cfg.max_len), \
            tokenize_all(self.tok, val["text"], self.cfg.max_len)
        fit_y, val_y = [self.index[l] for l in fit["label"]], [self.index[l] for l in val["label"]]
        torch.manual_seed(self.cfg.seed)
        self.model = self._new_model(self.base_dir)          # the head is new and random; the encoder pre-trained
        log(f"Fine-tuning {self.app}: {len(fit)} texts (+{len(val)} validation), "
            f"{sum(p.numel() for p in self.model.parameters()):,} parameters, device {self.device}, "
            f"{'bfloat16' if self.cfg.bf16 else 'float32'} training")
        best_epoch, history = train_loop(self.model, fit_x, fit_y, val_x, val_y, self.cfg, self.pad_id,
                                         self.device, log)
        self.temperature = round(fit_temperature(logits_for(self.model, val_x, 64, self.pad_id, self.device),
                                                 torch.tensor(val_y)), 5)
        seconds = time.perf_counter() - start
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out, safe_serialization=True)
        meta = {**ident, "n_fit": len(fit), "n_val": len(val), "best_epoch": best_epoch, "history": history,
                "temperature": self.temperature, "train_seconds": round(seconds, 1), "device": self.device}
        (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        self.model.eval()
        log(f"Fine-tuned in {seconds / 60:.1f} min; best epoch {best_epoch}; T = {self.temperature:.3f}; saved {out}")
        return {**self._summary(meta), "train_cached": False, "_load_seconds": 0.0}

    def _summary(self, meta: dict) -> dict:
        keep = ("base", "config", "n_fit", "n_val", "best_epoch", "history", "temperature", "train_seconds",
                "device", "torch", "transformers", "train_sha1")
        return {"supervision": "supervised", "fine_tuned": True, "calibration": self.calibration,
                **{k: meta[k] for k in keep}}

    def predict_many(self, texts: list[str], batch_size: int = 64) -> list[Prediction]:
        seqs = tokenize_all(self.tok, texts, self.cfg.max_len)
        probs = torch.softmax(logits_for(self.model, seqs, batch_size, self.pad_id, self.device) / self.temperature, 1)
        preds = []
        for row in probs.tolist():
            p = {label: row[i] for i, label in enumerate(self.labels)}
            label = max(p, key=p.get)
            preds.append(Prediction(label, p[label], p))
        return preds

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]


def add_finetune_arguments(parser) -> None:
    parser.add_argument("--epochs", type=int, help="maximum epochs (default: the task's recipe)")
    parser.add_argument("--lr", type=float, help="peak learning rate")
    parser.add_argument("--train-batch-size", type=int, help="training batch size")
    parser.add_argument("--train-seed", type=int, help="seed for the validation split, the head and the order")
    parser.add_argument("--train-limit", type=int, help="fine-tune on a stratified subset of this many texts")
    parser.add_argument("--fp32", action="store_true", help="train in float32 instead of bfloat16 autocast")
    parser.add_argument("--batch-size", type=int, default=64, help="texts per forward pass when classifying")
    parser.add_argument("--device", help="mps | cuda | cpu (default: the app's *_DEVICE setting, else mps when available)")


def config_from_args(cfg: FTConfig, args) -> FTConfig:
    overrides = {"max_epochs": args.epochs, "lr": args.lr, "batch_size": args.train_batch_size,
                 "seed": args.train_seed, "train_limit": args.train_limit, "bf16": False if args.fp32 else None}
    return replace(cfg, **{k: v for k, v in overrides.items() if v is not None})

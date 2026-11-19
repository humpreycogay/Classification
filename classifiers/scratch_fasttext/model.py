"""fastText-style classifier: the mean of word and hashed-bigram embeddings, then one linear layer.

After Joulin et al. (2016), "Bag of Tricks for Efficient Text Classification". This is a reimplementation of the
idea in PyTorch; it does not use the fastText library. Every weight starts random (trained from scratch).
"""

import torch
from torch import nn


class FastText(nn.Module):
    """Average the vectors of the text's words and word pairs; a linear head scores the labels."""

    def __init__(self, rows: int, n_labels: int, dim: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_labels = n_labels
        self.emb = nn.Embedding(rows, dim, padding_idx=0)   # words + hashed bigram buckets
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dim, n_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # x: (batch, length) ids, 0 = padding
        mask = (x != 0).unsqueeze(-1).float()
        mean = (self.emb(x) * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.out(self.drop(mean))                   # (batch, labels) logits

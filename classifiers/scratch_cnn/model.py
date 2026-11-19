"""Kim (2014)-style CNN: convolution filters over word windows, max-pooled over the text.

After Kim (2014), "Convolutional Neural Networks for Sentence Classification". Every weight starts random (trained
from scratch); there are no pre-trained word vectors.
"""

import torch
import torch.nn.functional as F
from torch import nn


class TextCNN(nn.Module):
    """Each filter learns a phrase pattern; max-pooling keeps its strongest match anywhere in the text."""

    def __init__(self, rows: int, n_labels: int, dim: int = 128, filters: int = 100,
                 widths: tuple = (3, 4, 5), dropout: float = 0.5) -> None:
        super().__init__()
        self.n_labels, self.min_len = n_labels, max(widths)
        self.emb = nn.Embedding(rows, dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(dim, filters, w) for w in widths)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(filters * len(widths), n_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # x: (batch, length) ids, 0 = padding
        if x.size(1) < self.min_len:                       # a text shorter than the widest window
            x = F.pad(x, (0, self.min_len - x.size(1)))
        lengths = (x != 0).sum(1, keepdim=True)            # (batch, 1) real tokens; padding sits only at the end
        e = self.emb(x).transpose(1, 2)                    # (batch, dim, length): Conv1d wants channels first
        pooled = []
        for conv, w in zip(self.convs, (c.kernel_size[0] for c in self.convs)):
            h = F.relu(conv(e))                            # (batch, filters, windows)
            # Keep only the windows the text would have on its own: those that end inside the text (window 0 is
            # always kept, for texts shorter than the window). Windows that run into the batch's padding exist only
            # because of the batch, so without this mask a text's answer would depend on its batch-mates.
            start = torch.arange(h.size(2), device=x.device).unsqueeze(0)          # (1, windows)
            outside = start > (lengths - w).clamp(min=0)                          # (batch, windows)
            h = h.masked_fill(outside.unsqueeze(1), float("-inf"))
            pooled.append(h.max(dim=2).values)             # strongest match per filter
        return self.out(self.drop(torch.cat(pooled, dim=1)))

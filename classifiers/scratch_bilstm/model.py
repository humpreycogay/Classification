"""Bidirectional LSTM: reads the text forwards and backwards; final states plus max-pooling feed the head.

LSTM: Hochreiter & Schmidhuber (1997); bidirectional reading as in Graves & Schmidhuber (2005). Every weight starts
random (trained from scratch); there are no pre-trained word vectors.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTM(nn.Module):
    """A reader with memory, in both directions."""

    def __init__(self, rows: int, n_labels: int, dim: int = 128, hidden: int = 128, layers: int = 1,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.n_labels = n_labels
        self.emb = nn.Embedding(rows, dim, padding_idx=0)
        self.lstm = nn.LSTM(dim, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(4 * hidden, n_labels)   # [last forward, last backward, max-pool fwd, max-pool bwd]

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # x: (batch, length) ids, 0 = padding
        lengths = (x != 0).sum(1).clamp(min=1).cpu()           # pack_padded_sequence wants lengths on the CPU
        packed = pack_padded_sequence(self.drop(self.emb(x)), lengths, batch_first=True, enforce_sorted=False)
        out, (h, _) = self.lstm(packed)                         # padding is skipped, never read
        out, _ = pad_packed_sequence(out, batch_first=True, padding_value=float("-inf"))
        pooled = out.max(dim=1).values                          # (batch, 2*hidden)
        last = torch.cat([h[-2], h[-1]], dim=1)                 # final forward and backward states
        return self.out(self.drop(torch.cat([last, pooled], dim=1)))

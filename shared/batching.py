"""Length-sorted batching for encoder apps: less padding per batch, same outputs in the original order.

Texts are sorted by length (longest first, so a too-large batch fails at once rather than at the end), cut
into batches, classified, and the results are put back in input order. Each text is scored on its own
(attention masks hide the padding), so only padding changes, not which label wins.

Part of the shared benchmark library (shared/).
"""

from __future__ import annotations

from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def length_order(items: Sequence, key: Callable[[object], int] = len) -> list[int]:
    """Indices of items, longest first (ties keep input order)."""
    return sorted(range(len(items)), key=lambda i: -key(items[i]))


def restore(order: list[int], results: Sequence[R]) -> list[R]:
    """Results computed in `order` put back in input order."""
    out: list = [None] * len(order)
    for position, i in enumerate(order):
        out[i] = results[position]
    return out


def sorted_batches(items: Sequence[T], batch_size: int, run: Callable[[list[T]], list[R]],
                   key: Callable[[T], int] = len) -> list[R]:
    """run(batch) over length-sorted batches of items; results in input order."""
    order = length_order(items, key)
    ordered = [items[i] for i in order]
    results: list[R] = []
    for start in range(0, len(ordered), max(1, batch_size)):
        results += run(ordered[start:start + batch_size])
    return restore(order, results)

"""Shared-prompt-start KV cache for MLX decision apps (Nimble; later Plumb): compute the prompt start every
text shares once, and reuse it, but only where that pays and gives the same logits.

Every prompt of one task starts with the same tokens (system prompt, schema, chat header) before the text.
build() finds that common token prefix on a few probe prompts, stopping one token early (the last shared token
may merge with a text's first token under BPE), and runs it through the backbone once. It is used only when it
is at least `min_share` of the shortest probe prompt (below that, forking the cache costs more than it saves) and
when, on every probe, the cached path reproduces the plain logits (atol, default 1e-3); otherwise build()
returns None and the app scores every prompt in full.

Part of the shared benchmark library (shared/), but imported only by MLX apps: other apps' .venvs have no mlx.
"""

from __future__ import annotations

from typing import Callable, Sequence

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache


def common_prefix(prompts: Sequence[Sequence[int]]) -> list[int]:
    """The token prefix every prompt starts with, less its last token (BPE merges), never a whole prompt."""
    first = prompts[0]
    n = 0
    while all(len(p) > n + 1 and p[n] == first[n] for p in prompts):
        n += 1
    return list(first[:max(0, n - 1)])


def fork_cache(prefix_cache: list) -> list:
    """A copy of a completed prefix cache (batch 1) that the caller may extend; the prefix is never modified.
    Handles attention (KVCache) and linear-attention / SSM states (ArraysCache), as in Qwen3.5."""
    result = []
    for source in prefix_cache:
        if isinstance(source, KVCache):
            target = KVCache()
            target.state = tuple(mx.broadcast_to(x, (1,) + x.shape[1:]) for x in source.state)
        elif isinstance(source, ArraysCache):
            if source.left_padding is not None or source.lengths is not None:
                raise ValueError("Only unpadded, completed prefix states can be forked.")
            target = ArraysCache(len(source.state))
            target.state = [None if x is None else mx.broadcast_to(x, (1,) + x.shape[1:]) for x in source.state]
        else:
            raise TypeError(f"Unsupported cache: {type(source).__name__}")
        result.append(target)
    return result


class SharedPrefix:
    """A computed prompt start. hidden(ids) runs only the rest of a prompt that starts with it."""

    def __init__(self, backbone: Callable, ids: list[int], cache: list) -> None:
        self.backbone, self.ids, self.cache = backbone, ids, cache

    def __len__(self) -> int:
        return len(self.ids)

    def matches(self, ids: Sequence[int]) -> bool:
        return list(ids[:len(self.ids)]) == self.ids

    def hidden(self, ids: Sequence[int]) -> mx.array:
        """Last-position hidden state of the full prompt `ids` (which must start with the prefix)."""
        cache = fork_cache(self.cache)
        return self.backbone(mx.array([list(ids[len(self.ids):])]), cache=cache)[:, -1, :]


def build(backbone: Callable, make_cache: Callable[[], list], probe_prompts: Sequence[Sequence[int]],
          plain_logits: Callable[[int], mx.array], head: Callable[[mx.array, int], mx.array],
          min_share: float, atol: float = 1e-3) -> SharedPrefix | None:
    """The shared prompt start of probe_prompts, computed once, or None when it is below min_share of the
    shortest prompt or its cached path does not reproduce the plain logits on every probe.

    plain_logits(i): the app's own logits for probe i (full prompt, no prefix cache).
    head(hidden, i): the same logits from the last hidden state, as the app computes them for probe i.
    """
    ids = common_prefix(probe_prompts)
    if not ids or len(ids) < min_share * min(map(len, probe_prompts)):
        return None
    cache = make_cache()
    backbone(mx.array([ids]), cache=cache)
    mx.eval([c.state for c in cache])
    prefix = SharedPrefix(backbone, ids, cache)
    for i, prompt in enumerate(probe_prompts):
        plain = plain_logits(i)
        cached = head(prefix.hidden(prompt), i)
        mx.eval(plain, cached)
        if not mx.allclose(plain, cached, rtol=0, atol=atol).item():
            return None
    return prefix

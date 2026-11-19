"""Speed, memory, token and cost figures read from one app's run metadata (for the report).

All values come from what the app (timing.py) and the runner (run_all.py) recorded; missing
measurements are None, never guessed. Costs follow pricing.json:

    measured   the API reported the billed amount (none do yet)
    estimated  tokens or requests x a configured price
    unknown    a paid app without a configured price
    none       local or own-server apps: no per-request charge (compute time is reported instead)
"""

from __future__ import annotations

import json
from pathlib import Path

PRICING_PATH = Path(__file__).resolve().parent / "pricing.json"


def load_pricing(path: Path = PRICING_PATH) -> dict:
    if not path.exists():
        return {}
    return {k: v for k, v in json.loads(path.read_text()).items() if not k.startswith("_")}


def config_text(meta: dict) -> str:
    """How the app grouped work in the main run, e.g. 'batch_size 16' or 'concurrency 4'."""
    timing = meta.get("timing") or {}
    if timing.get("axis"):
        return f"{timing['axis'].replace('_', ' ')} {timing['value']}"
    args = meta.get("args") or {}
    for key in ("batch_size", "concurrency"):
        if args.get(key) is not None:
            return f"{key.replace('_', ' ')} {args[key]}"
    return (timing.get("per_call") or {}).get("unit") or "–"


def interactive(meta: dict) -> dict:
    return ((meta.get("timing") or {}).get("interactive")) or {}


def per_call(meta: dict) -> dict:
    return ((meta.get("timing") or {}).get("per_call")) or {}


def memory(meta: dict) -> dict:
    """Peak memory of the app process, in-process GPU memory, server-side model memory and model size."""
    runner = meta.get("runner") or {}
    timing = meta.get("timing") or {}
    server = runner.get("server_memory") or {}
    server_bytes = server.get("loaded_bytes") if server.get("kind") == "ollama" else server.get("peak_rss_bytes")
    footprint = runner.get("peak_footprint_bytes")
    return {
        "footprint": footprint,
        "rss": runner.get("peak_rss_bytes"),
        "gpu": (timing.get("gpu_memory") or {}).get("peak_bytes"),
        "gpu_method": (timing.get("gpu_memory") or {}).get("method"),
        "server": server_bytes,
        "server_gpu": server.get("gpu_bytes"),
        "server_method": server.get("method"),
        "model_disk": timing.get("model_bytes") or runner.get("model_disk_bytes"),
        # Client + server: what the machine needs to run this app (None if the app process was not measured).
        "total": (footprint + (server_bytes or 0)) if footprint else None,
    }


def cost(app_name: str, app_cost: str, meta: dict, pricing: dict) -> dict:
    """Cost per item and per 1,000 items, with its basis (measured / estimated / unknown / none)."""
    if app_cost != "paid":
        return {"basis": "none", "per_item": 0.0, "per_1000": 0.0, "currency": None,
                "note": "local or own server: no per-request charge"}
    price = pricing.get(app_name) or {}
    requests = meta.get("requests") or {}
    n = meta.get("n") or 0
    if requests.get("billed") is not None and n:
        per_item = requests["billed"] / n
        return {"basis": "measured", "per_item": per_item, "per_1000": per_item * 1000,
                "currency": price.get("currency"), "note": "billed amount reported by the API"}
    per_item = None
    if price.get("per_request") is not None and requests.get("requests") and n:
        per_item = price["per_request"] * requests["requests"] / n
    elif (price.get("per_1k_input_tokens") is not None and price.get("per_1k_output_tokens") is not None
          and requests.get("input_tokens") is not None and n):
        per_item = (requests["input_tokens"] * price["per_1k_input_tokens"]
                    + (requests.get("output_tokens") or 0) * price["per_1k_output_tokens"]) / 1000 / n
    if per_item is None:
        return {"basis": "unknown", "per_item": None, "per_1000": None, "currency": price.get("currency"),
                "note": "paid API without a configured price (runner/pricing.json)"}
    return {"basis": "estimated", "per_item": per_item, "per_1000": per_item * 1000, "currency": price.get("currency"),
            "note": f"tokens or requests x configured price ({price.get('source') or 'source not given'})"}


def sweep_points(meta: dict) -> list[dict]:
    """Sweep rows with a latency per call: batch p50, or concurrency / throughput (Little's law) for HTTP apps."""
    rows = ((meta.get("timing") or {}).get("sweep")) or []
    out = []
    for r in rows:
        if r.get("per_call") and r["per_call"].get("p50") is not None:
            latency, how = r["per_call"]["p50"], "batch p50"
        else:
            latency, how = r["value"] / r["items_per_second"], "concurrency / throughput"
        # Sweeps recorded before 2026-09-27 compared with the main run instead of the first setting.
        agreement = r.get("agreement_with_first", r.get("agreement_with_main_run"))
        out.append({**r, "latency": latency, "latency_method": how, "agreement_with_first": agreement})
    return out


def on_battery(meta: dict) -> bool | None:
    cond = (meta.get("runner") or {}).get("conditions")
    return None if not cond else bool(cond.get("on_battery"))

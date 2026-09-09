"""
aggregation/adaptive.py
All aggregation strategies — fully implemented:
  - FedAvg   (baseline)
  - FedProx  (benchmark 1)
  - SCAFFOLD (benchmark 2) — now fully implemented
  - Adaptive (proposed — gradient cosine similarity)
"""

import numpy as np
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# FedAvg
# ─────────────────────────────────────────────────────────────────────────────
def fedavg_aggregate(client_weights, client_sizes):
    total = sum(client_sizes)
    return sum((n / total) * w for w, n in zip(client_weights, client_sizes))


# ─────────────────────────────────────────────────────────────────────────────
# FedProx  (server side = FedAvg; proximal term enforced client-side)
# ─────────────────────────────────────────────────────────────────────────────
def fedprox_aggregate(client_weights, client_sizes):
    return fedavg_aggregate(client_weights, client_sizes)


# ─────────────────────────────────────────────────────────────────────────────
# SCAFFOLD — stateless wrapper with module-level singleton state
# ─────────────────────────────────────────────────────────────────────────────
_scaffold_c_global = {}     # model_dim -> np.ndarray
_scaffold_c_local  = {}     # client_id -> np.ndarray

def reset_scaffold_state():
    global _scaffold_c_global, _scaffold_c_local
    _scaffold_c_global = {}
    _scaffold_c_local  = {}

def scaffold_aggregate(global_weights, client_weights, client_sizes,
                       local_lr=0.001, local_steps=5, server_lr=1.0):
    global _scaffold_c_global, _scaffold_c_local
    model_dim = len(global_weights)
    N = len(client_weights)

    # Initialise global control variate
    key = model_dim
    if key not in _scaffold_c_global:
        _scaffold_c_global[key] = np.zeros(model_dim)
    c_global = _scaffold_c_global[key]

    c_deltas = []
    for i, local_w in enumerate(client_weights):
        if i not in _scaffold_c_local or len(_scaffold_c_local[i]) != model_dim:
            _scaffold_c_local[i] = np.zeros(model_dim)
        c_local = _scaffold_c_local[i]

        # Option II: c_new = (w_global - w_local) / (K * lr)
        denom = max(local_steps * local_lr, 1e-8)
        c_new = (global_weights - local_w) / denom
        c_delta = c_new - c_local
        _scaffold_c_local[i] = c_new.copy()
        c_deltas.append(c_delta)

    # Server weight update
    avg_delta_w = np.mean([w - global_weights for w in client_weights], axis=0)
    new_global  = global_weights + server_lr * avg_delta_w

    # Update global control variate
    avg_delta_c = np.mean(c_deltas, axis=0)
    _scaffold_c_global[key] = c_global + avg_delta_c

    return new_global


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive — proposed gradient cosine similarity method
# ─────────────────────────────────────────────────────────────────────────────
def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def adaptive_aggregate(global_weights, client_weights, client_sizes,
                       temperature=1.0, fallback_to_fedavg=True):
    total    = sum(client_sizes)
    n        = len(client_weights)
    deltas   = [w - global_weights for w in client_weights]
    g_delta  = sum((sz / total) * d for d, sz in zip(deltas, client_sizes))
    sims     = np.array([cosine_similarity(d, g_delta) for d in deltas])

    if np.all(np.abs(sims) < 1e-8):
        if fallback_to_fedavg:
            alpha = np.array(client_sizes, dtype=float) / total
            return sum(a * w for a, w in zip(alpha, client_weights)), alpha
        sims = np.ones(n)

    shifted = (sims - sims.max()) / temperature
    alpha   = np.exp(shifted) / np.exp(shifted).sum()
    return sum(a * w for a, w in zip(alpha, client_weights)), alpha


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
def get_aggregator(strategy: str, **kwargs):
    s = strategy.lower().strip()
    if s == "fedavg":
        return lambda gw, cws, sizes: (fedavg_aggregate(cws, sizes), None)
    elif s == "fedprox":
        return lambda gw, cws, sizes: (fedprox_aggregate(cws, sizes), None)
    elif s == "scaffold":
        lr    = kwargs.get("local_lr",    0.001)
        steps = kwargs.get("local_steps", 5)
        return lambda gw, cws, sizes: (scaffold_aggregate(gw, cws, sizes, lr, steps), None)
    elif s == "adaptive":
        tau = kwargs.get("temperature", 1.0)
        return lambda gw, cws, sizes: adaptive_aggregate(gw, cws, sizes, tau)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use: fedavg|fedprox|scaffold|adaptive")

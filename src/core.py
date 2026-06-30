from __future__ import annotations
"""CTR-prediction metrics for AdCTR (InMobi), implemented from scratch.

These are the standard online-ad metrics, not generic classification metrics:
  * **Normalized Entropy (NE)** — log loss divided by the background (base-rate)
    entropy, so it is scale-free and in ~[0, 1]; lower is better (He et al. 2014).
  * **AUC** — rank-based Mann-Whitney statistic with tie handling.
  * **Calibration curve** — predicted vs. observed CTR per score bin.
  * **lift@k** — CTR among the top-k% scored impressions relative to baseline.
"""
import numpy as np


def log_loss(y, p, eps: float = 1e-15) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _background_entropy(y) -> float:
    pbar = float(np.clip(np.asarray(y, dtype=float).mean(), 1e-7, 1 - 1e-7))
    return -(pbar * np.log(pbar) + (1 - pbar) * np.log(1 - pbar))


def normalized_entropy(y, p) -> float:
    """NE = log_loss / background_entropy in ~[0,1] (lower is better)."""
    bg = _background_entropy(y)
    if bg <= 0:
        return 0.0
    return log_loss(y, p) / bg


def _rankdata(a) -> np.ndarray:
    """Average ranks (1-based) with tie handling, from scratch."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    a_sorted = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a_sorted[j + 1] == a_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def auc(y, p) -> float:
    """Mann-Whitney AUC = (sum_ranks_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg)."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _rankdata(p)
    s = ranks[y == 1].sum()
    return float((s - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def calibration_curve(y, p, n_bins: int = 10):
    """Quantile-binned predicted vs. observed CTR. Returns (pred, obs, count)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    idx = np.clip(np.digitize(p, qs[1:-1]), 0, n_bins - 1)
    pred, obs, cnt = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        pred.append(float(p[m].mean()))
        obs.append(float(y[m].mean()))
        cnt.append(int(m.sum()))
    return np.array(pred), np.array(obs), np.array(cnt)


def lift_at_k(y, p, k: float = 0.1) -> float:
    """CTR in the top-k% scored impressions divided by the baseline CTR (>=1 is skilful)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = max(1, int(round(k * len(y))))
    order = np.argsort(-p, kind="mergesort")
    overall = y.mean()
    if overall <= 0:
        return 1.0
    return float(y[order[:m]].mean() / overall)

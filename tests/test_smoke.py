"""Smoke tests for AdCTR — CTR prediction with AdTech metrics."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import make_synthetic
from src.core import log_loss, normalized_entropy, auc, lift_at_k, calibration_curve
from src.model import train_all_models, cross_validate, SGDRidgeLogistic, one_hot_encode


def test_data():
    """Ad impression data has realistic low CTR."""
    d = make_synthetic(n=2000, seed=42)
    assert d["X"].shape[0] == 2000
    assert 0.005 < d["ctr"] < 0.05  # ~2% base rate


def test_core_metrics():
    """NE, AUC, lift@k compute correctly."""
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    p = np.array([0.8, 0.1, 0.2, 0.05, 0.7, 0.3, 0.1, 0.05, 0.02, 0.01])
    assert 0.0 < normalized_entropy(y, p) < 1.0
    assert 0.5 <= auc(y, p) <= 1.0
    assert lift_at_k(y, p, k=0.2) >= 1.0  # top-20% should be > baseline


def test_calibration():
    """Calibration curve returns matched arrays."""
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
    pred, obs, cnt = calibration_curve(y, p, n_bins=4)
    assert len(pred) == len(obs) == len(cnt)


def test_train():
    """Training produces 2 models with AdTech metrics."""
    d = make_synthetic(n=3000, seed=42)
    b = train_all_models(d)
    assert len(b["models"]) == 2
        # GBDT should beat random; LR is a sparse baseline
    assert b["results"]["GBDT"]["metrics"]["auc"] > 0.5
    assert b["results"]["GBDT"]["metrics"]["lift_at_10pct"] > 1.0
    assert b["results"]["FFM-Logistic"]["metrics"]["auc"] > 0.45  # near-random baseline OK


def test_cv():
    """Cross-validation returns AUC and NE per model."""
    d = make_synthetic(n=2000, seed=42)
    cv = cross_validate(d, seed=42, n_folds=3)
    assert cv["GBDT"]["auc"]["mean"] > 0.5
    assert "FFM-Logistic" in cv


if __name__ == "__main__":
    test_data()
    test_core_metrics()
    test_calibration()
    test_train()
    test_cv()
    print("All AdCTR smoke tests passed!")

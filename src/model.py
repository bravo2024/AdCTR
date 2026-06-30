"""model.py — CTR prediction models for AdCTR (InMobi).

Implements two approaches used in real online-ad pipelines:

1. **Logistic Regression with SGD** on one-hot sparse features — mirrors
   the production standard (McMahan et al. 2013, He et al. 2014).
2. **GBDT** — captures non-linear interactions automatically; the strong
   baseline from He et al. (2014).

Categoricals are one-hot encoded (fit on train only), and evaluation
uses AdTech metrics (NE, AUC, lift@k) from ``src.core``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.core import log_loss, normalized_entropy, auc, lift_at_k


def one_hot_encode(X_train, X_test, cat_cols, num_cols):
    """One-hot encode categoricals (fit on train only), pass through numericals."""
    parts_tr, parts_te, names = [], [], []
    for c in cat_cols:
        cats = X_train[c].astype(str).unique().tolist()
        for cat in cats:
            names.append(f"{c}={cat}")
            parts_tr.append((X_train[c].astype(str) == cat).values.astype(float))
            parts_te.append((X_test[c].astype(str) == cat).values.astype(float))
    for c in num_cols:
        names.append(c)
        parts_tr.append(X_train[c].values.astype(float))
        parts_te.append(X_test[c].values.astype(float))
    return np.column_stack(parts_tr), np.column_stack(parts_te), names


class SGDRidgeLogistic:
    """Logistic regression via mini-batch SGD with L2 (He et al. 2014 style)."""

    def __init__(self, lr=0.1, epochs=25, batch_size=256, l2=1e-4, seed=42):
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.l2, self.seed = l2, seed

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        self.w_ = rng.normal(0, 0.01, d); self.b_ = 0.0
        self.history_ = []
        for ep in range(self.epochs):
            perm = rng.permutation(n)
            lr_t = self.lr / (1.0 + 0.1 * ep)
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                xb, yb = X[idx], y[idx]
                p = 1.0 / (1.0 + np.exp(-np.clip(xb @ self.w_ + self.b_, -35, 35)))
                err = p - yb
                self.w_ -= lr_t * (xb.T @ err / len(idx) + self.l2 * self.w_)
                self.b_ -= lr_t * err.mean()
            full_p = 1.0 / (1.0 + np.exp(-np.clip(X @ self.w_ + self.b_, -35, 35)))
            self.history_.append(log_loss(y, full_p))
        return self

    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(X, float) @ self.w_ + self.b_, -35, 35)))


def _evaluate(y_true, y_proba):
    """Compute the AdTech metric bundle."""
    return {
        "log_loss": log_loss(y_true, y_proba),
        "normalized_entropy": normalized_entropy(y_true, y_proba),
        "auc": auc(y_true, y_proba),
        "lift_at_10pct": lift_at_k(y_true, y_proba, k=0.10),
        "lift_at_5pct": lift_at_k(y_true, y_proba, k=0.05),
    }


def train_all_models(data, seed=42, test_size=0.25):
    """Train FFM-LR and GBDT on the same split; evaluate with AdTech metrics."""
    X = data["X"].copy()
    y = data["y"].values if hasattr(data["y"], "values") else np.asarray(data["y"])
    cat_cols = data.get("categorical_features", [])
    num_cols = data.get("numerical_features", [])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    Xtr_enc, Xte_enc, feat_names = one_hot_encode(X_train, X_test, cat_cols, num_cols)

    results = {}

    # --- FFM-LR (from scratch) ---
    lr_model = SGDRidgeLogistic(lr=0.5, epochs=40, batch_size=256, l2=1e-5, seed=seed)
    lr_model.fit(Xtr_enc, y_train)
    lr_proba = lr_model.predict_proba(Xte_enc)
    results["FFM-Logistic"] = {"metrics": _evaluate(y_test, lr_proba), "y_proba": lr_proba}

    # --- GBDT (lightgbm) ---
    import lightgbm as lgb
    gbm = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(1 - y_train.mean()) / max(y_train.mean(), 1e-6),
        random_state=seed, verbose=-1,
    )
    gbm.fit(Xtr_enc, y_train)
    gbm_proba = gbm.predict_proba(Xte_enc)[:, 1]
    results["GBDT"] = {"metrics": _evaluate(y_test, gbm_proba), "y_proba": gbm_proba}

    return {
        "models": {"FFM-Logistic": lr_model, "GBDT": gbm},
        "results": results, "feature_names": feat_names,
        "X_train_enc": Xtr_enc, "X_test_enc": Xte_enc,
        "y_train": y_train, "y_test": y_test,
        "n_train": len(y_train), "n_test": len(y_test),
    }


def cross_validate(data, seed=42, n_folds=3):
    """Stratified K-fold CV with NE and AUC per fold."""
    from sklearn.model_selection import StratifiedKFold
    X = data["X"].copy()
    y = data["y"].values if hasattr(data["y"], "values") else np.asarray(data["y"])
    cat_cols = data.get("categorical_features", [])
    num_cols = data.get("numerical_features", [])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv = {n: {"auc": [], "ne": []} for n in ["FFM-Logistic", "GBDT"]}
    import lightgbm as lgb
    for tr_idx, te_idx in skf.split(X, y):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        Xtr_e, Xte_e, _ = one_hot_encode(Xtr, Xte, cat_cols, num_cols)
        m = SGDRidgeLogistic(lr=0.5, epochs=30, batch_size=256, l2=1e-5, seed=seed)
        m.fit(Xtr_e, ytr)
        p = m.predict_proba(Xte_e)
        cv["FFM-Logistic"]["auc"].append(auc(yte, p))
        cv["FFM-Logistic"]["ne"].append(normalized_entropy(yte, p))
        g = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.05,
                               subsample=0.8, random_state=seed, verbose=-1)
        g.fit(Xtr_e, ytr)
        pg = g.predict_proba(Xte_e)[:, 1]
        cv["GBDT"]["auc"].append(auc(yte, pg))
        cv["GBDT"]["ne"].append(normalized_entropy(yte, pg))
    return {
        name: {
            "auc": {"mean": float(np.mean(s["auc"])), "std": float(np.std(s["auc"]))},
            "ne": {"mean": float(np.mean(s["ne"])), "std": float(np.std(s["ne"]))},
        }
        for name, s in cv.items()
    }

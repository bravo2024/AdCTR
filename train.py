"""CLI training entrypoint for AdCTR.

Generates the synthetic impression log, trains both models (FFM-style
logistic regression and GBDT), runs 3-fold CV, and persists the GBDT
plus the full metric bundle to models/.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.data import make_synthetic
from src.model import train_all_models, cross_validate
from src.persist import save_model


def main():
    data = make_synthetic()
    ctr = float(data["y"].mean())
    print(f"impressions: {len(data['y']):,}  |  base CTR: {ctr:.2%}")

    out = train_all_models(data)
    for name, res in out["results"].items():
        m = res["metrics"]
        print(
            f"{name:14s} AUC {m['auc']:.4f}  NE {m['normalized_entropy']:.4f}  "
            f"lift@5% {m['lift_at_5pct']:.2f}x  lift@10% {m['lift_at_10pct']:.2f}x"
        )

    cv = cross_validate(data)
    for name, s in cv.items():
        print(f"{name:14s} CV AUC {s['auc']['mean']:.4f} ± {s['auc']['std']:.4f}")

    Path("models").mkdir(exist_ok=True)
    save_model(out["models"]["GBDT"])
    metrics = {
        "n_train": out["n_train"],
        "n_test": out["n_test"],
        "base_ctr": ctr,
        "holdout": {n: r["metrics"] for n, r in out["results"].items()},
        "cv": cv,
    }
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("saved GBDT -> models/model.pkl, metrics -> models/metrics.json")


if __name__ == "__main__":
    main()

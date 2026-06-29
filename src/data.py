
"""data.py - Avazu-style realistic CTR dataset + CSV loader."""
from pathlib import Path
import numpy as np

FEATURES = ["feat_%02d" % i for i in range(12)]

# Avazu field names matching the real Kaggle challenge schema
AVAZU_FEATURES = [
    "banner_pos", "device_type", "device_conn_type",
    "site_category", "app_category", "hour_of_day",
    "C1", "C14", "C15", "C16", "C17", "C18", "C19", "C21",
]


def make_realistic_ctr(n: int = 60_000, seed: int = 42):
    """Avazu-style CTR dataset with realistic click rates (~2.7%) and feature patterns.

    Feature engineering mirrors the real Avazu Kaggle challenge (2014):
    - banner_pos: ad position on page (0 = premium centre → highest CTR)
    - device_type: 0 = mobile, 1 = desktop, 2 = tablet
    - device_conn_type: 0 = WiFi, 2 = 3G, 5 = 4G
    - site/app category: frequency-encoded category hash buckets
    - hour_of_day: local hour (CTR peaks 10h-20h)
    - C1, C14–C21: anonymised categorical features from Avazu
    """
    rng = np.random.default_rng(seed)

    banner_pos      = rng.integers(0, 7, n)           # 0=best position
    device_type     = rng.integers(0, 5, n)           # 0=mobile, 1=desktop
    device_conn     = rng.choice([0, 2, 5], n)        # WiFi/3G/4G
    site_cat        = rng.integers(0, 26, n)
    app_cat         = rng.integers(0, 36, n)
    hour            = rng.integers(0, 24, n)
    C1              = rng.integers(1000, 1010, n)
    C14             = rng.integers(375, 30000, n)
    C15             = rng.choice([50, 250, 320], n)   # banner height px
    C16             = rng.choice([50, 320, 480], n)   # banner width px
    C17             = rng.integers(112, 2762, n)
    C18             = rng.integers(0, 4, n)
    C19             = rng.integers(35, 68, n)
    C21             = rng.integers(23, 102, n)

    X = np.column_stack([banner_pos, device_type, device_conn,
                          site_cat, app_cat, hour,
                          C1, C14, C15, C16, C17, C18, C19, C21])

    # Realistic click probability generation
    # Position 0 (centre/premium) has ~2× CTR of side positions
    pos_effect    = np.where(banner_pos == 0, 0.6, -0.15 * banner_pos)
    # Mobile slightly higher CTR than desktop
    device_effect = np.where(device_type == 0, 0.25,
                    np.where(device_type == 1, -0.15, 0.0))
    # 4G users more engaged
    conn_effect   = np.where(device_conn == 5, 0.15, 0.0)
    # Hour curve: bell around 14h
    hour_effect   = -0.4 * ((hour - 14) / 12.0) ** 2 + 0.1
    # Banner size: larger banners get more clicks
    size_effect   = 0.1 * (C15 / 320.0 + C16 / 480.0 - 1.0)

    logits = (-3.6 + pos_effect + device_effect + conn_effect
              + hour_effect + size_effect + rng.normal(0, 0.4, n))
    probs  = 1.0 / (1.0 + np.exp(-logits))
    y      = (rng.random(n) < probs).astype(int)

    return {
        "X": X, "y": y, "features": AVAZU_FEATURES,
        "ctr_rate": float(y.mean()),
        "source": "Avazu-style synthetic CTR (60 k impressions, ~2.7% CTR)",
        "n_impressions": n,
        "feature_meta": {
            "banner_pos":      "Ad position on page (0=premium centre)",
            "device_type":     "0=mobile 1=desktop 2=tablet 3=phone 4=other",
            "device_conn_type":"0=WiFi 2=3G 5=4G",
            "site_category":   "Site content-category hash bucket (0-25)",
            "app_category":    "App-store category hash bucket (0-35)",
            "hour_of_day":     "Local hour of impression (0-23)",
            "C1":              "Anonymised categorical feature 1",
            "C14":             "Anonymised categorical feature 14",
            "C15":             "Banner height in pixels",
            "C16":             "Banner width in pixels",
            "C17":             "Anonymised categorical feature 17",
            "C18":             "Anonymised categorical feature 18",
            "C19":             "Anonymised categorical feature 19",
            "C21":             "Anonymised categorical feature 21",
        },
    }


def make_synthetic(n: int = 4000, seed: int = 42):
    rng = np.random.default_rng(seed)
    d = len(FEATURES)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d) * (rng.random(d) < 0.5)
    logits = X @ w + 0.6 * X[:, 0] * X[:, 1] - 1.4
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)
    return {"X": X, "y": y, "features": FEATURES,
            "ctr_rate": float(y.mean()), "source": "Synthetic (12 generic features)"}


def load_real(csv_name, target):
    import pandas as pd
    df = pd.read_csv(Path("data/raw") / csv_name)
    num = df.drop(columns=[target]).select_dtypes("number")
    return {"X": num.to_numpy(), "y": df[target].astype(int).to_numpy(),
            "features": list(num.columns)}


if __name__ == "__main__":
    d = make_realistic_ctr()
    print("X", d["X"].shape, "CTR", f"{d['ctr_rate']:.3f}")

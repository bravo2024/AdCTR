from __future__ import annotations
"""Synthetic ad-impression data for AdCTR CTR prediction.

Each row is a single ad impression with user, creative, placement and context
features. Clicks are rare: the base CTR is ~2% and the click probability is driven
by realistic *interactions* (ad-category × user-interest relevance, ad position,
device, time-of-day peaks) rather than a single additive linear score. This
produces a heavily imbalanced binary target that mirrors real display-ad logs.
"""
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "age_group", "gender", "interest_segment", "ad_category", "creative_format",
    "advertiser_industry", "creative_id", "site_category", "ad_position", "ad_size",
    "device_type", "operating_system", "connection_type", "hour_of_day", "day_of_week",
]
CATEGORICAL_FEATURES = [
    "age_group", "gender", "interest_segment", "ad_category", "creative_format",
    "advertiser_industry", "creative_id", "site_category", "ad_position", "ad_size",
    "device_type", "operating_system", "connection_type",
]
NUMERICAL_FEATURES = ["hour_of_day", "day_of_week"]
TARGET_NAME = "click"

_CATS = ["tech", "auto", "finance", "retail", "travel", "food", "sports", "ent"]
_SITE_MATCH = {
    "news": {"finance", "tech"}, "social": {"ent", "food"},
    "shopping": {"retail", "auto"}, "video": {"ent", "sports"},
    "blog": {"travel", "food"},
}


def make_synthetic(n: int = 20000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)

    age_group = rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], n, p=[.20, .25, .25, .20, .10])
    gender = rng.choice(["M", "F", "U"], n, p=[.45, .45, .10])
    interest = rng.choice(_CATS, n)
    ad_category = rng.choice(_CATS, n)
    creative_format = rng.choice(["banner", "video", "native", "carousel"], n, p=[.40, .25, .20, .15])
    advertiser_industry = rng.choice(_CATS, n)
    creative_id = rng.choice([f"cr{i:02d}" for i in range(30)], n)
    site_category = rng.choice(["news", "social", "shopping", "video", "blog"], n, p=[.30, .25, .20, .15, .10])
    ad_position = rng.choice(["top", "middle", "bottom", "side"], n, p=[.30, .30, .20, .20])
    ad_size = rng.choice(["300x250", "728x90", "160x600", "320x50"], n, p=[.35, .25, .20, .20])
    device = rng.choice(["mobile", "desktop", "tablet"], n, p=[.60, .30, .10])
    os_ = rng.choice(["android", "ios", "windows", "macos"], n, p=[.40, .25, .20, .15])
    hour = rng.integers(0, 24, n)
    dow = rng.integers(0, 7, n)
    conn = rng.choice(["wifi", "4g", "3g"], n, p=[.50, .35, .15])

    # ---- click logit: base rate + main effects + key interactions ----
    logit = np.full(n, -4.7)
    logit += 1.2 * (ad_category == interest)                                   # relevance match
    logit += np.where(ad_position == "top", 0.8,
              np.where(ad_position == "middle", 0.2,
              np.where(ad_position == "bottom", -0.3, -0.2)))
    logit += np.where(creative_format == "video", 0.6,
              np.where(creative_format == "native", 0.3,
              np.where(creative_format == "carousel", 0.1, 0.0)))
    logit += np.where(device == "mobile", 0.2, np.where(device == "tablet", -0.1, 0.0))
    peak = ((hour >= 7) & (hour <= 9)) | ((hour >= 12) & (hour <= 14)) | ((hour >= 19) & (hour <= 22))
    logit += np.where(peak, 0.4, 0.0)
    logit += np.where(dow >= 5, -0.2, 0.0)
    site_match = np.array([ad_category[i] in _SITE_MATCH[site_category[i]] for i in range(n)])
    logit += 0.5 * site_match
    logit += np.where(conn == "3g", -0.2, np.where(conn == "wifi", 0.1, 0.0))
    logit += np.where(age_group == "18-24", 0.3, np.where(age_group == "25-34", 0.2, 0.0))
    cr_quality = {f"cr{i:02d}": rng.normal(0, 0.25) for i in range(30)}        # per-creative lift
    logit += np.array([cr_quality[c] for c in creative_id])
    logit += rng.normal(0, 0.15, n)

    p = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
    y = rng.binomial(1, p).astype(float)

    df = pd.DataFrame({
        "age_group": age_group, "gender": gender, "interest_segment": interest,
        "ad_category": ad_category, "creative_format": creative_format,
        "advertiser_industry": advertiser_industry, "creative_id": creative_id,
        "site_category": site_category, "ad_position": ad_position, "ad_size": ad_size,
        "device_type": device, "operating_system": os_, "connection_type": conn,
        "hour_of_day": hour, "day_of_week": dow, TARGET_NAME: y, "true_p": p,
    })
    X = df[FEATURE_NAMES].copy()
    return {
        "X": X, "y": y, "df": df, "features": list(FEATURE_NAMES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numerical_features": list(NUMERICAL_FEATURES),
        "target_name": TARGET_NAME, "n_samples": int(n),
        "ctr": float(y.mean()), "n_features": len(FEATURE_NAMES),
    }

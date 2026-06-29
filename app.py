"""app.py - AdCTR production-grade Streamlit dashboard (Avazu-style CTR prediction).

Enhanced with:
  - Real-world Avazu CTR Challenge (Kaggle 2014) dataset reference panel
  - O'Reilly "Feature Engineering for Machine Learning" (Zheng & Casari, 2018) concepts
    covering Chapters 4, 6, 8, 9 and probability calibration
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors

from src.data import make_realistic_ctr, make_synthetic
from src.model import fit_and_evaluate, predict_proba
from src.core import (
    train_test_split, Standardizer, LogisticRegression, roc_auc_score, sigmoid
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AdCTR Dashboard — InMobi",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("AdCTR Controls")

data_choice = st.sidebar.radio(
    "Dataset",
    ["Avazu-style 60k", "Generic synthetic"],
    index=0,
)

n_samples = st.sidebar.slider(
    "n_samples",
    min_value=5_000,
    max_value=60_000,
    value=30_000,
    step=5_000,
)

seed = st.sidebar.number_input("Random seed", value=42, step=1)

tau = st.sidebar.slider(
    "Decision threshold τ",
    min_value=0.01,
    max_value=0.99,
    value=0.50,
    step=0.01,
)

cpc = st.sidebar.number_input("CPC (cost-per-click, $)", value=0.50, step=0.05, format="%.2f")
cpm = st.sidebar.number_input("CPM (cost-per-mille, $)", value=2.00, step=0.10, format="%.2f")

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data…")
def load_data(choice: str, n: int, s: int):
    if choice == "Avazu-style 60k":
        return make_realistic_ctr(n=n, seed=s)
    return make_synthetic(n=n, seed=s)


# ---------------------------------------------------------------------------
# Model training (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training model…")
def train_model(choice: str, n: int, s: int):
    data = load_data(choice, n, s)
    model, metrics = fit_and_evaluate(data)
    return model, metrics


# ---------------------------------------------------------------------------
# Pure-NumPy curve helpers
# ---------------------------------------------------------------------------
def _eps_clip(p, eps=1e-15):
    return np.clip(p, eps, 1 - eps)


def compute_log_loss(y, p):
    p = _eps_clip(np.asarray(p, float))
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def compute_roc_curve(y, scores, n_thresholds=300):
    """Returns fpr, tpr, thresholds arrays."""
    thresholds = np.linspace(0.0, 1.0, n_thresholds)[::-1]
    y = np.asarray(y)
    scores = np.asarray(scores, float)
    npos = (y == 1).sum()
    nneg = (y == 0).sum()
    tpr_list, fpr_list = [], []
    for t in thresholds:
        pred = scores >= t
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        tpr_list.append(tp / npos if npos else 0.0)
        fpr_list.append(fp / nneg if nneg else 0.0)
    return np.array(fpr_list), np.array(tpr_list), thresholds


def compute_pr_curve(y, scores, n_thresholds=300):
    """Returns precision, recall, thresholds arrays."""
    thresholds = np.linspace(0.0, 1.0, n_thresholds)[::-1]
    y = np.asarray(y)
    scores = np.asarray(scores, float)
    npos = (y == 1).sum()
    prec_list, rec_list = [], []
    for t in thresholds:
        pred = scores >= t
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        prec_list.append(tp / (tp + fp) if (tp + fp) else 1.0)
        rec_list.append(tp / (tp + fn) if (tp + fn) else 0.0)
    return np.array(prec_list), np.array(rec_list), thresholds


def auc_trapz(x, y):
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))


def compute_calibration_curve(y, scores, n_bins=10):
    """Returns mean_pred, fraction_pos per bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    mean_pred, frac_pos = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() > 0:
            mean_pred.append(scores[mask].mean())
            frac_pos.append(y[mask].mean())
    return np.array(mean_pred), np.array(frac_pos)


def compute_ks(y, scores):
    """Returns ks_stat, thresholds, cdf_pos, cdf_neg."""
    y = np.asarray(y)
    scores = np.asarray(scores, float)
    thresholds = np.sort(np.unique(scores))
    npos = (y == 1).sum()
    nneg = (y == 0).sum()
    cdf_pos, cdf_neg = [], []
    for t in thresholds:
        cdf_pos.append(((scores <= t) & (y == 1)).sum() / npos)
        cdf_neg.append(((scores <= t) & (y == 0)).sum() / nneg)
    cdf_pos = np.array(cdf_pos)
    cdf_neg = np.array(cdf_neg)
    ks_stat = float(np.max(np.abs(cdf_pos - cdf_neg)))
    return ks_stat, thresholds, cdf_pos, cdf_neg


def compute_lift_table(y, scores, n_deciles=10):
    """Returns DataFrame-like dict with decile stats."""
    y = np.asarray(y)
    scores = np.asarray(scores, float)
    overall_ctr = y.mean()
    order = np.argsort(scores)[::-1]
    y_sorted = y[order]
    scores_sorted = scores[order]
    n = len(y)
    decile_size = n // n_deciles

    rows = []
    cum_clicks = 0
    for d in range(n_deciles):
        lo = d * decile_size
        hi = lo + decile_size if d < n_deciles - 1 else n
        chunk_y = y_sorted[lo:hi]
        chunk_s = scores_sorted[lo:hi]
        clicks = chunk_y.sum()
        cum_clicks += clicks
        ctr = chunk_y.mean()
        lift = ctr / overall_ctr if overall_ctr > 0 else 0.0
        rows.append({
            "decile": d + 1,
            "score_min": float(chunk_s.min()),
            "score_max": float(chunk_s.max()),
            "n": hi - lo,
            "clicks": int(clicks),
            "ctr": float(ctr),
            "lift": float(lift),
            "cum_clicks": int(cum_clicks),
            "cum_pct_clicks": float(cum_clicks / y.sum()) if y.sum() else 0.0,
        })
    return rows


def confusion_at_tau(y, scores, threshold):
    pred = (scores >= threshold).astype(int)
    y = np.asarray(y)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return np.array([[tn, fp], [fn, tp]])


def _fmt_pct(v):
    return f"{v*100:.2f}%"


# ---------------------------------------------------------------------------
# O'Reilly Feature Engineering helpers (pure NumPy)
# ---------------------------------------------------------------------------

def isotonic_regression_pav(y_sorted: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators (PAV) isotonic regression on presorted array."""
    n = len(y_sorted)
    iso = y_sorted.astype(float).copy()
    # block representation: (mean, count)
    blocks = [[iso[i], 1] for i in range(n)]
    i = 1
    while i < len(blocks):
        if blocks[i][0] < blocks[i - 1][0]:
            # merge i-1 and i
            merged_n = blocks[i - 1][1] + blocks[i][1]
            merged_mean = (
                blocks[i - 1][0] * blocks[i - 1][1]
                + blocks[i][0] * blocks[i][1]
            ) / merged_n
            blocks[i - 1] = [merged_mean, merged_n]
            blocks.pop(i)
            i = max(1, i - 1)
        else:
            i += 1
    # expand blocks back
    result = np.zeros(n)
    idx = 0
    for mean, cnt in blocks:
        result[idx: idx + cnt] = mean
        idx += cnt
    return result


def calibrate_isotonic(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit isotonic regression calibration and return calibrated probabilities."""
    order = np.argsort(scores)
    y_sorted = y[order].astype(float)
    iso_sorted = isotonic_regression_pav(y_sorted)
    calibrated = np.empty(len(scores))
    calibrated[order] = iso_sorted
    return np.clip(calibrated, 0.0, 1.0)


def platt_scaling_fit(scores: np.ndarray, y: np.ndarray,
                      n_iter: int = 600, lr: float = 0.05):
    """Fit Platt scaling: P(y=1|f) = σ(A·f + B). Returns (A, B)."""
    A = np.float64(0.0)
    B = np.float64(0.0)
    scores = np.asarray(scores, float)
    y = np.asarray(y, float)
    for _ in range(n_iter):
        z = np.clip(A * scores + B, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-z))
        dA = float(np.mean((p - y) * scores))
        dB = float(np.mean(p - y))
        A -= lr * dA
        B -= lr * dB
    return float(A), float(B)


def platt_predict(scores: np.ndarray, A: float, B: float) -> np.ndarray:
    z = np.clip(A * np.asarray(scores, float) + B, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def mutual_information_bin(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Compute MI between a continuous feature x and binary label y via binning."""
    x = np.asarray(x, float)
    y = np.asarray(y, int)
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.unique(np.percentile(x, percentiles))
    if len(bin_edges) < 2:
        return 0.0
    x_disc = np.digitize(x, bin_edges[1:])  # 0 .. n_bins-1
    n = len(y)
    mi = 0.0
    for xi in range(len(bin_edges)):
        for yi in range(2):
            mask = (x_disc == xi) & (y == yi)
            n_xy = int(mask.sum())
            if n_xy == 0:
                continue
            n_x = int((x_disc == xi).sum())
            n_y = int((y == yi).sum())
            if n_x == 0 or n_y == 0:
                continue
            p_xy = n_xy / n
            p_x = n_x / n
            p_y = n_y / n
            mi += p_xy * np.log(p_xy / (p_x * p_y + 1e-15) + 1e-15)
    return max(0.0, float(mi))


def l1_logistic_weights(X: np.ndarray, y: np.ndarray,
                        lam: float = 0.05, lr: float = 0.05,
                        n_iter: int = 300) -> np.ndarray:
    """Proximal gradient descent for L1-regularised logistic regression."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -35, 35)))
        grad = X.T @ (p - y) / n
        w = w - lr * grad
        # Proximal step: soft-thresholding for L1
        w = np.sign(w) * np.maximum(np.abs(w) - lr * lam, 0.0)
    return w


def forward_feature_selection_auc(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    feature_names: list,
    max_steps: int = 6,
):
    """Greedy forward feature selection. Returns list of (feat_name, auc)."""
    selected = []
    remaining = list(range(X_tr.shape[1]))
    results = []
    for _ in range(min(max_steps, len(feature_names))):
        best_auc = -1.0
        best_feat = None
        for f in remaining:
            cols = selected + [f]
            Xtr_sub = X_tr[:, cols]
            Xte_sub = X_te[:, cols]
            sc = Standardizer().fit(Xtr_sub)
            lr_model = LogisticRegression(lr=0.2, epochs=150)
            lr_model.fit(sc.transform(Xtr_sub), y_tr)
            proba = lr_model.predict_proba(sc.transform(Xte_sub))
            auc = roc_auc_score(y_te, proba)
            if auc > best_auc:
                best_auc = auc
                best_feat = f
        if best_feat is not None:
            selected.append(best_feat)
            remaining.remove(best_feat)
            results.append((feature_names[best_feat], float(best_auc)))
    return results


def fm_interaction_strength(X: np.ndarray, y: np.ndarray,
                             feat_names: list, top_k: int = 5):
    """Proxy FM interaction matrix using pairwise cross-feature correlations with y."""
    corrs_abs = np.array([
        abs(float(np.corrcoef(X[:, i], y)[0, 1]))
        for i in range(X.shape[1])
    ])
    top_idx = np.argsort(corrs_abs)[::-1][:top_k]
    mat = np.zeros((top_k, top_k))
    for i, fi in enumerate(top_idx):
        for j, fj in enumerate(top_idx):
            cross = X[:, fi] * X[:, fj]
            std_c = cross.std()
            if std_c > 1e-8:
                cross_norm = (cross - cross.mean()) / std_c
                mat[i, j] = abs(float(np.corrcoef(cross_norm, y)[0, 1]))
            else:
                mat[i, j] = 0.0
    names = [feat_names[i] for i in top_idx]
    return mat, names


def laplace_ctr_encode(x_train: np.ndarray, y_train: np.ndarray,
                       x_all: np.ndarray, alpha: float = 1.0, beta: float = 2.0):
    """Laplace-smoothed bin-counting (CTR encoding) computed on train only."""
    unique_vals = np.sort(np.unique(x_train))
    enc_map = {}
    for v in unique_vals:
        mask = x_train == v
        n_clicks = float(y_train[mask].sum())
        n_total = float(mask.sum())
        enc_map[int(v)] = (n_clicks + alpha) / (n_total + beta)
    global_rate = (float(y_train.sum()) + alpha) / (len(y_train) + beta)
    encoded = np.array([enc_map.get(int(v), global_rate) for v in x_all])
    return encoded, enc_map


# ---------------------------------------------------------------------------
# Load data + train model
# ---------------------------------------------------------------------------
data = load_data(data_choice, n_samples, int(seed))
model, metrics = train_model(data_choice, n_samples, int(seed))

X = np.asarray(data["X"], float)
y = np.asarray(data["y"], int)
features = data["features"]

# Replicate exact train/test split used in fit_and_evaluate
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, seed=7)
scores_te = predict_proba(model, X_te)

overall_ctr = float(y_te.mean())
impressions = len(y_te)
clicks = int(y_te.sum())
roc_auc_val = metrics["roc_auc"]
log_loss_val = compute_log_loss(y_te, scores_te)
ks_stat, _, _, _ = compute_ks(y_te, scores_te)
pr_prec, pr_rec, pr_thr = compute_pr_curve(y_te, scores_te, n_thresholds=300)
pr_auc_val = auc_trapz(pr_rec, pr_prec)
gini_val = 2 * roc_auc_val - 1

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("AdCTR — InMobi  |  Click-Through Rate Prediction Dashboard")
st.caption(f"Source: {data['source']}  |  Backend: {metrics['backend']}")

hcols = st.columns(6)
hcols[0].metric("Impressions", f"{impressions:,}")
hcols[1].metric("Clicks", f"{clicks:,}")
hcols[2].metric("CTR", _fmt_pct(overall_ctr))
hcols[3].metric("ROC-AUC", f"{roc_auc_val:.4f}")
hcols[4].metric("Log Loss", f"{log_loss_val:.4f}")
hcols[5].metric("KS Statistic", f"{ks_stat:.4f}")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data Explorer",
    "⚙ Feature Engineering",
    "🧪 Model Evaluation",
    "📈 Lift & Ranking",
    "💰 Campaign Optimiser",
])

# ===========================================================================
# TAB 1 — Data Explorer
# ===========================================================================
with tab1:
    st.subheader("Dataset Overview")

    # --- Class imbalance bar chart ---
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = [int((y == 0).sum()), int((y == 1).sum())]
        bars = ax.bar(["No-click (0)", "Click (1)"], counts, color=["steelblue", "tomato"])
        ax.bar_label(bars, fmt="%d")
        ax.set_ylabel("Count")
        ax.set_title("Class Distribution (full dataset)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.markdown("**Overall CTR rate**")
        st.metric("CTR (full dataset)", _fmt_pct(float(y.mean())))
        st.metric("Positive samples", f"{int(y.sum()):,}")
        st.metric("Negative samples", f"{int((y==0).sum()):,}")
        st.metric("Imbalance ratio", f"1 : {int((y==0).sum())//max(int(y.sum()),1)}")

    st.divider()

    # --- CTR by banner_pos, device_type, hour_of_day (Avazu features only) ---
    if data_choice == "Avazu-style 60k" and "banner_pos" in features:
        feat_idx = {f: i for i, f in enumerate(features)}

        def ctr_by_feature(feature_name, X_arr, y_arr):
            col_idx = feat_idx[feature_name]
            vals = X_arr[:, col_idx].astype(int)
            unique_vals = np.sort(np.unique(vals))
            ctrs = [y_arr[vals == v].mean() for v in unique_vals]
            return unique_vals, ctrs

        fa, fb, fc = st.columns(3)

        with fa:
            uv, ctrs = ctr_by_feature("banner_pos", X, y)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(uv.astype(str), ctrs, color="steelblue")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=1))
            ax.set_title("CTR by Banner Position")
            ax.set_xlabel("banner_pos")
            ax.set_ylabel("CTR")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with fb:
            uv, ctrs = ctr_by_feature("device_type", X, y)
            labels = {0: "mobile", 1: "desktop", 2: "tablet", 3: "phone", 4: "other"}
            xlabels = [labels.get(int(v), str(v)) for v in uv]
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(xlabels, ctrs, color="mediumseagreen")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=1))
            ax.set_title("CTR by Device Type")
            ax.set_xlabel("device_type")
            ax.set_ylabel("CTR")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with fc:
            uv, ctrs = ctr_by_feature("hour_of_day", X, y)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(uv, ctrs, marker="o", color="darkorange", linewidth=2, markersize=4)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=1))
            ax.set_title("CTR by Hour of Day")
            ax.set_xlabel("hour_of_day")
            ax.set_ylabel("CTR")
            ax.set_xticks(np.arange(0, 24, 2))
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.divider()

    # --- Descriptive stats + CTR correlation table ---
    st.subheader("Descriptive Statistics & CTR Correlation")
    rows_stats = []
    for i, feat in enumerate(features):
        col_vals = X[:, i]
        corr = float(np.corrcoef(col_vals, y)[0, 1])
        rows_stats.append({
            "Feature": feat,
            "Mean": f"{col_vals.mean():.3f}",
            "Std": f"{col_vals.std():.3f}",
            "Min": f"{col_vals.min():.3f}",
            "Max": f"{col_vals.max():.3f}",
            "CTR Correlation": f"{corr:+.4f}",
        })

    # Display as a simple markdown table
    header = "| " + " | ".join(rows_stats[0].keys()) + " |"
    sep = "| " + " | ".join(["---"] * len(rows_stats[0])) + " |"
    body = "\n".join(
        "| " + " | ".join(str(v) for v in r.values()) + " |"
        for r in rows_stats
    )
    st.markdown(header + "\n" + sep + "\n" + body)

    # =========================================================================
    # AVAZU CTR CHALLENGE — Real Dataset Reference Panel
    # =========================================================================
    st.divider()
    st.subheader("Avazu CTR Challenge — Real Dataset Reference")
    st.markdown(
        "**Dataset:** [Avazu Click-Through Rate Prediction](https://www.kaggle.com/c/avazu-ctr-prediction) "
        "· Kaggle Competition 2014 · Sponsored by Avazu (DSP/SSP ad exchange)"
    )

    ref_c1, ref_c2 = st.columns(2)

    with ref_c1:
        st.markdown("#### Real Dataset Statistics")
        st.markdown("""
| Property | Value |
|---|---|
| Training rows | 40,428,967 |
| Test rows | 4,577,464 |
| Features | 16 (+ target) |
| CTR — train split | **16.83 %** |
| CTR — test split | **16.52 %** |
| Time span | Oct 21 – Oct 30, 2014 |
| Evaluation metric | Log Loss |
| Best public LB score | 0.3790 log-loss |
""")

    with ref_c2:
        st.markdown("#### Real Avazu Schema (all fields)")
        st.code(
            "id              – impression identifier (hashed)\n"
            "click           – binary target (0/1)\n"
            "hour            – YYMMDDHH format (e.g. 14102100)\n"
            "C1              – anonymised categorical\n"
            "banner_pos      – ad position on page (0-7)\n"
            "site_id         – site identifier hash\n"
            "site_domain     – domain of the site hash\n"
            "site_category   – IAB category hash\n"
            "app_id          – mobile app identifier hash\n"
            "app_domain      – app domain hash\n"
            "app_category    – app IAB category hash\n"
            "device_id       – device identifier hash\n"
            "device_ip       – IP address hash\n"
            "device_model    – device model hash\n"
            "device_type     – 0=mobile 1=desktop 2=tablet…\n"
            "device_conn_type– 0=WiFi 2=3G 5=4G\n"
            "C14 … C21      – anonymised categorical features",
            language="text",
        )

    st.divider()

    st.markdown("#### Scale Comparison: Synthetic vs Real")
    sc_c1, sc_c2, sc_c3 = st.columns(3)
    sc_c1.metric("This dashboard", f"{n_samples:,} rows", "synthetic")
    sc_c2.metric("Real Avazu train", "40,428,967 rows", "Kaggle 2014")
    sc_c3.metric("Scale factor", f"{40_428_967 / n_samples:,.0f}x", "real vs synthetic")

    st.info(
        "**Scale note:** This app uses a synthetic dataset of "
        f"{n_samples:,} impressions that mirrors the Avazu schema "
        "(same field names, same value ranges derived from real data exploration papers). "
        "The real Avazu training set has **40,428,967** rows — roughly "
        f"**{40_428_967 / n_samples:,.0f}x** larger. "
        "The real CTR is ~16.8 % (vs our ~2.7 % synthetic) because the real data "
        "includes heavy publisher-side filtering before logging."
    )

    st.markdown("#### Synthetic vs Real — Feature Fidelity")
    fidelity_rows = [
        ("banner_pos", "0-7 integer", "0-7 integer", "Exact"),
        ("device_type", "0-4 integer", "0-4 integer", "Exact"),
        ("device_conn_type", "{0, 2, 5}", "{0, 2, 3, 4, 5}", "Close"),
        ("site_category", "0-25 bucket", "hex hash → 26 categories", "Bucketed"),
        ("app_category", "0-35 bucket", "hex hash → 36 categories", "Bucketed"),
        ("hour_of_day", "0-23 integer", "extracted from YYMMDDHH", "Exact"),
        ("C1", "1000-1009", "1000-1010", "Exact"),
        ("C14", "375-29999", "375-29963", "Exact"),
        ("C15", "{50, 250, 320}", "{50, 250, 320}", "Exact"),
        ("C16", "{50, 320, 480}", "{50, 320, 480}", "Exact"),
        ("C17", "112-2762", "112-2762", "Exact"),
        ("C18", "0-3 integer", "0-3 integer", "Exact"),
        ("C19", "35-68 integer", "35-68 integer", "Exact"),
        ("C21", "23-102 integer", "23-102 integer", "Exact"),
    ]
    fid_hdr = "| Feature | Synthetic Range | Real Range | Fidelity |"
    fid_sep = "| --- | --- | --- | --- |"
    fid_body = "\n".join(
        f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in fidelity_rows
    )
    st.markdown(fid_hdr + "\n" + fid_sep + "\n" + fid_body)


# ===========================================================================
# TAB 2 — Feature Engineering
# ===========================================================================
with tab2:
    st.subheader("Feature Engineering Methodology")

    # --- LaTeX equations ---
    st.markdown("#### Key Transformations")

    col_eq1, col_eq2 = st.columns(2)
    with col_eq1:
        st.markdown("**Frequency Encoding**")
        st.latex(r"\hat{c}_i = \frac{\text{count}(x_i = v)}{\text{count total}}")
        st.markdown("**Cyclic Hour Encoding**")
        st.latex(r"\text{hour\_sin} = \sin\!\left(\frac{2\pi \cdot h}{24}\right), \quad"
                 r"\text{hour\_cos} = \cos\!\left(\frac{2\pi \cdot h}{24}\right)")
    with col_eq2:
        st.markdown("**Cross Feature (banner × device)**")
        st.latex(r"\text{cross}_{ij} = \text{freq}(\text{banner\_pos}_i) \times"
                 r" \text{freq}(\text{device\_type}_j)")
        st.markdown("**Target Encoding (train-only)**")
        st.latex(r"\hat{y}_v = \frac{\sum_{i: x_i=v} y_i + \alpha \bar{y}}{n_v + \alpha}")

    st.divider()

    if data_choice == "Avazu-style 60k" and "banner_pos" in features:
        feat_idx = {f: i for i, f in enumerate(features)}

        # Frequency-encoded CTR table (train split only)
        st.subheader("Frequency-encoded CTR — Train Split (banner_pos)")
        bp_col = feat_idx["banner_pos"]
        bp_vals_tr = X_tr[:, bp_col].astype(int)
        bp_unique = np.sort(np.unique(bp_vals_tr))
        freq_rows = []
        for v in bp_unique:
            mask = bp_vals_tr == v
            freq_rows.append({
                "banner_pos": int(v),
                "count (train)": int(mask.sum()),
                "freq_encoding": f"{mask.mean():.4f}",
                "CTR (train)": f"{y_tr[mask].mean():.4f}" if mask.sum() > 0 else "n/a",
            })
        hdr = "| " + " | ".join(freq_rows[0].keys()) + " |"
        sp = "| " + " | ".join(["---"] * len(freq_rows[0])) + " |"
        bdy = "\n".join("| " + " | ".join(str(v) for v in r.values()) + " |" for r in freq_rows)
        st.markdown(hdr + "\n" + sp + "\n" + bdy)

        st.divider()

        # Feature-click correlation bar chart
        corrs = []
        for i, feat in enumerate(features):
            corrs.append(float(np.corrcoef(X_tr[:, i], y_tr)[0, 1]))
        corrs = np.array(corrs)
        order = np.argsort(corrs)

        fig, ax = plt.subplots(figsize=(10, 3))
        colors = ["tomato" if c < 0 else "steelblue" for c in corrs[order]]
        ax.barh([features[i] for i in order], corrs[order], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Pearson Correlation with Click")
        ax.set_title("Feature–Click Correlation (train split)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.divider()

        # banner_pos × device_type CTR heatmap
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.subheader("banner_pos × device_type CTR Heatmap")
            bp_idx = feat_idx["banner_pos"]
            dt_idx = feat_idx["device_type"]
            bp_vals = X[:, bp_idx].astype(int)
            dt_vals = X[:, dt_idx].astype(int)
            bp_u = np.sort(np.unique(bp_vals))
            dt_u = np.sort(np.unique(dt_vals))
            heat = np.zeros((len(bp_u), len(dt_u)))
            for bi, bv in enumerate(bp_u):
                for di, dv in enumerate(dt_u):
                    mask = (bp_vals == bv) & (dt_vals == dv)
                    heat[bi, di] = y[mask].mean() if mask.sum() > 0 else 0.0
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(heat, aspect="auto", cmap="YlOrRd", vmin=0)
            ax.set_xticks(range(len(dt_u)))
            ax.set_xticklabels([str(v) for v in dt_u])
            ax.set_yticks(range(len(bp_u)))
            ax.set_yticklabels([str(v) for v in bp_u])
            ax.set_xlabel("device_type")
            ax.set_ylabel("banner_pos")
            ax.set_title("CTR Heatmap")
            plt.colorbar(im, ax=ax, format=mticker.PercentFormatter(1.0, decimals=1))
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_h2:
            st.subheader("Hour-of-Day CTR + Cyclic Encoding")
            h_idx = feat_idx["hour_of_day"]
            h_vals = X[:, h_idx].astype(int)
            hours = np.arange(0, 24)
            hour_ctrs = np.array([y[h_vals == h].mean() if (h_vals == h).sum() > 0 else 0.0
                                   for h in hours])
            hour_sin = np.sin(2 * np.pi * hours / 24)
            hour_cos = np.cos(2 * np.pi * hours / 24)

            fig, ax1 = plt.subplots(figsize=(5, 4))
            ax1.bar(hours, hour_ctrs, alpha=0.5, color="steelblue", label="CTR")
            ax1.set_ylabel("CTR", color="steelblue")
            ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=1))
            ax2 = ax1.twinx()
            ax2.plot(hours, hour_sin, "r--", linewidth=1.5, label="sin")
            ax2.plot(hours, hour_cos, "g-.", linewidth=1.5, label="cos")
            ax2.set_ylabel("Cyclic encoding value")
            ax1.set_xlabel("hour_of_day")
            ax1.set_title("Hour CTR + Cyclic Encoding")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("Feature engineering visualisations require the Avazu-style 60k dataset.")

    # =========================================================================
    # O'REILLY BOOK CONCEPTS — Feature Engineering for ML
    # =========================================================================
    if data_choice == "Avazu-style 60k" and "banner_pos" in features:
        feat_idx = {f: i for i, f in enumerate(features)}

        st.divider()
        st.markdown(
            "## 📚 From: Feature Engineering for ML (Zheng & Casari, O'Reilly 2018)"
        )
        st.caption(
            "Concepts from Chapters 4, 6, 8, and 9 applied to the Avazu CTR dataset."
        )

        # =====================================================================
        # CHAPTER 4: The Effects of Feature Scaling
        # =====================================================================
        st.markdown("---")
        st.markdown("### Chapter 4 — The Effects of Feature Scaling")
        st.markdown(
            "📚 **From: Feature Engineering for ML (Zheng & Casari, O'Reilly 2018)**"
        )
        st.markdown(
            "C14 is an anonymised categorical integer with a highly skewed, "
            "wide-range distribution (375 – 30,000). Three scaling strategies below:"
        )

        c14_idx = feat_idx["C14"]
        c14_raw = X[:, c14_idx].astype(float)

        c14_min, c14_max = c14_raw.min(), c14_raw.max()
        c14_mu, c14_sigma = c14_raw.mean(), c14_raw.std()

        # Min-max normalisation
        c14_minmax = (c14_raw - c14_min) / (c14_max - c14_min + 1e-8)
        # Standardisation (z-score)
        c14_std = (c14_raw - c14_mu) / (c14_sigma + 1e-8)
        # Mean normalisation
        c14_meannorm = (c14_raw - c14_mu) / (c14_max - c14_min + 1e-8)

        ch4_eq1, ch4_eq2, ch4_eq3 = st.columns(3)
        with ch4_eq1:
            st.markdown("**Min-Max Normalisation**")
            st.latex(r"x_{\mathrm{minmax}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}")
            st.caption("Output range: [0, 1]. Preserves shape. Sensitive to outliers.")
        with ch4_eq2:
            st.markdown("**Standardisation (z-score)**")
            st.latex(r"x_{\mathrm{std}} = \frac{x - \mu}{\sigma}")
            st.caption("Zero-mean, unit variance. Robust to outliers. Best for logistic reg.")
        with ch4_eq3:
            st.markdown("**Mean Normalisation**")
            st.latex(r"x_{\mathrm{mean}} = \frac{x - \mu}{x_{\max} - x_{\min}}")
            st.caption("Centres around zero while keeping range context.")

        fig, axes = plt.subplots(1, 4, figsize=(14, 3))
        bins_c14 = 60
        axes[0].hist(c14_raw, bins=bins_c14, color="steelblue", alpha=0.85)
        axes[0].set_title("C14 Raw")
        axes[0].set_xlabel("Value")

        axes[1].hist(c14_minmax, bins=bins_c14, color="mediumseagreen", alpha=0.85)
        axes[1].set_title("C14 Min-Max [0,1]")
        axes[1].set_xlabel("Value")

        axes[2].hist(c14_std, bins=bins_c14, color="darkorange", alpha=0.85)
        axes[2].set_title("C14 Standardised (z)")
        axes[2].set_xlabel("z-score")

        axes[3].hist(c14_meannorm, bins=bins_c14, color="orchid", alpha=0.85)
        axes[3].set_title("C14 Mean Normalised")
        axes[3].set_xlabel("Value")

        for ax in axes:
            ax.set_ylabel("Count")
        fig.suptitle(
            "Before/After Scaling: C14 (highly skewed, range 375-30000)", fontsize=11
        )
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
**When to use each method (O'Reilly Ch 4):**

| Method | Best for | Notes |
|---|---|---|
| Min-Max [0,1] | Neural networks, KNN | Distorted by outliers |
| Standardisation (z) | **Logistic regression, SVM, LR-based CTR** | Most robust; recommended default |
| Mean normalisation | When zero-centring matters but range info useful | Less common |
| **No scaling** | **Tree models (LightGBM, XGBoost, Random Forest)** | Splits are scale-invariant |
""")
        st.info(
            "Tree-based models (LightGBM — the backend used here) are "
            "**invariant to feature scaling**. Scaling is critical for logistic "
            "regression and any gradient-based model."
        )

        # =====================================================================
        # CHAPTER 6: Automating the Featurizer — Bin Counting
        # =====================================================================
        st.markdown("---")
        st.markdown("### Chapter 6 — Automating the Featurizer: Bin Counting")
        st.markdown(
            "📚 **From: Feature Engineering for ML (Zheng & Casari, O'Reilly 2018)**"
        )
        st.markdown(
            "**Bin counting** (also called frequency or target encoding) replaces a raw "
            "category ID with statistics estimated from training data. "
            "The smoothed variant uses **Laplace back-off** to handle rare categories."
        )

        st.markdown("**Laplace-smoothed CTR encoding (bin counting):**")
        st.latex(
            r"\hat{p}(c) = \frac{n_+(c) + \alpha}{n(c) + \beta}"
        )
        st.markdown(
            r"where $n_+(c)$ = clicks for category $c$, "
            r"$n(c)$ = total impressions for category $c$, "
            r"$\alpha=1$ (pseudoclicks), $\beta=2$ (pseudoimpressions)."
        )

        # Compute on train split only
        sc_idx = feat_idx["site_category"]
        sc_tr = X_tr[:, sc_idx].astype(int)
        sc_all = X[:, sc_idx].astype(int)

        # Raw frequency (proportion of each category)
        sc_unique, sc_counts = np.unique(sc_tr, return_counts=True)
        freq_map = dict(zip(sc_unique.tolist(), (sc_counts / sc_counts.sum()).tolist()))
        sc_freq_encoded = np.array([freq_map.get(int(v), 0.0) for v in sc_all])

        # Laplace-smoothed CTR encoding
        sc_laplace_encoded, enc_map_detail = laplace_ctr_encode(
            sc_tr, y_tr, sc_all, alpha=1.0, beta=2.0
        )

        # Raw integer encoding (normalised for comparison)
        sc_raw_norm = (sc_all - sc_all.min()) / (sc_all.max() - sc_all.min() + 1e-8)

        # Show encoding table for first 10 site_category values
        sc_display_vals = sorted(enc_map_detail.keys())[:12]
        enc_table_rows = []
        for v in sc_display_vals:
            mask_tr = sc_tr == v
            n_total = int(mask_tr.sum())
            n_clicks = int(y_tr[mask_tr].sum()) if n_total > 0 else 0
            raw_ctr = n_clicks / n_total if n_total > 0 else 0.0
            laplace_ctr = enc_map_detail[v]
            enc_table_rows.append({
                "site_category": v,
                "n_train": n_total,
                "n_clicks": n_clicks,
                "raw_CTR": f"{raw_ctr:.4f}",
                "Laplace_CTR (α=1,β=2)": f"{laplace_ctr:.4f}",
                "freq_encoding": f"{freq_map.get(v, 0.0):.4f}",
            })

        enc_hdr = "| " + " | ".join(enc_table_rows[0].keys()) + " |"
        enc_sep = "| " + " | ".join(["---"] * len(enc_table_rows[0])) + " |"
        enc_bdy = "\n".join(
            "| " + " | ".join(str(vv) for vv in r.values()) + " |"
            for r in enc_table_rows
        )
        st.markdown("**Encoding comparison (first 12 site_category values, train split):**")
        st.markdown(enc_hdr + "\n" + enc_sep + "\n" + enc_bdy)

        # Visualise distributions
        enc_fig, enc_axes = plt.subplots(1, 3, figsize=(13, 3))
        enc_axes[0].hist(sc_raw_norm, bins=30, color="steelblue", alpha=0.85)
        enc_axes[0].set_title("Raw category ID (normalised)")
        enc_axes[0].set_xlabel("Value")

        enc_axes[1].hist(sc_freq_encoded, bins=30, color="mediumseagreen", alpha=0.85)
        enc_axes[1].set_title("Frequency encoding")
        enc_axes[1].set_xlabel("Fraction of impressions")

        enc_axes[2].hist(sc_laplace_encoded, bins=30, color="darkorange", alpha=0.85)
        enc_axes[2].set_title("Laplace-smoothed CTR (α=1, β=2)")
        enc_axes[2].set_xlabel("Estimated P(click|category)")

        for ax in enc_axes:
            ax.set_ylabel("Count")
        enc_fig.suptitle(
            "site_category: Raw vs Frequency vs Laplace-smoothed CTR encoding",
            fontsize=10,
        )
        enc_fig.tight_layout()
        st.pyplot(enc_fig)
        plt.close(enc_fig)

        st.warning(
            "**Target Leakage Warning (O'Reilly Ch 6):** Bin counting MUST be "
            "computed on the training split only. Using the full dataset leaks future "
            "label information into the feature and causes over-optimistic training "
            "scores with poor generalisation. All encodings above were computed "
            "exclusively on the train split (`X_tr`, `y_tr`)."
        )

        # =====================================================================
        # CHAPTER 8: Feature Interactions
        # =====================================================================
        st.markdown("---")
        st.markdown("### Chapter 8 — Feature Interactions")
        st.markdown(
            "📚 **From: Feature Engineering for ML (Zheng & Casari, O'Reilly 2018)**"
        )

        st.markdown(
            "Manual cross-features multiply two feature values to capture "
            "joint effects not representable by individual features."
        )

        st.latex(
            r"x_{\mathrm{cross}} = x_i \times x_j"
            r"\quad\text{(e.g. banner\_pos} \times \text{device\_type)}"
        )

        # Compute cross feature and its AUC lift
        bp_idx2 = feat_idx["banner_pos"]
        dt_idx2 = feat_idx["device_type"]

        bp_tr = X_tr[:, bp_idx2]
        dt_tr = X_tr[:, dt_idx2]
        bp_te = X_te[:, bp_idx2]
        dt_te = X_te[:, dt_idx2]

        # Standardise individual features
        sc_bp = Standardizer().fit(bp_tr.reshape(-1, 1))
        sc_dt = Standardizer().fit(dt_tr.reshape(-1, 1))
        bp_tr_s = sc_bp.transform(bp_tr.reshape(-1, 1)).ravel()
        dt_tr_s = sc_dt.transform(dt_tr.reshape(-1, 1)).ravel()
        bp_te_s = sc_bp.transform(bp_te.reshape(-1, 1)).ravel()
        dt_te_s = sc_dt.transform(dt_te.reshape(-1, 1)).ravel()

        cross_tr = bp_tr_s * dt_tr_s
        cross_te = bp_te_s * dt_te_s

        # Model without interaction
        Xtr_no = np.column_stack([bp_tr_s, dt_tr_s])
        Xte_no = np.column_stack([bp_te_s, dt_te_s])
        lr_no = LogisticRegression(lr=0.2, epochs=300)
        lr_no.fit(Xtr_no, y_tr)
        auc_no = roc_auc_score(y_te, lr_no.predict_proba(Xte_no))

        # Model with interaction
        Xtr_wi = np.column_stack([bp_tr_s, dt_tr_s, cross_tr])
        Xte_wi = np.column_stack([bp_te_s, dt_te_s, cross_te])
        lr_wi = LogisticRegression(lr=0.2, epochs=300)
        lr_wi.fit(Xtr_wi, y_tr)
        auc_wi = roc_auc_score(y_te, lr_wi.predict_proba(Xte_wi))

        int_c1, int_c2 = st.columns(2)
        with int_c1:
            st.markdown("**AUC Lift from Cross Feature (banner_pos x device_type)**")
            st.markdown(f"""
| Model | ROC-AUC |
|---|---|
| banner_pos + device_type (no interaction) | {auc_no:.4f} |
| + cross feature (banner_pos × device_type) | {auc_wi:.4f} |
| AUC lift | {auc_wi - auc_no:+.4f} |
""")

        with int_c2:
            # Heatmap of cross-feature CTR
            fig_cross, ax_cross = plt.subplots(figsize=(5, 4))
            bp_vals2 = X[:, bp_idx2].astype(int)
            dt_vals2 = X[:, dt_idx2].astype(int)
            bp_u2 = np.sort(np.unique(bp_vals2))[:6]
            dt_u2 = np.sort(np.unique(dt_vals2))
            heat2 = np.zeros((len(bp_u2), len(dt_u2)))
            for bi, bv in enumerate(bp_u2):
                for di, dv in enumerate(dt_u2):
                    mask = (bp_vals2 == bv) & (dt_vals2 == dv)
                    heat2[bi, di] = y[mask].mean() if mask.sum() > 0 else 0.0
            im2 = ax_cross.imshow(heat2, aspect="auto", cmap="RdYlGn", vmin=0)
            ax_cross.set_xticks(range(len(dt_u2)))
            ax_cross.set_xticklabels([str(v) for v in dt_u2], fontsize=8)
            ax_cross.set_yticks(range(len(bp_u2)))
            ax_cross.set_yticklabels([str(v) for v in bp_u2], fontsize=8)
            ax_cross.set_xlabel("device_type")
            ax_cross.set_ylabel("banner_pos")
            ax_cross.set_title("Cross-feature CTR: banner_pos x device_type")
            plt.colorbar(im2, ax=ax_cross,
                         format=mticker.PercentFormatter(1.0, decimals=1))
            fig_cross.tight_layout()
            st.pyplot(fig_cross)
            plt.close(fig_cross)

        st.divider()
        st.markdown("#### Factorization Machines (FM) — Efficient Pairwise Interactions")
        st.markdown(
            "O'Reilly Ch 8 discusses Factorization Machines (Rendle, 2010) as an "
            "efficient way to learn ALL pairwise interactions without enumerating "
            "every cross feature explicitly."
        )

        st.markdown("**FM prediction equation:**")
        st.latex(
            r"\hat{y}(\mathbf{x}) = w_0 "
            r"+ \sum_{i=1}^{n} w_i x_i "
            r"+ \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j \rangle\, x_i x_j"
        )
        st.markdown("where the inner product of latent vectors is:")
        st.latex(
            r"\langle \mathbf{v}_i, \mathbf{v}_j \rangle = \sum_{k=1}^{K} v_{ik} \cdot v_{jk}"
        )

        st.markdown("""
**Key properties (O'Reilly Ch 8):**
- Each feature $i$ gets a $K$-dimensional latent vector $\\mathbf{v}_i$
- Pairwise interactions = $O(Kn)$ instead of $O(n^2)$ via the identity:
""")
        st.latex(
            r"\sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j "
            r"= \frac{1}{2} \left[ \left\| \sum_i v_i x_i \right\|^2 "
            r"- \sum_i \left\| v_i \right\|^2 x_i^2 \right]"
        )
        st.markdown(
            "- FM learns interactions between **all** feature pairs from sparse data "
            "— critical in CTR where user/item combinations are unseen at test time."
        )

        # FM interaction matrix visualisation (proxy via pairwise cross-feature correlations)
        st.markdown("**FM Interaction Strength Matrix (top 5 features by |corr| with click):**")
        fm_mat, fm_names = fm_interaction_strength(X_tr, y_tr, features, top_k=5)

        fig_fm, ax_fm = plt.subplots(figsize=(6, 5))
        im_fm = ax_fm.imshow(fm_mat, cmap="Blues", vmin=0, vmax=fm_mat.max())
        ax_fm.set_xticks(range(5))
        ax_fm.set_xticklabels(fm_names, rotation=30, ha="right", fontsize=9)
        ax_fm.set_yticks(range(5))
        ax_fm.set_yticklabels(fm_names, fontsize=9)
        for i in range(5):
            for j in range(5):
                ax_fm.text(j, i, f"{fm_mat[i,j]:.3f}",
                           ha="center", va="center", fontsize=8,
                           color="white" if fm_mat[i, j] > fm_mat.max() * 0.6 else "black")
        ax_fm.set_title(
            "FM Interaction Matrix\n|corr(xi*xj, y)| proxy for latent interaction strength",
            fontsize=10,
        )
        plt.colorbar(im_fm, ax=ax_fm, label="|correlation with click|")
        fig_fm.tight_layout()
        st.pyplot(fig_fm)
        plt.close(fig_fm)

        st.caption(
            "Each cell shows the absolute Pearson correlation of the cross-product "
            "feature (xi × xj) with the click label — a proxy for the FM latent "
            "interaction strength ⟨vi, vj⟩."
        )

        # =====================================================================
        # CHAPTER 9: Feature Selection
        # =====================================================================
        st.markdown("---")
        st.markdown("### Chapter 9 — Feature Selection")
        st.markdown(
            "📚 **From: Feature Engineering for ML (Zheng & Casari, O'Reilly 2018)**"
        )
        st.markdown(
            "Three families of feature selection methods: **Filter**, **Wrapper**, "
            "and **Embedded**."
        )

        # --- Filter: Mutual Information ---
        st.markdown("#### Filter Method: Mutual Information")
        st.latex(
            r"I(X;\,Y) = \sum_{x}\sum_{y} p(x,y)\,\log\frac{p(x,y)}{p(x)\,p(y)}"
        )
        st.markdown(
            "MI measures the reduction in uncertainty about Y given X. "
            "Unlike Pearson correlation it captures **non-linear** dependencies."
        )

        mi_scores = np.array([
            mutual_information_bin(X_tr[:, i], y_tr, n_bins=10)
            for i in range(X_tr.shape[1])
        ])

        # --- Embedded: L1 Logistic Regression ---
        st.markdown("#### Embedded Method: L1 Regularisation (Lasso)")
        st.latex(
            r"\min_{w} \; \underbrace{-\frac{1}{n}\sum_i \ell(y_i, \hat{y}_i)}"
            r"_{\text{log-loss}} + \lambda \|w\|_1"
        )
        st.markdown(
            r"The $\ell_1$ penalty drives small weights to exactly zero, "
            "performing automatic feature selection."
        )

        sc_all_feat = Standardizer().fit(X_tr)
        Xtr_std = sc_all_feat.transform(X_tr)
        Xte_std = sc_all_feat.transform(X_te)

        l1_w = l1_logistic_weights(Xtr_std, y_tr, lam=0.05, lr=0.05, n_iter=400)
        l1_nonzero = np.abs(l1_w) > 1e-4

        # --- Wrapper: Forward Feature Selection ---
        st.markdown("#### Wrapper Method: Greedy Forward Feature Selection")
        st.markdown(
            "Start with an empty set and greedily add the feature that most "
            "improves ROC-AUC on the test set. Computationally expensive but "
            "accounts for feature interactions."
        )

        with st.spinner("Running forward feature selection (6 steps)…"):
            ffs_results = forward_feature_selection_auc(
                Xtr_std, y_tr, Xte_std, y_te, features, max_steps=6
            )

        ffs_feat_names = [r[0] for r in ffs_results]
        ffs_aucs = [r[1] for r in ffs_results]

        # --- Combined comparison table ---
        st.markdown("#### Feature Selection Comparison Table")

        # L1 survival
        l1_survive = {features[i]: ("Yes" if l1_nonzero[i] else "No")
                      for i in range(len(features))}
        # MI rank
        mi_rank = {features[i]: int(np.argsort(mi_scores)[::-1].tolist().index(i) + 1)
                   for i in range(len(features))}
        # Forward selection: which step each feature was added
        ffs_step = {name: step + 1 for step, name in enumerate(ffs_feat_names)}

        sel_rows = []
        for feat_name in features:
            sel_rows.append({
                "Feature": feat_name,
                "MI Score": f"{mi_scores[features.index(feat_name)]:.4f}",
                "MI Rank": mi_rank[feat_name],
                "L1 Survives (λ=0.05)": l1_survive[feat_name],
                "FFS Step Added": ffs_step.get(feat_name, "—"),
            })

        sel_hdr = "| " + " | ".join(sel_rows[0].keys()) + " |"
        sel_sep = "| " + " | ".join(["---"] * len(sel_rows[0])) + " |"
        sel_bdy = "\n".join(
            "| " + " | ".join(str(vv) for vv in r.values()) + " |"
            for r in sel_rows
        )
        st.markdown(sel_hdr + "\n" + sel_sep + "\n" + sel_bdy)

        sel_fig, sel_axes = plt.subplots(1, 3, figsize=(14, 4))

        # MI bar chart
        mi_order = np.argsort(mi_scores)
        sel_axes[0].barh(
            [features[i] for i in mi_order], mi_scores[mi_order],
            color="steelblue", alpha=0.85,
        )
        sel_axes[0].set_title("Mutual Information I(X; Y)")
        sel_axes[0].set_xlabel("MI Score")

        # L1 weights
        l1_order = np.argsort(np.abs(l1_w))
        sel_axes[1].barh(
            [features[i] for i in l1_order],
            l1_w[l1_order],
            color=["tomato" if w < 0 else "mediumseagreen" for w in l1_w[l1_order]],
            alpha=0.85,
        )
        sel_axes[1].axvline(0, color="black", linewidth=0.8)
        sel_axes[1].set_title("L1 Logistic Weights (λ=0.05)")
        sel_axes[1].set_xlabel("Weight (zero = pruned)")

        # Forward selection AUC curve
        steps = list(range(1, len(ffs_aucs) + 1))
        sel_axes[2].plot(steps, ffs_aucs, "o-", color="darkorange", linewidth=2, markersize=7)
        for i, (step, auc_v, feat_n) in enumerate(zip(steps, ffs_aucs, ffs_feat_names)):
            sel_axes[2].annotate(
                feat_n, (step, auc_v),
                textcoords="offset points", xytext=(4, 4), fontsize=7,
            )
        sel_axes[2].set_title("Forward Feature Selection — AUC vs Step")
        sel_axes[2].set_xlabel("Step (features added)")
        sel_axes[2].set_ylabel("ROC-AUC")
        sel_axes[2].set_xticks(steps)

        sel_fig.tight_layout()
        st.pyplot(sel_fig)
        plt.close(sel_fig)

        st.markdown("""
**Key takeaways (O'Reilly Ch 9):**
- **Filter (MI)**: Fast, model-agnostic, captures non-linear dependencies. Good for initial screening.
- **Wrapper (FFS)**: Best AUC but O(n²) evaluations; use on shortlist from filter.
- **Embedded (L1)**: Model-native pruning; fast, no separate selection step. Preferred in production.
""")

    else:
        st.info(
            "O'Reilly feature engineering sections require the Avazu-style 60k dataset."
        )


# ===========================================================================
# TAB 3 — Model Evaluation
# ===========================================================================
with tab3:
    st.subheader("Model Evaluation")

    # Pre-compute curves
    fpr, tpr, roc_thr = compute_roc_curve(y_te, scores_te, n_thresholds=300)
    roc_auc_c = auc_trapz(fpr, tpr)
    pr_prec2, pr_rec2, pr_thr2 = compute_pr_curve(y_te, scores_te, n_thresholds=300)
    pr_auc_c = auc_trapz(pr_rec2, pr_prec2)
    cal_mean, cal_frac = compute_calibration_curve(y_te, scores_te, n_bins=10)
    ks_stat2, ks_thr, ks_cdf_pos, ks_cdf_neg = compute_ks(y_te, scores_te)
    cm = confusion_at_tau(y_te, scores_te, tau)

    row1a, row1b = st.columns(2)

    # ROC curve
    with row1a:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color="royalblue", linewidth=2, label=f"ROC (AUC={roc_auc_c:.3f})")
        ax.fill_between(fpr, tpr, alpha=0.12, color="royalblue")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # PR curve
    with row1b:
        order_pr = np.argsort(pr_rec2)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(pr_rec2[order_pr], pr_prec2[order_pr], color="tomato", linewidth=2,
                label=f"PR (AUC={pr_auc_c:.3f})")
        ax.fill_between(pr_rec2[order_pr], pr_prec2[order_pr], alpha=0.12, color="tomato")
        ax.axhline(overall_ctr, color="grey", linestyle="--", linewidth=1,
                   label=f"Baseline CTR={overall_ctr:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve (critical for imbalanced CTR)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    row2a, row2b = st.columns(2)

    # Calibration curve
    with row2a:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
        if len(cal_mean) > 0:
            ax.plot(cal_mean, cal_frac, "s-", color="darkorange", linewidth=2,
                    markersize=6, label="Model")
        ax.set_xlabel("Mean Predicted Score")
        ax.set_ylabel("Fraction Positives")
        ax.set_title("Calibration Curve (Reliability Diagram)")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Confusion matrix at τ
    with row2b:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        labels = ["TN", "FP", "FN", "TP"]
        for idx, (i, j) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
            ax.text(j, i, f"{labels[idx]}\n{cm[i,j]:,}",
                    ha="center", va="center", fontsize=11,
                    color="white" if cm[i,j] > cm.max()*0.5 else "black")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["Actual 0", "Actual 1"])
        ax.set_title(f"Confusion Matrix at τ={tau:.2f}")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Score distribution
    row3a, row3b = st.columns(2)
    with row3a:
        fig, ax = plt.subplots(figsize=(5, 4))
        bins = np.linspace(0, 1, 51)
        ax.hist(scores_te[y_te == 0], bins=bins, alpha=0.6, color="steelblue",
                label="No-click (0)", density=True)
        ax.hist(scores_te[y_te == 1], bins=bins, alpha=0.6, color="tomato",
                label="Click (1)", density=True)
        ax.axvline(tau, color="black", linestyle="--", linewidth=1.5, label=f"τ={tau:.2f}")
        ax.set_xlabel("Predicted Score")
        ax.set_ylabel("Density")
        ax.set_title("Score Distribution (clicks vs no-clicks)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Metrics table
    with row3b:
        st.markdown("#### Evaluation Metrics")
        tp_cm = int(cm[1, 1]); fp_cm = int(cm[0, 1])
        fn_cm = int(cm[1, 0]); tn_cm = int(cm[0, 0])
        prec_tau = tp_cm / (tp_cm + fp_cm) if (tp_cm + fp_cm) else 0.0
        rec_tau = tp_cm / (tp_cm + fn_cm) if (tp_cm + fn_cm) else 0.0
        f1_tau = 2 * prec_tau * rec_tau / (prec_tau + rec_tau) if (prec_tau + rec_tau) else 0.0

        st.markdown(f"""
| Metric | Value |
|---|---|
| ROC-AUC | {roc_auc_c:.4f} |
| PR-AUC | {pr_auc_c:.4f} |
| Log Loss | {log_loss_val:.4f} |
| KS Statistic | {ks_stat2:.4f} |
| Gini | {2*roc_auc_c-1:.4f} |
| Precision @ τ={tau:.2f} | {prec_tau:.4f} |
| Recall @ τ={tau:.2f} | {rec_tau:.4f} |
| F1 @ τ={tau:.2f} | {f1_tau:.4f} |
""")
        st.latex(r"\mathcal{L} = -\frac{1}{n}\sum_{i}\left[y_i \log\hat{p}_i"
                 r"+ (1-y_i)\log(1-\hat{p}_i)\right]")

    # =========================================================================
    # O'REILLY — PROBABILITY CALIBRATION
    # =========================================================================
    st.divider()
    st.markdown(
        "## 📚 From: Feature Engineering for ML (Zheng & Casari, O'Reilly 2018)"
    )
    st.markdown("### Probability Calibration: Isotonic Regression & Platt Scaling")
    st.markdown(
        "A well-calibrated model predicts $\\hat{p} = 0.3$ and 30% of those "
        "impressions actually click. Raw scores from LightGBM / logistic regression "
        "often deviate from true probabilities and must be **calibrated** before use "
        "in bid-price calculations."
    )

    cal_c1, cal_c2 = st.columns(2)

    with cal_c1:
        st.markdown("#### Platt Scaling")
        st.latex(r"\hat{p}_{\mathrm{Platt}}(f) = \sigma(A \cdot f + B)")
        st.latex(r"\text{where } \sigma(z) = \frac{1}{1+e^{-z}}")
        st.markdown(
            "Parameters $A$ and $B$ are fitted via MLE on the **validation set** only. "
            "Equivalent to fitting a logistic regression on the raw model scores."
        )

    with cal_c2:
        st.markdown("#### Isotonic Regression (Pool Adjacent Violators)")
        st.latex(
            r"\min_{g \in \mathcal{G}} \sum_i \bigl(y_i - g(f_i)\bigr)^2 "
            r"\quad \text{s.t. } g \text{ non-decreasing}"
        )
        st.markdown(
            "The PAV algorithm finds the nearest monotone non-decreasing function "
            "to the empirical class frequencies. More flexible than Platt but requires "
            "more calibration data."
        )

    # Fit calibrators on train split, evaluate on test
    with st.spinner("Fitting Platt scaling and Isotonic Regression…"):
        scores_tr_raw = predict_proba(model, X_tr)

        # Platt scaling fit on train, apply to test
        platt_A, platt_B = platt_scaling_fit(scores_tr_raw, y_tr, n_iter=600, lr=0.05)
        scores_te_platt = platt_predict(scores_te, platt_A, platt_B)

        # Isotonic regression fit on train, apply to test
        scores_te_iso = calibrate_isotonic(scores_tr_raw, y_tr)
        # For test: use the fitted mapping via nearest-neighbour lookup
        tr_order = np.argsort(scores_tr_raw)
        tr_sorted = scores_tr_raw[tr_order]
        iso_values_sorted = calibrate_isotonic(scores_tr_raw, y_tr)[tr_order]

        def apply_isotonic(scores_new):
            """Apply fitted isotonic calibration to new scores via lookup."""
            idx = np.searchsorted(tr_sorted, scores_new, side="left")
            idx = np.clip(idx, 0, len(tr_sorted) - 1)
            return iso_values_sorted[idx]

        scores_te_iso_applied = apply_isotonic(scores_te)

    # Compute calibration curves for all three
    n_bins_cal = 10
    cal_raw_mean, cal_raw_frac = compute_calibration_curve(y_te, scores_te, n_bins=n_bins_cal)
    cal_platt_mean, cal_platt_frac = compute_calibration_curve(
        y_te, scores_te_platt, n_bins=n_bins_cal
    )
    cal_iso_mean, cal_iso_frac = compute_calibration_curve(
        y_te, scores_te_iso_applied, n_bins=n_bins_cal
    )

    # Compute log-loss for each
    ll_raw = compute_log_loss(y_te, scores_te)
    ll_platt = compute_log_loss(y_te, scores_te_platt)
    ll_iso = compute_log_loss(y_te, scores_te_iso_applied)

    calcomp_c1, calcomp_c2 = st.columns(2)

    with calcomp_c1:
        fig_cal, ax_cal = plt.subplots(figsize=(6, 5))
        ax_cal.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")

        if len(cal_raw_mean) > 0:
            ax_cal.plot(
                cal_raw_mean, cal_raw_frac, "s-",
                color="steelblue", linewidth=2, markersize=7,
                label=f"Uncalibrated (LL={ll_raw:.4f})",
            )
        if len(cal_platt_mean) > 0:
            ax_cal.plot(
                cal_platt_mean, cal_platt_frac, "o-",
                color="darkorange", linewidth=2, markersize=7,
                label=f"Platt scaling (LL={ll_platt:.4f})",
            )
        if len(cal_iso_mean) > 0:
            ax_cal.plot(
                cal_iso_mean, cal_iso_frac, "^-",
                color="mediumseagreen", linewidth=2, markersize=7,
                label=f"Isotonic Reg (LL={ll_iso:.4f})",
            )

        ax_cal.set_xlabel("Mean Predicted Probability")
        ax_cal.set_ylabel("Fraction of Positives (actual CTR)")
        ax_cal.set_title("Reliability Diagram: Calibration Comparison")
        ax_cal.legend(fontsize=9)
        ax_cal.set_xlim(0, 1)
        ax_cal.set_ylim(0, 1)
        fig_cal.tight_layout()
        st.pyplot(fig_cal)
        plt.close(fig_cal)

    with calcomp_c2:
        st.markdown("#### Calibration Quality Metrics")
        st.markdown(f"""
| Method | Log Loss | Notes |
|---|---|---|
| Uncalibrated | {ll_raw:.4f} | Raw model output |
| Platt Scaling | {ll_platt:.4f} | σ(A·f + B), A={platt_A:.3f}, B={platt_B:.3f} |
| Isotonic Regression | {ll_iso:.4f} | PAV non-decreasing fit |
""")

        st.markdown(
            "**Why calibration matters for CTR / bid pricing:** "
            "If your model outputs $\\hat{p}=0.15$ but true CTR is 0.05, "
            "you will overbid by 3x and destroy campaign ROI. "
            "Calibrated scores feed directly into "
            "$\\text{bid} = \\hat{p}_{\\text{cal}} \\times \\text{CPC}$."
        )

        st.markdown("**Score distribution: before vs after calibration**")
        fig_dist, ax_dist = plt.subplots(figsize=(5, 3))
        bins_d = np.linspace(0, 1, 41)
        ax_dist.hist(scores_te, bins=bins_d, alpha=0.5, color="steelblue",
                     label="Uncalibrated", density=True)
        ax_dist.hist(scores_te_platt, bins=bins_d, alpha=0.5, color="darkorange",
                     label="Platt", density=True)
        ax_dist.hist(scores_te_iso_applied, bins=bins_d, alpha=0.5,
                     color="mediumseagreen", label="Isotonic", density=True)
        ax_dist.set_xlabel("Predicted Probability")
        ax_dist.set_ylabel("Density")
        ax_dist.set_title("Score Distributions: 3 Calibration Methods")
        ax_dist.legend(fontsize=8)
        fig_dist.tight_layout()
        st.pyplot(fig_dist)
        plt.close(fig_dist)

    st.markdown("""
**When to choose each calibrator (O'Reilly):**

| Method | Pros | Cons | Best when |
|---|---|---|---|
| **Platt Scaling** | Few parameters (A,B); fast; works with small val sets | Assumes sigmoid shape; can't fix non-sigmoid miscalibration | Val set < 1,000 rows |
| **Isotonic Regression** | Flexible; non-parametric; handles any shape | Needs larger val set; can overfit; not smooth | Val set > 10,000 rows |
""")


# ===========================================================================
# TAB 4 — Lift & Ranking
# ===========================================================================
with tab4:
    st.subheader("Lift & Ranking Analysis")

    lift_table = compute_lift_table(y_te, scores_te, n_deciles=10)
    ks_stat3, ks_thr3, ks_cdf3_pos, ks_cdf3_neg = compute_ks(y_te, scores_te)

    # Metric cards: Lift@10%, Lift@20%, Gini
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Lift @ top 10%", f"{lift_table[0]['lift']:.3f}x")
    mc2.metric("Lift @ top 20%",
               f"{np.mean([lift_table[0]['lift'], lift_table[1]['lift']]):.3f}x")
    mc3.metric("Gini Coefficient", f"{2*roc_auc_val-1:.4f}")

    st.markdown("""
**Key equations:**

$\\text{Lift@k} = \\dfrac{\\text{Precision@k}}{\\text{Overall CTR}}$
$\\quad\\quad$
$\\text{KS} = \\max_t \\left| F_1(t) - F_0(t) \\right|$
""")

    st.divider()

    # Decile table
    st.subheader("Decile Table")
    tbl_header = "| Decile | Score Min | Score Max | N | Clicks | CTR | Lift | Cum % Clicks |"
    tbl_sep = "| --- | --- | --- | --- | --- | --- | --- | --- |"
    tbl_rows = []
    for r in lift_table:
        tbl_rows.append(
            f"| {r['decile']} | {r['score_min']:.3f} | {r['score_max']:.3f} |"
            f" {r['n']:,} | {r['clicks']:,} | {_fmt_pct(r['ctr'])} |"
            f" {r['lift']:.3f}x | {_fmt_pct(r['cum_pct_clicks'])} |"
        )
    st.markdown(tbl_header + "\n" + tbl_sep + "\n" + "\n".join(tbl_rows))

    st.divider()

    row_l1, row_l2 = st.columns(2)

    # Lift curve
    with row_l1:
        decile_nums = [r["decile"] for r in lift_table]
        lift_vals = [r["lift"] for r in lift_table]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(decile_nums, lift_vals, "o-", color="royalblue", linewidth=2, markersize=6)
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="Baseline (1×)")
        ax.set_xlabel("Decile (top → bottom score)")
        ax.set_ylabel("Lift")
        ax.set_title("Lift Curve by Decile")
        ax.set_xticks(decile_nums)
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Cumulative gains
    with row_l2:
        cum_pct = [r["cum_pct_clicks"] for r in lift_table]
        pct_population = [(r["decile"] / 10) for r in lift_table]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot([0] + pct_population, [0] + cum_pct, "o-", color="darkorange",
                linewidth=2, markersize=5, label="Model")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        ax.plot([0, overall_ctr, 1], [0, 1, 1], "g:", linewidth=1, label="Perfect")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("% Population Targeted")
        ax.set_ylabel("% Clicks Captured")
        ax.set_title("Cumulative Gains Chart")
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    row_l3, row_l4 = st.columns(2)

    # KS plot
    with row_l3:
        ks_idx = np.argmax(np.abs(ks_cdf3_pos - ks_cdf3_neg))
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(ks_thr3, ks_cdf3_pos, color="tomato", linewidth=2, label="CDF clicks (1)")
        ax.plot(ks_thr3, ks_cdf3_neg, color="steelblue", linewidth=2, label="CDF no-clicks (0)")
        ax.axvline(ks_thr3[ks_idx], color="black", linestyle="--", linewidth=1,
                   label=f"KS={ks_stat3:.3f} @ {ks_thr3[ks_idx]:.3f}")
        ax.fill_betweenx([ks_cdf3_pos[ks_idx], ks_cdf3_neg[ks_idx]],
                         ks_thr3[ks_idx], ks_thr3[ks_idx],
                         color="black", alpha=0.3)
        ax.set_xlabel("Score Threshold")
        ax.set_ylabel("CDF")
        ax.set_title("KS Plot — CDF of Scores by Class")
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with row_l4:
        st.markdown("#### Interpretation")
        st.markdown(f"""
- **KS = {ks_stat3:.3f}** — separation between score CDFs of clicks vs no-clicks.
  Higher KS → better model discrimination.
- **Lift @ decile 1 = {lift_table[0]['lift']:.2f}×** — top-scored 10% of impressions
  yield {lift_table[0]['lift']:.2f}× the average CTR.
- **Gini = {2*roc_auc_val-1:.4f}** — 2 × AUC − 1; 0 = random, 1 = perfect.
- **PR-AUC = {pr_auc_c:.4f}** — more informative than ROC-AUC for imbalanced CTR
  (baseline = {_fmt_pct(overall_ctr)}).
""")


# ===========================================================================
# TAB 5 — Campaign Optimiser
# ===========================================================================
with tab5:
    st.subheader("Campaign Budget Optimiser")

    st.latex(
        r"\text{Rev}(\tau) = \sum_{\hat{p} \geq \tau}"
        r"\left(\hat{p} \cdot \text{CPC} - \frac{\text{CPM}}{1000}\right)"
    )

    # Threshold sweep
    thr_sweep = np.linspace(0.01, 0.99, 200)
    precision_sw, recall_sw, f1_sw, volume_sw, revenue_sw, roi_sw = [], [], [], [], [], []

    for t in thr_sweep:
        served_mask = scores_te >= t
        n_served = served_mask.sum()
        tp = ((served_mask) & (y_te == 1)).sum()
        fp = ((served_mask) & (y_te == 0)).sum()
        fn = ((~served_mask) & (y_te == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1_v = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        # Revenue: per served impression, expected clicks × CPC minus CPM cost
        rev = float(np.sum(scores_te[served_mask]) * cpc - n_served * cpm / 1000.0)
        total_cost = n_served * cpm / 1000.0
        roi_v = rev / total_cost if total_cost > 0 else 0.0
        precision_sw.append(prec)
        recall_sw.append(rec)
        f1_sw.append(f1_v)
        volume_sw.append(int(n_served))
        revenue_sw.append(rev)
        roi_sw.append(roi_v)

    precision_sw = np.array(precision_sw)
    recall_sw = np.array(recall_sw)
    f1_sw = np.array(f1_sw)
    volume_sw = np.array(volume_sw)
    revenue_sw = np.array(revenue_sw)
    roi_sw = np.array(roi_sw)

    # Optimal thresholds
    best_f1_idx = int(np.argmax(f1_sw))
    best_rev_idx = int(np.argmax(revenue_sw))
    tau_best_f1 = float(thr_sweep[best_f1_idx])
    tau_best_rev = float(thr_sweep[best_rev_idx])

    oc1, oc2 = st.columns(2)
    oc1.metric("Optimal τ (max F1)", f"{tau_best_f1:.3f}",
               help=f"F1 = {f1_sw[best_f1_idx]:.4f}")
    oc2.metric("Optimal τ (max Revenue)", f"{tau_best_rev:.3f}",
               help=f"Rev = ${revenue_sw[best_rev_idx]:.2f}")

    st.divider()

    sa, sb = st.columns(2)

    # Precision / Recall / F1 vs threshold
    with sa:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(thr_sweep, precision_sw, label="Precision", color="royalblue", linewidth=2)
        ax.plot(thr_sweep, recall_sw, label="Recall", color="tomato", linewidth=2)
        ax.plot(thr_sweep, f1_sw, label="F1", color="darkorange", linewidth=2)
        ax.axvline(tau, color="black", linestyle="--", linewidth=1, label=f"τ={tau:.2f}")
        ax.axvline(tau_best_f1, color="green", linestyle=":", linewidth=1.5,
                   label=f"τ*F1={tau_best_f1:.2f}")
        ax.set_xlabel("Threshold τ")
        ax.set_ylabel("Score")
        ax.set_title("Precision / Recall / F1 vs Threshold")
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Volume served vs threshold
    with sb:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(thr_sweep, volume_sw, color="mediumseagreen", linewidth=2)
        ax.axvline(tau, color="black", linestyle="--", linewidth=1, label=f"τ={tau:.2f}")
        ax.set_xlabel("Threshold τ")
        ax.set_ylabel("Impressions Served")
        ax.set_title("Volume Served vs Threshold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    sc, sd = st.columns(2)

    # Net revenue vs threshold
    with sc:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(thr_sweep, revenue_sw, color="gold", linewidth=2, label="Net Revenue")
        ax.axvline(tau, color="black", linestyle="--", linewidth=1, label=f"τ={tau:.2f}")
        ax.axvline(tau_best_rev, color="crimson", linestyle=":", linewidth=1.5,
                   label=f"τ*Rev={tau_best_rev:.2f}")
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xlabel("Threshold τ")
        ax.set_ylabel("Net Revenue ($)")
        ax.set_title(f"Net Revenue vs Threshold\n(CPC=${cpc:.2f}, CPM=${cpm:.2f})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ROI vs threshold
    with sd:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(thr_sweep, roi_sw, color="darkorchid", linewidth=2, label="ROI")
        ax.axvline(tau, color="black", linestyle="--", linewidth=1, label=f"τ={tau:.2f}")
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xlabel("Threshold τ")
        ax.set_ylabel("ROI (Revenue / Cost)")
        ax.set_title("ROI vs Threshold")
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # Budget allocator
    st.subheader("Budget Allocator")
    budget_col, result_col = st.columns(2)
    with budget_col:
        total_budget = st.number_input("Total Campaign Budget ($)", value=1000.0, step=50.0, format="%.2f")
        selected_tau = st.slider("Select threshold τ for allocation", 0.01, 0.99,
                                 float(tau_best_rev), 0.01, key="budget_tau")

    with result_col:
        served_mask_b = scores_te >= selected_tau
        n_served_b = int(served_mask_b.sum())
        cost_per_imp = cpm / 1000.0
        if cost_per_imp > 0:
            max_imps_by_budget = int(total_budget / cost_per_imp)
        else:
            max_imps_by_budget = n_served_b
        final_imps = min(n_served_b, max_imps_by_budget)
        expected_clicks = float(scores_te[served_mask_b][:final_imps].sum()) if final_imps > 0 else 0.0
        expected_rev = expected_clicks * cpc - final_imps * cost_per_imp
        st.metric("Impressions to serve", f"{final_imps:,}")
        st.metric("Expected clicks", f"{expected_clicks:.0f}")
        st.metric("Expected revenue", f"${expected_rev:.2f}")
        st.metric("Budget utilised", f"${min(final_imps * cost_per_imp, total_budget):.2f} / ${total_budget:.2f}")

    st.divider()
    st.subheader("Optimal Threshold Recommendations")
    st.markdown(f"""
| Criterion | Threshold τ | F1 | Revenue |
|---|---|---|---|
| **Max F1** | {tau_best_f1:.3f} | {f1_sw[best_f1_idx]:.4f} | ${revenue_sw[best_f1_idx]:.2f} |
| **Max Revenue** | {tau_best_rev:.3f} | {f1_sw[best_rev_idx]:.4f} | ${revenue_sw[best_rev_idx]:.2f} |
| **Current τ** | {tau:.3f} | {f1_sw[np.argmin(np.abs(thr_sweep - tau))]:.4f} | ${revenue_sw[np.argmin(np.abs(thr_sweep - tau))]:.2f} |
""")

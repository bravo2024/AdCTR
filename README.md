# AdCTR

> Click-through rate prediction with feature engineering from the Avazu CTR Challenge schema.

Trains a LightGBM classifier on a synthetic dataset mirroring the 40M-row Avazu ad exchange schema, then surfaces model diagnostics, feature engineering walkthroughs, and a campaign budget simulator in a Streamlit dashboard.

Built-in synthetic data generator means everything runs immediately with no downloads.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

LightGBM with 300 estimators, balanced class weighting, trained on 3,000 synthetic impressions:

| Metric | Value |
|---|---|
| Backend | LightGBM |
| ROC AUC | 0.849 |
| Accuracy | 0.791 |
| F1 Score | 0.688 |
| Training samples | 3,000 |
| Test samples | 1,000 |

When LightGBM is not installed, a pure-NumPy logistic regression baseline is used instead.

## Features

| Tab | What it does |
|---|---|
| **Data Explorer** | Class distribution, CTR breakdowns by banner position / device type / hour, feature statistics |
| **Feature Engineering** | Frequency encoding, cyclic hour encoding, Laplace-smoothed bin counting, cross-feature interactions, FM interaction matrix |
| **Model Evaluation** | ROC/PR curves, confusion matrix, KS statistic, calibration plots (Platt scaling + isotonic regression) |
| **Lift & Ranking** | Decile lift table, cumulative clicks chart, Gini coefficient |
| **Campaign Optimiser** | CPC/CPM configuration, profit curve, optimal threshold finder |

## Repo Structure

```
AdCTR/
  src/         data, model, evaluate, persist modules
  train.py     training pipeline
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

The included synthetic data mirrors the Avazu CTR Challenge (Kaggle 2014) schema — 23 features including banner position, device type, hour, site/app category, and anonymised categoricals. The app also supports a realistic 60k-row variant that matches real Avazu value ranges and distributions.

For the real dataset: [Avazu CTR Prediction](https://www.kaggle.com/c/avazu-ctr-prediction) (40M training rows, 2014).

## License

MIT

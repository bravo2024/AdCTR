# AdCTR

Click-through-rate prediction on a synthetic ad-impression log: 20,000 rows,
~3.3% base CTR, with the click probability built from feature *interactions*
(ad-category matching the user's interest segment, position on the page,
device, time-of-day peaks) rather than one flat additive score — closer to
how real ad logs behave than a toy linear dataset would be.

Two models, trained on the same one-hot-encoded split:

- **Logistic regression via mini-batch SGD, written from scratch**
  (`SGDRidgeLogistic` in `src/model.py`) — the sparse-LR style a lot of real
  ad-serving pipelines still run in production (McMahan et al. 2013, He et
  al. 2014).
- **GBDT via LightGBM** — the usual non-linear baseline from the same
  literature.

Both are scored with the metrics people actually use for this problem
instead of plain accuracy: log loss, **normalized entropy (NE)**, **AUC**,
and **lift@k**. At a 3% base rate, accuracy tells you almost nothing; NE and
lift@k tell you whether the ranking is any good and whether the raw
probabilities are usable for bidding.

## What the numbers say (25% holdout)

| Model | AUC | NE | lift@5% | lift@10% |
|---|---|---|---|---|
| SGD Logistic | 0.635 | **0.98** | 2.16× | 1.80× |
| GBDT | 0.624 | 2.42 | 2.04× | 1.86× |

3-fold CV AUC comes out around 0.61 for both models, so the holdout AUC gap
above is mostly split variance, not one model genuinely outranking the
other.

The number worth actually looking at is NE, not AUC. GBDT ranks about as
well as the logistic model, but its `scale_pos_weight` correction (added to
compensate for the ~3% positive rate) wrecks its calibration — NE above 1.0
means it's doing worse than just predicting the base rate everywhere. The
logistic model's output probabilities stay usable. If the score has to feed
a bid price directly, rather than just order impressions, the model with
the "worse" AUC is the one you'd actually ship. That's the point of scoring
on NE instead of AUC alone: a model can look fine on ranking and still be
useless once you need calibrated probabilities.

These numbers come straight from `models/metrics.json` (written by
`train.py`), not hand-typed.

## Known issues

- **`app.py` does not run.** It imports `make_realistic_ctr`,
  `fit_and_evaluate`, and `predict_proba` from `src.data` / `src.model`, and
  `train_test_split`, `Standardizer`, `LogisticRegression`, `roc_auc_score`,
  and `sigmoid` from `src.core` — none of which exist in the current
  `src/` (which only exposes `make_synthetic`, `train_all_models`,
  `cross_validate`, and the from-scratch metric functions in `core.py`).
  `app.py`'s own docstring describes an Avazu-dataset / O'Reilly-book
  version of this project (five tabs: Data Explorer, Feature Engineering,
  Model Evaluation, Lift & Ranking, Campaign Optimiser) that `src/` has
  since been rewritten away from, and the dashboard was never brought back
  in sync. `streamlit run app.py` fails on the import line before any page
  renders. Making it work again means either rebuilding the dashboard
  against the current one-hot + SGD/LightGBM pipeline, or restoring the
  older `src/` API it expects — bigger than a README/requirements pass, so
  it's left broken and flagged here instead of half-patched.
- `src/evaluate.py` (`save_metrics` / `print_report`) isn't imported by
  `train.py`, `app.py`, or the tests. `train.py` writes
  `models/metrics.json` itself inline. Looks like leftover code from before
  that got inlined.
- The module docstring in `src/data.py` says the base CTR is "~2%"; the
  generator actually produces ~3.3% (see `models/metrics.json`, and
  `test_data` in `tests/test_smoke.py`, which asserts
  `0.005 < ctr < 0.05`). Harmless, just a stale comment.

## Running it

```bash
pip install -r requirements.txt
python train.py    # builds the synthetic data, trains both models, prints the metric table
pytest -q
```

`train.py` generates data, trains, evaluates, and writes
`models/model.pkl` + `models/metrics.json`. That, and the test suite, are
the parts of this repo that currently work end to end — see "Known issues"
for the dashboard.

## Layout

```
src/data.py     synthetic ad-impression generator (pandas + numpy, nothing downloaded)
src/model.py    one-hot encoding, SGD logistic regression, LightGBM GBDT, train/CV glue
src/core.py     log loss, normalized entropy, AUC, calibration curve, lift@k — all from scratch
src/persist.py  pickle save/load for the trained GBDT
train.py        CLI entrypoint: trains, evaluates, writes models/model.pkl + models/metrics.json
app.py          Streamlit dashboard (currently broken — see Known issues)
```

## License

MIT

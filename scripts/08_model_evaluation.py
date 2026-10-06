"""
08_model_evaluation.py — Primary model evaluation.

Four validation schemes:
  A. Random StratifiedKFold (5-fold)
  B. MLST GroupKFold (5-fold)
  C. Leave-One-Region-Out
  D. Temporal (train <2016, test ≥2016)

Two models:
  - Logistic regression (primary, interpretable)
  - XGBoost (secondary, robustness)

No feature selection. All 377 features retained throughout.
"""

from __future__ import annotations
import json
import warnings
from datetime import datetime, timezone
from importlib.machinery import SourceFileLoader
from pathlib import Path as _P

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedKFold, GroupKFold)
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, f1_score, brier_score_loss)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("08_model_evaluation")

RANDOM_SEED = setup.RANDOM_SEED
LR_PARAMS   = dict(max_iter=5000, C=1.0, solver="lbfgs", random_state=RANDOM_SEED)
XGB_PARAMS  = dict(n_estimators=300, max_depth=4, learning_rate=0.1,
                   subsample=0.8, colsample_bytree=0.8,
                   random_state=RANDOM_SEED, eval_metric="logloss",
                   use_label_encoder=False, verbosity=0)


# ── Load analysis table ──────────────────────────────────────────────────
table = pd.read_csv(setup.PROCESSED / "06_analysis_table.csv",
                    low_memory=False, dtype={"genome_id": str})

META_COLS = {"genome_id", "label", "region", "region_sensitivity",
             "isolation_country", "sequencing_center", "mlst",
             "genome_quality", "collection_year", "method_tier",
             "has_annotation"}
feature_cols = [c for c in table.columns if c not in META_COLS]

X = table[feature_cols].values.astype("float32")
y = table["label"].values.astype(int)

log.info(f"Feature matrix: {X.shape}")
log.info(f"Resistant: {y.sum()} ({y.mean():.1%})")
log.info(f"Features: {len(feature_cols)}")


# ── Metrics helper ───────────────────────────────────────────────────────
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    m = {
        "n":         int(len(y_true)),
        "positive":  int(y_true.sum()),
        "negative":  int((y_true == 0).sum()),
        "prevalence": float(y_true.mean()),
    }
    if len(np.unique(y_true)) > 1:
        m["auroc"] = float(roc_auc_score(y_true, y_prob))
        m["auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        m["auroc"] = np.nan
        m["auprc"] = np.nan
    m["accuracy"]    = float(accuracy_score(y_true, y_pred))
    m["f1"]          = float(f1_score(y_true, y_pred, zero_division=0))
    m["sensitivity"] = float(((y_pred == 1) & (y_true == 1)).sum() /
                             max((y_true == 1).sum(), 1))
    m["specificity"] = float(((y_pred == 0) & (y_true == 0)).sum() /
                             max((y_true == 0).sum(), 1))
    m["brier"]       = float(brier_score_loss(y_true, y_prob))
    return m


# ── CV schemes ───────────────────────────────────────────────────────────
def cv_random(X, y, model_factory, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=RANDOM_SEED)
    aucs, auprcs, accs, f1s = [], [], [], []
    probs = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        m = model_factory()
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        probs[te] = p
        mm = compute_metrics(y[te], p)
        aucs.append(mm["auroc"]); auprcs.append(mm["auprc"])
        accs.append(mm["accuracy"]); f1s.append(mm["f1"])
    return {
        "auroc_mean":   float(np.mean(aucs)), "auroc_sd":  float(np.std(aucs)),
        "auprc_mean":   float(np.mean(auprcs)), "auprc_sd": float(np.std(auprcs)),
        "accuracy_mean": float(np.mean(accs)),
        "f1_mean":      float(np.mean(f1s)),
        "pooled_metrics": compute_metrics(y, probs),
        "fold_aurocs":  [round(a, 4) for a in aucs],
    }


def cv_mlst(X, y, mlst, model_factory, n_splits=5):
    mlst = pd.Series(mlst).fillna("UNKNOWN").astype(str).values
    # GroupKFold needs at least n_splits unique groups
    n_groups = len(set(mlst))
    k = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=k)
    aucs, auprcs = [], []
    probs = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups=mlst):
        m = model_factory()
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        probs[te] = p
        if len(np.unique(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], p))
            auprcs.append(average_precision_score(y[te], p))
    return {
        "n_splits":  k,
        "auroc_mean": float(np.mean(aucs)), "auroc_sd": float(np.std(aucs)),
        "auprc_mean": float(np.mean(auprcs)),
        "fold_aurocs": [round(a, 4) for a in aucs],
    }


def cv_leave_one_region(X, y, regions, model_factory):
    results = []
    for region in sorted(set(regions)):
        test_mask = np.array([r == region for r in regions])
        train_mask = ~test_mask
        if test_mask.sum() < 10 or len(np.unique(y[test_mask])) < 2:
            continue
        m = model_factory()
        m.fit(X[train_mask], y[train_mask])
        p = m.predict_proba(X[test_mask])[:, 1]
        mm = compute_metrics(y[test_mask], p)
        mm["region"]      = region
        mm["n_train"]     = int(train_mask.sum())
        mm["n_test"]      = int(test_mask.sum())
        mm["test_prev"]   = float(y[test_mask].mean())
        mm["train_prev"]  = float(y[train_mask].mean())
        results.append(mm)
    return pd.DataFrame(results).sort_values("auroc", ascending=False)


def cv_temporal(X, y, years, model_factory, cutoff=2016):
    years = pd.Series(years)
    mask_train = years < cutoff
    mask_test  = years >= cutoff
    n_skip = int(years.isna().sum())
    if mask_test.sum() < 10:
        return {"error": "not enough test samples ≥ cutoff"}
    m = model_factory()
    m.fit(X[mask_train.values], y[mask_train.values])
    p = m.predict_proba(X[mask_test.values])[:, 1]
    mm = compute_metrics(y[mask_test.values], p)
    mm.update({
        "cutoff":          cutoff,
        "n_train":         int(mask_train.sum()),
        "n_test":          int(mask_test.sum()),
        "n_excluded_year": n_skip,
        "train_prev":      float(y[mask_train.values].mean()),
        "test_prev":       float(y[mask_test.values].mean()),
    })
    return mm


# ── Run all schemes for both models ──────────────────────────────────────
results = {}

for model_name, factory in [
    ("logistic_regression", lambda: LogisticRegression(**LR_PARAMS)),
    ("xgboost",             lambda: XGBClassifier(**XGB_PARAMS)),
]:
    log.info(f"\n{'='*70}\n{model_name}\n{'='*70}")

    log.info("  Random 5-fold CV...")
    results[f"{model_name}__random"] = cv_random(X, y, factory)

    log.info("  MLST GroupKFold...")
    results[f"{model_name}__mlst"] = cv_mlst(X, y, table["mlst"].values, factory)

    log.info("  Leave-One-Region-Out...")
    loo_df = cv_leave_one_region(X, y, table["region"].values, factory)
    loo_df.to_csv(setup.RESULTS / f"table5_region_holdout__{model_name}.csv",
                  index=False)
    results[f"{model_name}__loo_region"] = loo_df.to_dict("records")
    log.info(f"    saved table5_region_holdout__{model_name}.csv")

    log.info("  Temporal holdout (pre-2016 vs post-2016)...")
    results[f"{model_name}__temporal"] = cv_temporal(
        X, y, table["collection_year"].values, factory, cutoff=2016)

# ── Summary table ────────────────────────────────────────────────────────
summary_rows = []
for key, val in results.items():
    if key.endswith("__random"):
        summary_rows.append({
            "model": key.replace("__random", ""),
            "scheme": "Random 5-fold",
            "auroc": val["auroc_mean"],
            "auroc_sd": val["auroc_sd"],
            "auprc": val["auprc_mean"],
            "accuracy": val["accuracy_mean"],
            "f1": val["f1_mean"],
        })
    elif key.endswith("__mlst"):
        summary_rows.append({
            "model": key.replace("__mlst", ""),
            "scheme": "MLST GroupKFold",
            "auroc": val["auroc_mean"],
            "auroc_sd": val["auroc_sd"],
            "auprc": val["auprc_mean"],
            "accuracy": np.nan,
            "f1": np.nan,
        })
    elif key.endswith("__temporal") and isinstance(val, dict) and "auroc" in val:
        summary_rows.append({
            "model": key.replace("__temporal", ""),
            "scheme": "Temporal ≥2016",
            "auroc": val["auroc"],
            "auroc_sd": np.nan,
            "auprc": val["auprc"],
            "accuracy": val["accuracy"],
            "f1": val["f1"],
        })

summary_df = pd.DataFrame(summary_rows).sort_values(["model", "scheme"])
summary_df.to_csv(setup.RESULTS / "table4_model_performance.csv", index=False)
log.info("\n" + summary_df.round(4).to_string(index=False))


# ── Save raw results ─────────────────────────────────────────────────────
(setup.RESULTS / "08_model_results.json").write_text(
    json.dumps(results, indent=2, default=str), encoding="utf-8")

setup.write_result("08_model_evaluation", {
    "analysis_name":     "model_evaluation",
    "dataset_version":   "api_v1",
    "n":                 int(len(y)),
    "n_features":        int(X.shape[1]),
    "results":           results,
    "timestamp":         datetime.now(timezone.utc).isoformat(),
    "software_version":  setup.software_versions(),
})
log.info("Done.")
"""
12_region_sensitivity.py — Compare primary vs sensitivity region mapping.

Primary    : Caribbean → Unknown
Sensitivity: Caribbean → North America

Reports LOO-region AUROC and AUPRC under both mappings, for both models.
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from importlib.machinery import SourceFileLoader
from pathlib import Path as _P

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("12_region_sensitivity")

RANDOM_SEED = setup.RANDOM_SEED
LR_PARAMS   = dict(max_iter=5000, C=1.0, solver="lbfgs", random_state=RANDOM_SEED)
XGB_PARAMS  = dict(n_estimators=300, max_depth=4, learning_rate=0.1,
                   subsample=0.8, colsample_bytree=0.8,
                   random_state=RANDOM_SEED, eval_metric="logloss",
                   use_label_encoder=False, verbosity=0)

# ─────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────
table = pd.read_csv(setup.PROCESSED / "06_analysis_table.csv",
                    low_memory=False, dtype={"genome_id": str})

META = {"genome_id", "label", "region", "region_sensitivity",
        "isolation_country", "sequencing_center", "mlst",
        "genome_quality", "collection_year", "method_tier", "has_annotation"}
features = [c for c in table.columns if c not in META]
X = table[features].values.astype("float32")
y = table["label"].values.astype(int)

log.info(f"N = {len(y)}, features = {X.shape[1]}")


# ─────────────────────────────────────────────────────────────────────────
# LOO-region function (parameterised by model + region column)
# ─────────────────────────────────────────────────────────────────────────
def loo_regions(region_col: str, model_name: str, model_factory):
    regions = table[region_col].values
    rows = []
    for region in sorted(set(regions)):
        test_mask  = regions == region
        train_mask = ~test_mask
        if test_mask.sum() < 10:
            continue
        if len(np.unique(y[test_mask])) < 2:
            continue

        model = model_factory()
        model.fit(X[train_mask], y[train_mask])
        p = model.predict_proba(X[test_mask])[:, 1]

        rows.append({
            "mapping":      region_col,
            "model":        model_name,
            "region":       region,
            "n_train":      int(train_mask.sum()),
            "n_test":       int(test_mask.sum()),
            "test_prev":    float(y[test_mask].mean()),
            "train_prev":   float(y[train_mask].mean()),
            "auroc":        float(roc_auc_score(y[test_mask], p)),
            "auprc":        float(average_precision_score(y[test_mask], p)),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────
# Run all combinations
# ─────────────────────────────────────────────────────────────────────────
models = {
    "logistic_regression": lambda: LogisticRegression(**LR_PARAMS),
    "xgboost":             lambda: XGBClassifier(**XGB_PARAMS),
}

all_results = []
for mapping in ("region", "region_sensitivity"):
    for model_name, factory in models.items():
        log.info(f"Mapping={mapping:20s}  Model={model_name}")
        df = loo_regions(mapping, model_name, factory)
        all_results.append(df)

combined = pd.concat(all_results, ignore_index=True)
combined.to_csv(setup.RESULTS / "tableS5_loo_region_mappings.csv", index=False)
log.info(f"Saved: results/tableS5_loo_region_mappings.csv ({len(combined)} rows)")


# ─────────────────────────────────────────────────────────────────────────
# Pivot summary: primary mapping only, both models side by side
# ─────────────────────────────────────────────────────────────────────────
primary = combined[combined["mapping"] == "region"].pivot(
    index="region", columns="model", values="auroc"
).round(3)
primary.to_csv(setup.RESULTS / "tableS5_auroc_by_region_primary.csv")
log.info("\nPrimary mapping — AUROC by region and model:")
log.info("\n" + primary.to_string())

sensitivity = combined[combined["mapping"] == "region_sensitivity"].pivot(
    index="region", columns="model", values="auroc"
).round(3)
sensitivity.to_csv(setup.RESULTS / "tableS5_auroc_by_region_sensitivity.csv")
log.info("\nSensitivity mapping — AUROC by region and model:")
log.info("\n" + sensitivity.to_string())


# ─────────────────────────────────────────────────────────────────────────
# Difference table (Primary vs Sensitivity) — per region and model
# ─────────────────────────────────────────────────────────────────────────
def build_diff_table(model_name):
    p = combined[(combined["mapping"] == "region") &
                 (combined["model"] == model_name)][["region", "auroc"]]
    s = combined[(combined["mapping"] == "region_sensitivity") &
                 (combined["model"] == model_name)][["region", "auroc"]]
    diff = p.merge(s, on="region", suffixes=("_primary", "_sensitivity"))
    diff["auroc_delta"] = diff["auroc_primary"] - diff["auroc_sensitivity"]
    return diff.sort_values("auroc_primary", ascending=False)


for model_name in models.keys():
    diff = build_diff_table(model_name)
    log.info(f"\nPrimary vs Sensitivity — {model_name}:")
    log.info("\n" + diff.round(3).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────────────────
setup.write_result("12_region_sensitivity", {
    "analysis_name":     "region_mapping_sensitivity",
    "dataset_version":   "api_v1",
    "n":                 int(len(y)),
    "n_features":        int(X.shape[1]),
    "rows_written":      int(len(combined)),
    "timestamp":         datetime.now(timezone.utc).isoformat(),
    "software_version":  setup.software_versions(),
})
log.info("Done.")
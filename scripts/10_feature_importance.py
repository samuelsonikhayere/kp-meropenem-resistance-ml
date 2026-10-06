"""
10_feature_importance.py — Feature importance from both models.
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from importlib.machinery import SourceFileLoader
from pathlib import Path as _P

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("10_feature_importance")

RANDOM_SEED = setup.RANDOM_SEED
LR_PARAMS   = dict(max_iter=5000, C=1.0, solver="lbfgs", random_state=RANDOM_SEED)
XGB_PARAMS  = dict(n_estimators=300, max_depth=4, learning_rate=0.1,
                   subsample=0.8, colsample_bytree=0.8,
                   random_state=RANDOM_SEED, eval_metric="logloss",
                   use_label_encoder=False, verbosity=0)

table = pd.read_csv(setup.PROCESSED / "06_analysis_table.csv",
                    low_memory=False, dtype={"genome_id": str})

META = {"genome_id", "label", "region", "region_sensitivity",
        "isolation_country", "sequencing_center", "mlst",
        "genome_quality", "collection_year", "method_tier", "has_annotation"}
features = [c for c in table.columns if c not in META]
X = table[features].values.astype("float32")
y = table["label"].values.astype(int)

# ── Logistic regression coefficients ─────────────────────────────────────
log.info("Fitting logistic regression on full annotated cohort...")
lr = LogisticRegression(**LR_PARAMS)
lr.fit(X, y)
lr_imp = pd.DataFrame({
    "feature":       features,
    "coefficient":   lr.coef_[0],
    "abs_coefficient": np.abs(lr.coef_[0]),
}).sort_values("abs_coefficient", ascending=False)

lr_imp.to_csv(setup.RESULTS / "table7_features_logreg.csv", index=False)
log.info("Saved: results/table7_features_logreg.csv")
log.info("\nTop 20 by |coefficient|:")
log.info("\n" + lr_imp.head(20).to_string(index=False))

# ── XGBoost importances ──────────────────────────────────────────────────
log.info("\nFitting XGBoost on full annotated cohort...")
xgb = XGBClassifier(**XGB_PARAMS)
xgb.fit(X, y)
xgb_imp = pd.DataFrame({
    "feature":     features,
    "importance":  xgb.feature_importances_,
}).sort_values("importance", ascending=False)

xgb_imp.to_csv(setup.RESULTS / "table7_features_xgboost.csv", index=False)
log.info("Saved: results/table7_features_xgboost.csv")
log.info("\nTop 20 by importance:")
log.info("\n" + xgb_imp.head(20).to_string(index=False))

setup.write_result("10_feature_importance", {
    "analysis_name":     "feature_importance",
    "dataset_version":   "api_v1",
    "n":                 int(len(y)),
    "n_features":        int(X.shape[1]),
    "logreg_top20":      lr_imp.head(20)["feature"].tolist(),
    "xgboost_top20":     xgb_imp.head(20)["feature"].tolist(),
    "timestamp":         datetime.now(timezone.utc).isoformat(),
    "software_version":  setup.software_versions(),
})
log.info("Done.")
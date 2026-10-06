"""
09_bootstrap_ci.py — Bootstrap 95% CIs for AUROC and AUPRC.

For each region: train on non-region, test on region, bootstrap the
test-set predictions 2,000 times with replacement.
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

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("09_bootstrap_ci")

RANDOM_SEED   = setup.RANDOM_SEED
N_BOOTSTRAP   = 2000
LR_PARAMS     = dict(max_iter=5000, C=1.0, solver="lbfgs", random_state=RANDOM_SEED)

table = pd.read_csv(setup.PROCESSED / "06_analysis_table.csv",
                    low_memory=False, dtype={"genome_id": str})

META = {"genome_id", "label", "region", "region_sensitivity",
        "isolation_country", "sequencing_center", "mlst",
        "genome_quality", "collection_year", "method_tier", "has_annotation"}
features = [c for c in table.columns if c not in META]
X = table[features].values.astype("float32")
y = table["label"].values.astype(int)
regions = table["region"].values


def bootstrap_ci(y_true, y_prob, n=N_BOOTSTRAP, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    aucs, auprcs = [], []
    n_samples = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, n_samples, n_samples)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        auprcs.append(average_precision_score(y_true[idx], y_prob[idx]))
    aucs = np.array(aucs); auprcs = np.array(auprcs)
    return {
        "auroc_point":  float(roc_auc_score(y_true, y_prob)),
        "auroc_mean":   float(aucs.mean()),
        "auroc_ci_lo":  float(np.percentile(aucs, 2.5)),
        "auroc_ci_hi":  float(np.percentile(aucs, 97.5)),
        "auprc_point":  float(average_precision_score(y_true, y_prob)),
        "auprc_mean":   float(auprcs.mean()),
        "auprc_ci_lo":  float(np.percentile(auprcs, 2.5)),
        "auprc_ci_hi":  float(np.percentile(auprcs, 97.5)),
        "n_bootstrap_used": int(len(aucs)),
    }


rows = []
for region in sorted(set(regions)):
    test_mask  = np.array([r == region for r in regions])
    train_mask = ~test_mask
    if test_mask.sum() < 10 or len(np.unique(y[test_mask])) < 2:
        continue

    model = LogisticRegression(**LR_PARAMS)
    model.fit(X[train_mask], y[train_mask])
    p = model.predict_proba(X[test_mask])[:, 1]

    ci = bootstrap_ci(y[test_mask], p)
    ci["region"]  = region
    ci["n_test"]  = int(test_mask.sum())
    ci["n_train"] = int(train_mask.sum())
    ci["test_prevalence"]  = float(y[test_mask].mean())
    ci["train_prevalence"] = float(y[train_mask].mean())
    rows.append(ci)

    log.info(f"{region:20s}  n={ci['n_test']:4d}  "
             f"AUROC={ci['auroc_point']:.3f} [{ci['auroc_ci_lo']:.3f}, {ci['auroc_ci_hi']:.3f}]")

df = pd.DataFrame(rows)
df.to_csv(setup.RESULTS / "tableS4_bootstrap_cis.csv", index=False)
log.info("Saved: results/tableS4_bootstrap_cis.csv")

setup.write_result("09_bootstrap_ci", {
    "analysis_name":     "bootstrap_ci",
    "dataset_version":   "api_v1",
    "n_bootstrap":       N_BOOTSTRAP,
    "results":           rows,
    "timestamp":         datetime.now(timezone.utc).isoformat(),
    "software_version":  setup.software_versions(),
})
log.info("Done.")
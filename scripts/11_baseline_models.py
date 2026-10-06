"""
11_baseline_models.py — Baseline comparisons.

  B1. Majority class
  B2. Annotation availability (has_annotation flag only)
  B3. Batch-marker products only
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from importlib.machinery import SourceFileLoader
from pathlib import Path as _P

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("11_baseline_models")

RANDOM_SEED = setup.RANDOM_SEED
LR_PARAMS   = dict(max_iter=5000, C=1.0, solver="lbfgs", random_state=RANDOM_SEED)

# ── Load both tables ─────────────────────────────────────────────────────
primary = pd.read_csv(setup.PROCESSED / "06_analysis_table.csv",
                      low_memory=False, dtype={"genome_id": str})
full    = pd.read_csv(setup.PROCESSED / "06_analysis_table_full.csv",
                      low_memory=False, dtype={"genome_id": str})

META = {"genome_id", "label", "region", "region_sensitivity",
        "isolation_country", "sequencing_center", "mlst",
        "genome_quality", "collection_year", "method_tier", "has_annotation"}

results = {}


def quick_cv_auroc(X, y, model_factory=LogisticRegression):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    aucs = []
    for tr, te in skf.split(X, y):
        if len(np.unique(y[te])) < 2:
            continue
        m = model_factory(**LR_PARAMS) if isinstance(model_factory, type) else model_factory()
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))


# ── B1: Majority class (trivially AUROC = 0.5) ───────────────────────────
results["majority_class"] = {"auroc": 0.5, "auroc_sd": 0.0}

# ── B2: Annotation availability only ─────────────────────────────────────
log.info("Baseline B2 — annotation availability")
y_full = full["label"].values.astype(int)
has_ann = full["has_annotation"].values.reshape(-1, 1)
mean, sd = quick_cv_auroc(has_ann, y_full)
results["annotation_availability"] = {"auroc": mean, "auroc_sd": sd,
                                       "n": int(len(y_full))}
log.info(f"  AUROC = {mean:.3f} ± {sd:.3f}")

# ── B3: Batch-marker products only ───────────────────────────────────────
BATCH_MARKERS = [
    "DNA gyrase subunit A (EC 5.99.1.3)",
    "DNA-directed RNA polymerase beta subunit (EC 2.7.7.6)",
    "DNA-directed RNA polymerase beta' subunit (EC 2.7.7.6)",
    "Transcription termination factor Rho",
    "Translation elongation factor Tu",
    "Multidrug efflux pump EmrD (of MFS type)",
    "Multidrug resistance regulator EmrR (MprA)",
    "Multidrug efflux system AcrAB-TolC, membrane fusion component AcrA",
    "Multiple antibiotic resistance protein MarA",
    "Multiple antibiotic resistance protein MarR",
    "Multidrug efflux system AcrAD-TolC, inner-membrane proton/drug antiporter AcrD (RND type)",
    "Dihydropteroate synthase type-2 (EC 2.5.1.15) @ Sulfonamide resistance protein",
]

X_p = primary[[c for c in primary.columns if c not in META]].values.astype("float32")
y_p = primary["label"].values.astype(int)
p_cols = [c for c in primary.columns if c not in META]

present_markers = [m for m in BATCH_MARKERS if m in p_cols]
if len(present_markers) >= 3:
    idx = [p_cols.index(m) for m in present_markers]
    X_b = X_p[:, idx]
    mean, sd = quick_cv_auroc(X_b, y_p)
    results["batch_markers"] = {"auroc": mean, "auroc_sd": sd,
                                 "n_markers": len(present_markers),
                                 "markers": present_markers}
    log.info(f"  AUROC = {mean:.3f} ± {sd:.3f}  ({len(present_markers)} markers)")

    # Africa holdout
    regions = primary["region"].values
    africa_test = regions == "Africa"
    if africa_test.sum() >= 10:
        m = LogisticRegression(**LR_PARAMS)
        m.fit(X_b[~africa_test], y_p[~africa_test])
        p = m.predict_proba(X_b[africa_test])[:, 1]
        if len(np.unique(y_p[africa_test])) > 1:
            results["batch_markers__africa_holdout"] = {
                "auroc": float(roc_auc_score(y_p[africa_test], p)),
                "n_test": int(africa_test.sum()),
            }
            log.info(f"  Africa holdout AUROC: "
                     f"{results['batch_markers__africa_holdout']['auroc']:.3f}")
else:
    results["batch_markers"] = {"error": "fewer than 3 markers present"}

# ── Save ─────────────────────────────────────────────────────────────────
df = pd.DataFrame([
    {"baseline": k,
     "auroc":    v.get("auroc"),
     "auroc_sd": v.get("auroc_sd"),
     "n":        v.get("n") or v.get("n_test")}
    for k, v in results.items()
])
df.to_csv(setup.RESULTS / "tableS3_baselines.csv", index=False)
log.info("Saved: results/tableS3_baselines.csv")
log.info("\n" + df.to_string(index=False))

setup.write_result("11_baseline_models", {
    "analysis_name":     "baseline_models",
    "dataset_version":   "api_v1",
    "results":           results,
    "timestamp":         datetime.now(timezone.utc).isoformat(),
    "software_version":  setup.software_versions(),
})
log.info("Done.")
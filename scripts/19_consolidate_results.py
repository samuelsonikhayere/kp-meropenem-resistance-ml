"""
19_consolidate_results.py — Assemble all manuscript numbers into one file.

Reads every results/ table and produces results/MANUSCRIPT_NUMBERS.md and
results/MANUSCRIPT_NUMBERS.json.
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from importlib.machinery import SourceFileLoader
from pathlib import Path as _P

import pandas as pd

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("19_consolidate_results")


def safe_read_csv(name):
    p = setup.RESULTS / name
    return pd.read_csv(p) if p.exists() else None


cascade        = safe_read_csv("table2_filter_cascade.csv")
region_primary = safe_read_csv("table3_region_distribution.csv")
model_perf     = safe_read_csv("table4_model_performance.csv")
boot_cis       = safe_read_csv("tableS4_bootstrap_cis.csv")
baselines      = safe_read_csv("tableS3_baselines.csv")
feature_summ   = safe_read_csv("tableS2_feature_matrix_summary.csv")
singlesource   = safe_read_csv("tableS1_single_source_effect.csv")
lr_loo         = safe_read_csv("table5_region_holdout__logistic_regression.csv")

with open(setup.RESULTS / "08_model_results.json", encoding="utf-8") as f:
    model_results = json.load(f)

primary = pd.read_csv(setup.PROCESSED / "06_analysis_table.csv", low_memory=False)
full    = pd.read_csv(setup.PROCESSED / "06_analysis_table_full.csv", low_memory=False)
assert len(full) == 3930
assert len(primary) == 3694
africa = primary[primary["region"] == "Africa"]
assert len(africa) == 79

feat_cols = setup.feature_columns(primary)

# ── Build markdown ───────────────────────────────────────────────────────
md = []
md.append("# Manuscript numbers — auto-generated\n")
md.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")

md.append("## Cohort")
md.append(f"- Verified cohort (analysis denominator): **{len(full)}** isolates")
md.append(f"- Annotated feature matrix (primary): **{len(primary)}** isolates")
md.append(f"- Genomes without AMR annotation: **{len(full) - len(primary)}**")
md.append(f"- African test isolates: **{len(africa)}**")
md.append(f"- Features: **{len(feat_cols)}** AMR products")
md.append("")

md.append("## Filter cascade")
if cascade is not None:
    md.append(cascade.to_markdown(index=False))
    md.append("")

md.append("## Regional distribution (primary mapping)")
if region_primary is not None:
    md.append(region_primary.to_markdown(index=False))
    md.append("")

md.append("## Model performance across validation schemes")
if model_perf is not None:
    md.append(model_perf.round(4).to_markdown(index=False))
    md.append("")

md.append("## Leave-one-region-out — logistic regression")
if lr_loo is not None:
    md.append(lr_loo.round(4).to_markdown(index=False))
    md.append("")

md.append("## Bootstrap CIs by region (LR)")
if boot_cis is not None:
    md.append(boot_cis.round(4).to_markdown(index=False))
    md.append("")

md.append("## Baselines")
if baselines is not None:
    md.append(baselines.round(4).to_markdown(index=False))
    md.append("")

md.append("## Feature matrix summary")
if feature_summ is not None:
    md.append(feature_summ.to_markdown(index=False))
    md.append("")

md.append("## Single-source submission effect")
if singlesource is not None:
    md.append(singlesource.to_markdown(index=False))
    md.append("")

md.append("## Key statistics for the abstract")
md.append(f"- Random CV AUROC (LR):  "
          f"{model_results.get('logistic_regression__random', {}).get('auroc_mean', 'NA')}")
md.append(f"- Random CV AUROC (XGB): "
          f"{model_results.get('xgboost__random', {}).get('auroc_mean', 'NA')}")
md.append(f"- MLST CV AUROC (LR):    "
          f"{model_results.get('logistic_regression__mlst', {}).get('auroc_mean', 'NA')}")
md.append(f"- MLST CV AUROC (XGB):   "
          f"{model_results.get('xgboost__mlst', {}).get('auroc_mean', 'NA')}")

md_path = setup.RESULTS / "MANUSCRIPT_NUMBERS.md"
md_path.write_text("\n".join(md), encoding="utf-8")
log.info(f"Saved: {md_path.relative_to(setup.ROOT)}")

(setup.RESULTS / "MANUSCRIPT_NUMBERS.json").write_text(
    json.dumps({
        "cohort": {"verified": len(full), "annotated": len(primary),
                   "african_test": len(africa), "features": len(feat_cols)},
        "model_results": model_results,
    }, indent=2, default=str), encoding="utf-8")
log.info("Saved: results/MANUSCRIPT_NUMBERS.json")

setup.write_result("19_consolidate_results", {
    "analysis_name":    "manuscript_consolidation",
    "cohort_size":      int(len(full)),
    "annotated":        int(len(primary)),
    "african_test":     int(len(africa)),
    "n_features":       int(len(feat_cols)),
    "timestamp":        datetime.now(timezone.utc).isoformat(),
    "software_version": setup.software_versions(),
})
log.info("Done.")
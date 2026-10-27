"""19: gather every manuscript number into results/MANUSCRIPT_NUMBERS.json and .md, and write the supplementary tables to one workbook.
Run last. If a number in the paper is not here, it should not be in the paper."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import importlib, json; S = importlib.import_module("00_setup")
import pandas as pd
log = S.get_logger("19_consolidate_results")
R = S.RESULTS; res = json.load(open(R / "08_model_results.json"))
tabs = {n: pd.read_csv(R / f"{n}.csv") for n in ["table1_overall", "table1_by_region", "table2_filter_cascade", "table3_region_distribution",
    "table4_validation_schemes", "table5_temporal_holdout", "table6_region_holdout", "table7_features_logreg",
    "tableS1_single_source_effect", "tableS2_feature_matrix_summary", "tableS3_baselines", "tableS3b_carbapenemase_rule_by_region",
    "tableS4_bootstrap_cis", "tableS5_loo_region_sensitivity", "tableS6_generalisation_gap", "tableS7_error_analysis_by_carbapenemase",
    "tableS8_top_sequence_types_shifted_regions", "tableS9_leave_one_centre_out", "tableS10_learning_curve_local_data", "tableS11_recalibration"]}
for t in tabs.values():
    for c in t.columns:
        if str(t[c].dtype) == "object": t[c] = t[c].astype(object).where(t[c].notna(), "Not recorded") if c.lower() in ("region", "largest named centre", "centre") else t[c]
nums = {"versions": S.software_versions(),
    "cohort": {"verified": int(tabs["table3_region_distribution"].Verified_n.sum()), "annotated": int(tabs["table3_region_distribution"].Annotated_n.sum()), "features": S.N_FEATURES_EXPECTED},
    "random": {m: {k: res[f"{m}__random"][k] for k in ["auroc", "auroc_sd", "auprc", "accuracy", "f1"]} for m in S.MODELS},
    "mlst": {m: {k: res[f"{m}__mlst"][k] for k in ["auroc", "auroc_sd", "auprc", "accuracy", "f1"]} for m in S.MODELS},
    "temporal": {m: {k: res[f"{m}__temporal"][k] for k in ["auroc", "auprc", "n_train", "n_test", "n_excluded_missing_year"]} for m in S.MODELS},
    "loro": {m: {r: {k: v.get(k) for k in ["auroc", "n", "prevalence"]} for r, v in res[f"{m}__loro"].items()} for m in S.MODELS}}
json.dump(nums, open(R / "MANUSCRIPT_NUMBERS.json", "w"), indent=2, default=str)
with open(R / "MANUSCRIPT_NUMBERS.md", "w", encoding="utf-8") as fh:
    fh.write("# Manuscript numbers (generated, do not edit by hand)\n\n")
    for n, t in tabs.items():
        fh.write(f"## {n}\n\n{t.to_markdown(index=False) if hasattr(t, 'to_markdown') else t.to_string(index=False)}\n\n")
try:
    with pd.ExcelWriter(R / "supplementary_tables.xlsx") as xw:
        for n, t in tabs.items():
            if n.startswith("tableS"): t.to_excel(xw, sheet_name=n[5:30], index=False)
except ImportError:
    log.info("openpyxl missing; workbook skipped")
S.write_result("19_consolidate_results", {"files": list(tabs)}); log.info("Consolidated %d tables" % len(tabs))

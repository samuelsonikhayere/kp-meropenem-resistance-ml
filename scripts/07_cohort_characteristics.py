"""
07_cohort_characteristics.py — Finalise Table 1 with region breakdowns.
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from importlib.machinery import SourceFileLoader
from pathlib import Path as _P

import pandas as pd

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("07_cohort_characteristics")

df = pd.read_csv(setup.PROCESSED / "06_analysis_table.csv",
                 low_memory=False, dtype={"genome_id": str})

# ── Overall cohort summary ───────────────────────────────────────────────
overall = [
    {"metric": "Total isolates",           "value": int(len(df))},
    {"metric": "Resistant, n (%)",
     "value": f"{int(df['label'].sum())} ({df['label'].mean():.1%})"},
    {"metric": "Susceptible, n (%)",
     "value": f"{int((df['label'] == 0).sum())} ({(df['label'] == 0).mean():.1%})"},
    {"metric": "Unique countries",         "value": int(df["isolation_country"].nunique())},
    {"metric": "Unique MLST types",        "value": int(df["mlst"].nunique())},
    {"metric": "Median collection year",
     "value": int(df["collection_year"].median()) if df["collection_year"].notna().any() else "NA"},
    {"metric": "Genome quality = Good, n (%)",
     "value": f"{(df['genome_quality'] == 'Good').sum()} ({(df['genome_quality'] == 'Good').mean():.1%})"},
    {"metric": "Tier 1 lab method, n",
     "value": int((df["method_tier"] == 1).sum()) if "method_tier" in df.columns else "NA"},
    {"metric": "Tier 2 lab method, n",
     "value": int((df["method_tier"] == 2).sum()) if "method_tier" in df.columns else "NA"},
    {"metric": "Tier 3 lab method, n",
     "value": int((df["method_tier"] == 3).sum()) if "method_tier" in df.columns else "NA"},
]
overall_df = pd.DataFrame(overall)
overall_df.to_csv(setup.RESULTS / "table1_overall.csv", index=False)
log.info("Saved: results/table1_overall.csv")

# ── Region-stratified summary ────────────────────────────────────────────
by_region = (df.groupby("region")
             .agg(n=("label", "size"),
                  resistant=("label", "sum"),
                  susceptible=("label", lambda x: (x == 0).sum()),
                  resistance_rate=("label", "mean"),
                  median_year=("collection_year", "median"),
                  n_countries=("isolation_country", "nunique"),
                  n_mlst=("mlst", "nunique"))
             .sort_values("n", ascending=False)
             .reset_index())
by_region["resistance_rate_pct"] = (by_region["resistance_rate"] * 100).round(1)
by_region["median_year"] = by_region["median_year"].astype("Int64")
by_region.to_csv(setup.RESULTS / "table1_by_region.csv", index=False)
log.info("Saved: results/table1_by_region.csv")
log.info("\n" + by_region.to_string(index=False))

# ── Region × phenotype breakdown ─────────────────────────────────────────
region_pheno = pd.crosstab(df["region"], df["label"],
                           rownames=["region"],
                           colnames=["label"])
region_pheno.to_csv(setup.RESULTS / "table1_region_phenotype.csv")
log.info("Saved: results/table1_region_phenotype.csv")

setup.write_result("07_cohort_characteristics", {
    "analysis_name":     "cohort_characteristics",
    "dataset_version":   "api_v1",
    "n":                 int(len(df)),
    "n_regions":         int(df["region"].nunique()),
    "timestamp":         datetime.now(timezone.utc).isoformat(),
    "software_version":  setup.software_versions(),
})
log.info("Done.")
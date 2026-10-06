"""
02_build_cohort.py — Build the verified cohort with evidence-hierarchy dedup.

Input  : processed/01_merged.csv
Output : processed/02_verified_cohort.csv
         results/table1_cohort_characteristics.csv
         logs/02_build_cohort.log
         provenance/02_build_cohort.json

Decision rules (locked):
  1. Keep only phenotype in {Resistant, Susceptible}
  2. Keep only laboratory_typing_method in VERIFIED_METHOD_TIERS
     (excludes Computational Prediction and NaN)
  3. For genomes with multiple rows, retain highest evidence tier
     (tier 1 > 2 > 3); ties broken by earliest date_inserted
  4. Binarise label: Resistant=1, Susceptible=0
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from importlib.machinery import SourceFileLoader
from pathlib import Path as _P

import pandas as pd

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("02_build_cohort")


# ─────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────
df = pd.read_csv(setup.PROCESSED / "01_merged.csv",
                 low_memory=False, dtype={"genome_id": str})
log.info(f"Input rows: {len(df)}")
cascade = {"input_rows": len(df)}


# ─────────────────────────────────────────────────────────────────────────
# Step 1 — phenotype values
# ─────────────────────────────────────────────────────────────────────────
pheno_counts = df["resistant_phenotype"].value_counts(dropna=False).to_dict()
log.info(f"Phenotype distribution: {pheno_counts}")
cascade["phenotype_counts_raw"] = {str(k): int(v) for k, v in pheno_counts.items()}

before = len(df)
df = df[df["resistant_phenotype"].isin(setup.BINARY_PHENOTYPES)].copy()
log.info(f"Dropped {before - len(df)} rows with non-binary phenotype")
cascade["after_binary_phenotype_filter"] = len(df)


# ─────────────────────────────────────────────────────────────────────────
# Step 2 — verified lab method filter
# ─────────────────────────────────────────────────────────────────────────
method_counts = df["laboratory_typing_method"].value_counts(dropna=False).to_dict()
log.info(f"Method distribution (post-binary): {method_counts}")
cascade["method_counts_post_binary"] = {str(k): int(v) for k, v in method_counts.items()}

before = len(df)
df["method_tier"] = df["laboratory_typing_method"].map(setup.VERIFIED_METHOD_TIERS)
df = df[df["method_tier"].notna()].copy()
log.info(f"Dropped {before - len(df)} rows without verified lab method")
cascade["after_verified_method_filter"] = len(df)


# ─────────────────────────────────────────────────────────────────────────
# Step 3 — evidence-hierarchy dedup
# ─────────────────────────────────────────────────────────────────────────
df["date_inserted"] = pd.to_datetime(df["date_inserted"], errors="coerce", utc=True)
df = df.sort_values(
    ["genome_id", "method_tier", "date_inserted"],
    ascending=[True, True, True],
)
before = len(df)
before_unique = df["genome_id"].nunique()
df = df.drop_duplicates(subset="genome_id", keep="first").reset_index(drop=True)
log.info(f"Dedup: {before} rows → {len(df)} genomes "
         f"({before - len(df)} records collapsed across {before_unique} unique genomes)")
cascade["after_dedup"] = len(df)


# ─────────────────────────────────────────────────────────────────────────
# Step 4 — binarise label
# ─────────────────────────────────────────────────────────────────────────
df["label"] = (df["resistant_phenotype"] == "Resistant").astype(int)
log.info(f"Resistant: {int(df['label'].sum())} ({df['label'].mean():.1%})")
log.info(f"Susceptible: {int((df['label'] == 0).sum())}")


# ─────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────
df.to_csv(setup.PROCESSED / "02_verified_cohort.csv", index=False)
log.info("Saved: processed/02_verified_cohort.csv")


# ─────────────────────────────────────────────────────────────────────────
# Table 1 — Cohort characteristics
# ─────────────────────────────────────────────────────────────────────────
rows = []
def add(k, v): rows.append({"characteristic": k, "value": v})

add("Total verified isolates",             len(df))
add("Resistant, n (%)",                    f"{int(df['label'].sum())} ({df['label'].mean():.1%})")
add("Susceptible, n (%)",                  f"{int((df['label'] == 0).sum())} ({(df['label'] == 0).mean():.1%})")
add("Unique countries",                    int(df["isolation_country"].nunique()))
add("Unique MLST sequence types",          int(df["mlst"].nunique()))
add("Median collection year",              int(df["collection_year"].median()) if df["collection_year"].notna().any() else "NA")
add("Earliest collection year",            int(df["collection_year"].min()) if df["collection_year"].notna().any() else "NA")
add("Latest collection year",              int(df["collection_year"].max()) if df["collection_year"].notna().any() else "NA")
add("Genome quality = Good, n (%)",        f"{(df['genome_quality'] == 'Good').sum()} ({(df['genome_quality'] == 'Good').mean():.1%})")
add("Tier 1 method (MIC/dilution/Etest)",  int((df["method_tier"] == 1).sum()))
add("Tier 2 method (automated platform)",  int((df["method_tier"] == 2).sum()))
add("Tier 3 method (disk diffusion)",      int((df["method_tier"] == 3).sum()))

table1 = pd.DataFrame(rows)
table1.to_csv(setup.RESULTS / "table1_cohort_characteristics.csv", index=False)
log.info("Saved: results/table1_cohort_characteristics.csv")


# ─────────────────────────────────────────────────────────────────────────
# Provenance + result object
# ─────────────────────────────────────────────────────────────────────────
prov = {
    "script":   "02_build_cohort.py",
    "run_at":   datetime.now(timezone.utc).isoformat(),
    "software": setup.software_versions(),
    "cascade":  cascade,
    "final": {
        "n":            int(len(df)),
        "resistant":    int(df["label"].sum()),
        "susceptible":  int((df["label"] == 0).sum()),
        "countries":    int(df["isolation_country"].nunique()),
        "mlsts":        int(df["mlst"].nunique()),
    },
}
(setup.PROVENANCE / "02_build_cohort.json").write_text(
    json.dumps(prov, indent=2), encoding="utf-8")

setup.write_result("02_build_cohort", {
    "analysis_name":     "verified_cohort_construction",
    "dataset_version":   "api_v1",
    "n":                 int(len(df)),
    "positive":          int(df["label"].sum()),
    "negative":          int((df["label"] == 0).sum()),
    "timestamp":         datetime.now(timezone.utc).isoformat(),
    "software_version":  setup.software_versions(),
})
log.info("Done.")
"""
04_region_assignment.py — Assign geographic region to each isolate.

Input  : processed/02_verified_cohort.csv
Output : processed/04_with_regions.csv
         results/table3_region_distribution.csv
         results/table3b_region_sensitivity.csv
         logs/04_region_assignment.log
         provenance/04_region_assignment.json

Primary mapping    : Caribbean reclassified to "Unknown"
                     (decision 1A: 246/247 Caribbean isolates are a single
                     WTSI submission from 2019-07-15; not a geographic region)
Sensitivity mapping: Caribbean folded into North America

Both columns are written so downstream scripts can switch.
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from importlib.machinery import SourceFileLoader
from pathlib import Path as _P

import pandas as pd

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("04_region_assignment")


# ─────────────────────────────────────────────────────────────────────────
# Safe CSV write — catches PermissionError with a clear message
# ─────────────────────────────────────────────────────────────────────────
def safe_write_csv(df: pd.DataFrame, path) -> None:
    """Write a DataFrame to CSV, with a helpful error if the file is locked."""
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        log.error("=" * 70)
        log.error(f"Could not write: {path}")
        log.error("The file is likely open in Excel or another program.")
        log.error("Please close the file, then re-run this script.")
        log.error("=" * 70)
        raise


# ─────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────
df = pd.read_csv(setup.PROCESSED / "02_verified_cohort.csv",
                 low_memory=False, dtype={"genome_id": str})
log.info(f"Input rows: {len(df)}")


# ─────────────────────────────────────────────────────────────────────────
# Primary mapping — Caribbean → Unknown
# ─────────────────────────────────────────────────────────────────────────
df["region"] = df["isolation_country"].map(setup.REGION_MAP)
df["region"] = df["region"].fillna("Unknown")

# Reclassify Caribbean → Unknown for the primary analysis
n_carib = int((df["region"] == "Caribbean").sum())
df.loc[df["region"] == "Caribbean", "region"] = "Unknown"
log.info(f"Reclassified {n_carib} Caribbean isolates → Unknown (primary mapping)")


# ─────────────────────────────────────────────────────────────────────────
# Sensitivity mapping — Caribbean → North America
# ─────────────────────────────────────────────────────────────────────────
df["region_sensitivity"] = df["isolation_country"].map(setup.REGION_MAP)
df["region_sensitivity"] = df["region_sensitivity"].fillna("Unknown")
df.loc[df["region_sensitivity"] == "Caribbean", "region_sensitivity"] = "North America"


# ─────────────────────────────────────────────────────────────────────────
# Distribution table (primary mapping)
# ─────────────────────────────────────────────────────────────────────────
dist = (df
        .groupby("region")
        .agg(n=("label", "size"),
             resistant=("label", "sum"),
             susceptible=("label", lambda x: (x == 0).sum()),
             resistance_rate=("label", "mean"))
        .sort_values("n", ascending=False)
        .reset_index())
dist["resistance_rate_pct"] = (dist["resistance_rate"] * 100).round(1)

safe_write_csv(dist, setup.RESULTS / "table3_region_distribution.csv")
log.info("Saved: results/table3_region_distribution.csv")
log.info("\n" + dist.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────
# Sensitivity distribution
# ─────────────────────────────────────────────────────────────────────────
dist_sens = (df
             .groupby("region_sensitivity")
             .agg(n=("label", "size"),
                  resistant=("label", "sum"),
                  susceptible=("label", lambda x: (x == 0).sum()),
                  resistance_rate=("label", "mean"))
             .sort_values("n", ascending=False)
             .reset_index())
dist_sens["resistance_rate_pct"] = (dist_sens["resistance_rate"] * 100).round(1)

safe_write_csv(dist_sens, setup.RESULTS / "table3b_region_sensitivity.csv")
log.info("Saved: results/table3b_region_sensitivity.csv")
log.info("\nSensitivity mapping (Caribbean → North America):")
log.info("\n" + dist_sens.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────
# Save the full cohort with regions
# ─────────────────────────────────────────────────────────────────────────
safe_write_csv(df, setup.PROCESSED / "04_with_regions.csv")
log.info("Saved: processed/04_with_regions.csv")


# ─────────────────────────────────────────────────────────────────────────
# Provenance + result object
# ─────────────────────────────────────────────────────────────────────────
prov = {
    "script":   "04_region_assignment.py",
    "run_at":   datetime.now(timezone.utc).isoformat(),
    "software": setup.software_versions(),
    "caribbean_reclassified_to_unknown": n_carib,
    "primary_region_counts":     dist.set_index("region")["n"].to_dict(),
    "sensitivity_region_counts": dist_sens.set_index("region_sensitivity")["n"].to_dict(),
}
(setup.PROVENANCE / "04_region_assignment.json").write_text(
    json.dumps(prov, indent=2), encoding="utf-8")

setup.write_result("04_region_assignment", {
    "analysis_name":       "region_assignment",
    "dataset_version":     "api_v1",
    "n":                   int(len(df)),
    "regions_primary":     dist.set_index("region")["n"].to_dict(),
    "regions_sensitivity": dist_sens.set_index("region_sensitivity")["n"].to_dict(),
    "timestamp":           datetime.now(timezone.utc).isoformat(),
    "software_version":    setup.software_versions(),
})
log.info("Done.")
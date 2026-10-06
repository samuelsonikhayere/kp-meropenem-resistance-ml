"""
03_filter_cascade.py — Exact filter cascade table for the manuscript.

Reads the provenance JSONs from 01 and 02 plus the raw CSVs, and produces
a single table showing records/isolates at each analysis stage.
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from importlib.machinery import SourceFileLoader
from pathlib import Path as _P

import pandas as pd

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("03_filter_cascade")


# ─────────────────────────────────────────────────────────────────────────
# Load cascade stages
# ─────────────────────────────────────────────────────────────────────────
raw_amr    = pd.read_csv(setup.RAW / "BVBRC_genome_amr.csv", low_memory=False)
raw_genome = pd.read_csv(setup.RAW / "BVBRC_genome.csv",     low_memory=False)
merged     = pd.read_csv(setup.PROCESSED / "01_merged.csv",
                         low_memory=False, dtype={"genome_id": str})
cohort     = pd.read_csv(setup.PROCESSED / "02_verified_cohort.csv",
                         low_memory=False, dtype={"genome_id": str})

# Standardise column names for stage-count queries
def snake(df):
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    return df

raw_amr_s = snake(raw_amr)
raw_genome_s = snake(raw_genome)

has_pheno = merged[merged["resistant_phenotype"].notna()]
binary    = merged[merged["resistant_phenotype"].isin(setup.BINARY_PHENOTYPES)]
verified  = binary[binary["laboratory_typing_method"].isin(setup.VERIFIED_METHOD_TIERS)]


# ─────────────────────────────────────────────────────────────────────────
# Cascade table
# ─────────────────────────────────────────────────────────────────────────
rows = [
    {"stage": "Raw K. pneumoniae genomes (API)",
     "records": int(len(raw_genome)),
     "unique_genomes": int(raw_genome_s["genome_id"].nunique()) if "genome_id" in raw_genome_s.columns else None,
     "note": "Downloaded via API; includes 7 K. pneumoniae phage rows"},

    {"stage": "Raw meropenem AMR records (API)",
     "records": int(len(raw_amr)),
     "unique_genomes": int(raw_amr_s["genome_id"].nunique()) if "genome_id" in raw_amr_s.columns else None,
     "note": "All meropenem phenotype records from BV-BRC"},

    {"stage": "After genome deduplication and phage removal",
     "records": None,
     "unique_genomes": int(merged["genome_id"].nunique()),
     "note": "One metadata row per genome_id; non-bacterial rows dropped"},

    {"stage": "After inner merge",
     "records": int(len(merged)),
     "unique_genomes": int(merged["genome_id"].nunique()),
     "note": "Metadata joined to AMR records"},

    {"stage": "After dropping missing phenotype",
     "records": int(len(has_pheno)),
     "unique_genomes": int(has_pheno["genome_id"].nunique()),
     "note": "resistant_phenotype = NaN excluded"},

    {"stage": "After keeping only R/S phenotypes",
     "records": int(len(binary)),
     "unique_genomes": int(binary["genome_id"].nunique()),
     "note": "Intermediate / Susceptible-dose dependent excluded"},

    {"stage": "After verified-method filter",
     "records": int(len(verified)),
     "unique_genomes": int(verified["genome_id"].nunique()),
     "note": "Computational Prediction and NaN method excluded"},

    {"stage": "After evidence-hierarchy deduplication (final cohort)",
     "records": int(len(cohort)),
     "unique_genomes": int(cohort["genome_id"].nunique()),
     "note": "One record per genome; best evidence tier retained"},
]

cascade = pd.DataFrame(rows)
cascade.to_csv(setup.RESULTS / "table2_filter_cascade.csv", index=False)
log.info("Saved: results/table2_filter_cascade.csv")
log.info("\n" + cascade.to_string(index=False))

# Provenance
prov = {
    "script":   "03_filter_cascade.py",
    "run_at":   datetime.now(timezone.utc).isoformat(),
    "software": setup.software_versions(),
    "cascade":  rows,
}
(setup.PROVENANCE / "03_filter_cascade.json").write_text(
    json.dumps(prov, indent=2), encoding="utf-8")
log.info("Done.")
"""
01_merge.py — Merge BV-BRC genome metadata and AMR phenotype tables.

Inputs (both freshly downloaded from the BV-BRC API by 00b_download_raw.py):
  raw_data/BVBRC_genome.csv
  raw_data/BVBRC_genome_amr.csv

Outputs:
  processed/01_merged.csv
  logs/01_merge.log
  provenance/01_merge.json

Decisions locked:
  - Genome table: deduplicate by Genome ID.
      * Prefer 'Complete' Genome Status
      * Then earliest Completion Date
  - Drop non-bacterial rows (7 K. pneumoniae phage records)
  - No filtering of AMR table at this stage
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from importlib.machinery import SourceFileLoader
from pathlib import Path as _P

import pandas as pd

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("01_merge")


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────
def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Snake_case column names; strip whitespace; unify punctuation."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
          .str.strip()
          .str.replace(r"\s+", "_", regex=True)
          .str.replace(r"\.", "_", regex=True)
          .str.replace(r"[^\w]", "_", regex=True)
          .str.lower()
    )
    return df


def load_raw(path, key_col_after_normalise="genome_id"):
    """Load a raw BV-BRC CSV and normalise column names."""
    df = pd.read_csv(path, low_memory=False)
    df = normalise_columns(df)
    if key_col_after_normalise in df.columns:
        df[key_col_after_normalise] = df[key_col_after_normalise].astype(str)
    return df


# ─────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────
log.info("Loading raw CSVs from API download")
genome = load_raw(setup.RAW / "BVBRC_genome.csv")
amr    = load_raw(setup.RAW / "BVBRC_genome_amr.csv")

log.info(f"genome metadata (raw):     {genome.shape}")
log.info(f"amr phenotypes (raw):      {amr.shape}")


# ─────────────────────────────────────────────────────────────────────────
# Step 1 — Genome table cleanup
# ─────────────────────────────────────────────────────────────────────────
# 1a. Drop non-K. pneumoniae species (7 phage rows)
species_col = "species" if "species" in genome.columns else None
if species_col:
    before = len(genome)
    genome = genome[genome[species_col] == "Klebsiella pneumoniae"].copy()
    log.info(f"Dropped {before - len(genome)} non-K. pneumoniae rows "
             f"(phage etc.); remaining {len(genome)}")

# 1b. Deduplicate by genome_id — prefer 'Complete' status, then earliest date
KEY = "genome_id"
genome[KEY] = genome[KEY].astype(str)

status_col = "genome_status"
date_col   = "completion_date"

# Convert completion_date for sorting (ISO strings sort correctly but parse defensively)
if date_col in genome.columns:
    genome["_completion_dt"] = pd.to_datetime(
        genome[date_col], errors="coerce", utc=True
    )
else:
    genome["_completion_dt"] = pd.NaT

# Prefer Complete status, then earliest completion date
if status_col in genome.columns:
    genome["_status_rank"] = (genome[status_col] != "Complete").astype(int)
else:
    genome["_status_rank"] = 0

genome = genome.sort_values([KEY, "_status_rank", "_completion_dt"])
n_before = len(genome)
genome = genome.drop_duplicates(subset=KEY, keep="first")
n_after = len(genome)
log.info(f"Deduplicated genome table: {n_before} → {n_after} rows "
         f"(dropped {n_before - n_after})")

genome = genome.drop(columns=["_status_rank", "_completion_dt"])


# ─────────────────────────────────────────────────────────────────────────
# Step 2 — Merge
# ─────────────────────────────────────────────────────────────────────────
assert KEY in amr.columns, f"{KEY} missing from amr table"
assert KEY in genome.columns, f"{KEY} missing from genome table"

merged = amr.merge(genome, on=KEY, how="inner", suffixes=("_amr", "_meta"))
log.info(f"Merged rows:               {len(merged)}")
log.info(f"Unique genome_ids:         {merged[KEY].nunique()}")
log.info(f"Antibiotics in merged:     "
         f"{sorted(merged['antibiotic'].dropna().unique().tolist())}")

# Sanity checks
assert merged["antibiotic"].nunique() == 1, "Multiple antibiotics present"
assert merged["antibiotic"].iloc[0] == "meropenem", "Not meropenem"


# ─────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────
out_path = setup.PROCESSED / "01_merged.csv"
merged.to_csv(out_path, index=False)
log.info(f"Saved: {out_path.relative_to(setup.ROOT)}")


# ─────────────────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────────────────
prov = {
    "script":      "01_merge.py",
    "run_at":      datetime.now(timezone.utc).isoformat(),
    "software":    setup.software_versions(),
    "inputs": {
        "genome_metadata": str((setup.RAW / "BVBRC_genome.csv").relative_to(setup.ROOT)),
        "amr_phenotypes":  str((setup.RAW / "BVBRC_genome_amr.csv").relative_to(setup.ROOT)),
    },
    "outputs":     {"merged": str(out_path.relative_to(setup.ROOT))},
    "counts": {
        "genome_rows_raw":          int(n_before),
        "genome_rows_after_clean":  int(n_after),
        "amr_rows":                 int(len(amr)),
        "merged_rows":              int(len(merged)),
        "merged_unique_genome_id":  int(merged[KEY].nunique()),
    },
    "decisions": {
        "genome_dedup_rule":  "Prefer Genome Status = 'Complete', then earliest Completion Date",
        "phage_filter":       "Dropped rows where Species != 'Klebsiella pneumoniae'",
    },
}
(setup.PROVENANCE / "01_merge.json").write_text(
    json.dumps(prov, indent=2), encoding="utf-8")
log.info("Done.")
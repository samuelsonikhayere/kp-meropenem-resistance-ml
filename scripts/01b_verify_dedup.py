"""
01b_verify_dedup.py — Verify that genome metadata dedup actually ran.
"""

from importlib.machinery import SourceFileLoader
from pathlib import Path as _P
import pandas as pd

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("01b_verify_dedup")

# Read raw genome table
genome_raw = pd.read_csv(setup.RAW / "BVBRC_genome.csv", low_memory=False)
log.info(f"Raw genome table: {genome_raw.shape}")
log.info(f"Columns (first 5): {genome_raw.columns.tolist()[:5]}")

# Check genome ID duplicates in raw
id_col = "Genome ID" if "Genome ID" in genome_raw.columns else "genome_id"
log.info(f"\nUsing ID column: {id_col}")
log.info(f"Unique genome IDs: {genome_raw[id_col].nunique()}")
log.info(f"Duplicated rows: {genome_raw[id_col].duplicated().sum()}")

# Read merged table
merged = pd.read_csv(setup.PROCESSED / "01_merged.csv", low_memory=False)
log.info(f"\nMerged table: {merged.shape}")

# Check how many metadata columns duplicate per genome_id
# The metadata columns have _meta suffix if there was a collision, or appear once
meta_cols = [c for c in merged.columns if c not in
             ["taxon_id", "genome_id", "genome_name_amr", "antibiotic",
              "resistant_phenotype", "measurement", "measurement_sign",
              "measurement_value", "measurement_unit",
              "laboratory_typing_method", "laboratory_typing_method_version",
              "laboratory_typing_platform", "vendor", "testing_standard",
              "testing_standard_year", "computational_method",
              "computational_method_version", "computational_method_performance",
              "evidence", "source", "pubmed"]]

log.info(f"\nMetadata columns in merged table: {len(meta_cols)}")

# For one duplicated genome, check if all metadata rows are identical
sample_dup = merged[merged.duplicated(subset="genome_id", keep=False)]["genome_id"].iloc[0]
sample_rows = merged[merged["genome_id"] == sample_dup]
log.info(f"\nSample genome with multiple rows: {sample_dup}")
log.info(f"Number of rows: {len(sample_rows)}")

# Are metadata fields identical across these rows?
if len(sample_rows) > 1:
    meta_sample = sample_rows[meta_cols].drop_duplicates()
    log.info(f"Distinct metadata combinations: {len(meta_sample)}")
    if len(meta_sample) == 1:
        log.info("✓ Metadata is consistent — dedup worked")
    else:
        log.info("✗ Metadata differs across AMR rows — dedup did not work")
        log.info(f"Differences:")
        for col in meta_cols:
            unique_vals = sample_rows[col].dropna().unique()
            if len(unique_vals) > 1:
                log.info(f"  {col}: {unique_vals}")
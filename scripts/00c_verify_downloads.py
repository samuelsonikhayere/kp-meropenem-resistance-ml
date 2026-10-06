"""
00c_verify_downloads.py — Lean verification of downloaded BV-BRC tables.
"""

from importlib.machinery import SourceFileLoader
from pathlib import Path as _P
import pandas as pd

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("00c_verify_downloads")

# ── Genome table ──────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("GENOME METADATA")
log.info("=" * 60)
genome = pd.read_csv(setup.RAW / "BVBRC_genome.csv", low_memory=False)
log.info(f"Shape: {genome.shape}")

# Species
log.info(f"\nSpecies breakdown:")
log.info(f"{genome['Species'].value_counts().to_string()}")

# Duplicate Genome IDs — investigate
log.info(f"\nDuplicate Genome IDs:")
log.info(f"  Total rows:              {len(genome)}")
log.info(f"  Unique Genome IDs:       {genome['Genome ID'].nunique()}")
log.info(f"  Duplicated rows:         {genome['Genome ID'].duplicated().sum()}")

# Look at one duplicated genome to understand why
dup_ids = genome[genome['Genome ID'].duplicated(keep=False)]['Genome ID'].unique()
if len(dup_ids):
    sample_id = dup_ids[0]
    log.info(f"\n  Sample duplicated genome: {sample_id}")
    cols_to_show = ['Genome ID', 'Genome Name', 'Genome Status', 'Completion Date',
                    'Sequencing Status', 'Assembly Accession']
    cols_to_show = [c for c in cols_to_show if c in genome.columns]
    sample_rows = genome[genome['Genome ID'] == sample_id][cols_to_show]
    log.info(f"{sample_rows.to_string()}")

# ── AMR table ─────────────────────────────────────────────────────────────
log.info("")
log.info("=" * 60)
log.info("AMR PHENOTYPE TABLE")
log.info("=" * 60)
amr = pd.read_csv(setup.RAW / "BVBRC_genome_amr.csv", low_memory=False)
log.info(f"Shape: {amr.shape}")

log.info(f"\nAntibiotic:")
log.info(f"{amr['Antibiotic'].value_counts().to_string()}")

log.info(f"\nResistant Phenotype:")
log.info(f"{amr['Resistant Phenotype'].value_counts(dropna=False).to_string()}")

log.info(f"\nLaboratory Typing Method:")
log.info(f"{amr['Laboratory Typing Method'].value_counts(dropna=False).to_string()}")

log.info(f"\nUnique Genome IDs in AMR:  {amr['Genome ID'].nunique()}")
log.info(f"Duplicated AMR rows:       {amr['Genome ID'].duplicated().sum()}")
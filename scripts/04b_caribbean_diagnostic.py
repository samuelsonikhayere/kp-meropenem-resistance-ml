"""
04b_caribbean_diagnostic.py — Characterise the Caribbean sample.
"""

from importlib.machinery import SourceFileLoader
from pathlib import Path as _P
import pandas as pd

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("04b_caribbean")

df = pd.read_csv(setup.PROCESSED / "04_with_regions.csv",
                 low_memory=False, dtype={"genome_id": str})

carib = df[df["region"] == "Caribbean"].copy()

log.info("=" * 70)
log.info(f"CARIBBEAN SAMPLE — n = {len(carib)}")
log.info("=" * 70)

log.info(f"\nCountries:")
log.info(f"{carib['isolation_country'].value_counts().to_string()}")

log.info(f"\nResistance rate by country:")
log.info(f"{carib.groupby('isolation_country')['label'].agg(['size','sum','mean']).round(3).to_string()}")

log.info(f"\nSequencing centers:")
log.info(f"{carib['sequencing_center'].value_counts(dropna=False).head(15).to_string()}")

log.info(f"\nCollection years:")
log.info(f"{carib['collection_year'].value_counts(dropna=False).sort_index().to_string()}")

log.info(f"\nMethod tiers:")
log.info(f"{carib['method_tier'].value_counts().to_string()}")

log.info(f"\nTesting standards:")
log.info(f"{carib['testing_standard'].value_counts(dropna=False).to_string()}")

log.info(f"\nAntibiotic: {carib['antibiotic'].unique()}")

# Compare with non-Caribbean
non_carib = df[df["region"] != "Caribbean"]
log.info("")
log.info("=" * 70)
log.info("CARIBBEAN vs REST")
log.info("=" * 70)
log.info(f"Caribbean n={len(carib)}, resistant={carib['label'].sum()} ({carib['label'].mean():.1%})")
log.info(f"Non-Caribbean n={len(non_carib)}, resistant={non_carib['label'].sum()} ({non_carib['label'].mean():.1%})")

# Do Caribbean isolates have AMR annotations?
carib_genes = pd.read_csv(setup.PROCESSED / "X_features.csv",
                          low_memory=False,
                          index_col=0,
                          dtype={0: str}) if (setup.PROCESSED / "X_features.csv").exists() else None
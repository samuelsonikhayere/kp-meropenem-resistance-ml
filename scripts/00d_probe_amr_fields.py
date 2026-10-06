"""
00d_probe_amr_fields.py — Probe whether genome_amr API returns more
fields when explicitly requested via select().
"""

import io
import requests
from importlib.machinery import SourceFileLoader
from pathlib import Path as _P
import pandas as pd

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("00d_probe_amr_fields")

BASE = "https://www.bv-brc.org/api"

# The columns in the current (sparse) API download
CURRENT_COLS = [
    "genome_id", "antibiotic", "resistant_phenotype",
    "laboratory_typing_method", "laboratory_typing_method_version",
    "laboratory_typing_platform", "vendor",
    "testing_standard", "testing_standard_year",
    "measurement", "measurement_sign", "measurement_value", "measurement_unit",
    "computational_method", "computational_method_version",
    "computational_method_performance",
    "evidence", "source", "pubmed",
]

# Probe 1: request all default columns for a small batch, see method sparsity
log.info("=" * 70)
log.info("PROBE 1 — genome_amr, default fields, first 100 records")
log.info("=" * 70)
url = (f"{BASE}/genome_amr/?"
       f"and(eq(taxon_id,573),eq(antibiotic,meropenem))"
       f"&limit(100)&http_accept=text/csv")
r = requests.get(url, timeout=60)
log.info(f"Status: {r.status_code}")
if r.status_code == 200:
    df = pd.read_csv(io.StringIO(r.text))
    log.info(f"Rows: {len(df)}, Cols: {df.shape[1]}")
    log.info(f"Columns: {df.columns.tolist()}")
    if "Laboratory Typing Method" in df.columns:
        log.info(f"\nLaboratory Typing Method non-null: "
                 f"{df['Laboratory Typing Method'].notna().sum()} / {len(df)}")
        log.info(f"{df['Laboratory Typing Method'].value_counts(dropna=False).head(10).to_string()}")
    if "Source" in df.columns:
        log.info(f"\nSource non-null: {df['Source'].notna().sum()} / {len(df)}")


# Probe 2: request explicit select() with all known field names
log.info("")
log.info("=" * 70)
log.info("PROBE 2 — genome_amr with explicit select() of all known fields")
log.info("=" * 70)
select_fields = ",".join(CURRENT_COLS)
url2 = (f"{BASE}/genome_amr/?"
        f"and(eq(taxon_id,573),eq(antibiotic,meropenem))"
        f"&limit(100)&select({select_fields})&http_accept=text/csv")
r2 = requests.get(url2, timeout=60)
log.info(f"Status: {r2.status_code}")
if r2.status_code == 200:
    df2 = pd.read_csv(io.StringIO(r2.text))
    log.info(f"Rows: {len(df2)}, Cols: {df2.shape[1]}")
    log.info(f"Columns: {df2.columns.tolist()}")
    if "laboratory_typing_method" in df2.columns:
        log.info(f"\nlaboratory_typing_method non-null: "
                 f"{df2['laboratory_typing_method'].notna().sum()} / {len(df2)}")

# Probe 3: check the genome_feature endpoint variant (source of the earlier
# method-rich download from the website)
log.info("")
log.info("=" * 70)
log.info("PROBE 3 — genome_amr with sorted output (default sort may affect sparsity)")
log.info("=" * 70)
url3 = (f"{BASE}/genome_amr/?"
        f"and(eq(taxon_id,573),eq(antibiotic,meropenem))"
        f"&limit(1000)&http_accept=text/csv")
r3 = requests.get(url3, timeout=120)
log.info(f"Status: {r3.status_code}")
if r3.status_code == 200:
    df3 = pd.read_csv(io.StringIO(r3.text))
    log.info(f"Rows: {len(df3)}")
    if "Laboratory Typing Method" in df3.columns:
        method_counts = df3["Laboratory Typing Method"].value_counts(dropna=False)
        log.info(f"\nMethod counts in first 1000 records:")
        log.info(f"{method_counts.to_string()}")
        log.info(f"\nNon-null fraction: {df3['Laboratory Typing Method'].notna().mean():.1%}")
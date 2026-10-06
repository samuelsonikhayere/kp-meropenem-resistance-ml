"""
04c_submission_source.py — Characterise single-source submission effects.

For each region, report:
  - Total isolates
  - Top sequencing center and its contribution
  - Resistance rate in that center's submission
  - Resistance rate in the rest of the region
  - Number of distinct sequencing centers
"""

from importlib.machinery import SourceFileLoader
from pathlib import Path as _P
import pandas as pd

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("04c_submission_source")

df = pd.read_csv(setup.PROCESSED / "04_with_regions.csv",
                 low_memory=False, dtype={"genome_id": str})

log.info("=" * 70)
log.info("SINGLE-SOURCE SUBMISSION EFFECT — by region")
log.info("=" * 70)

rows = []
for region in sorted(df["region"].unique()):
    sub = df[df["region"] == region]
    centers = sub["sequencing_center"].value_counts(dropna=False)
    if len(centers) == 0:
        continue

    top_center = centers.index[0]
    top_n = int(centers.iloc[0])
    top_frac = top_n / len(sub)

    top_sub = sub[sub["sequencing_center"] == top_center]
    rest_sub = sub[sub["sequencing_center"] != top_center]

    rows.append({
        "region":             region,
        "n":                  len(sub),
        "top_center":         str(top_center)[:40],
        "top_n":              top_n,
        "top_fraction":       round(top_frac, 3),
        "resistance_top":     round(top_sub["label"].mean(), 3) if len(top_sub) else None,
        "n_rest":             len(rest_sub),
        "resistance_rest":    round(rest_sub["label"].mean(), 3) if len(rest_sub) else None,
        "region_resistance":  round(sub["label"].mean(), 3),
    })

diag = pd.DataFrame(rows).sort_values("n", ascending=False)
log.info("\n" + diag.to_string(index=False))

diag.to_csv(setup.RESULTS / "tableS1_single_source_effect.csv", index=False)
log.info(f"\nSaved: results/tableS1_single_source_effect.csv")


log.info("")
log.info("=" * 70)
log.info("NUMBER OF DISTINCT SEQUENCING CENTERS PER REGION")
log.info("=" * 70)

for region in sorted(df["region"].unique()):
    sub = df[df["region"] == region]
    n_centers = sub["sequencing_center"].dropna().nunique()
    n_missing = int(sub["sequencing_center"].isna().sum())
    log.info(f"{region:20s}  n={len(sub):5d}  centers={n_centers:3d}  missing_center={n_missing}")


log.info("")
log.info("=" * 70)
log.info("CARIBBEAN SAMPLE — SOURCE DETAIL")
log.info("=" * 70)
carib = df[df["region"] == "Caribbean"]
if len(carib):
    log.info(f"Total Caribbean isolates: {len(carib)}")
    log.info(f"Sequencing center breakdown:")
    log.info(f"{carib['sequencing_center'].value_counts(dropna=False).to_string()}")

    log.info(f"\nPhenotype breakdown within WTSI-submitted Caribbean isolates:")
    wtsi = carib[carib["sequencing_center"] == "WTSI"].copy()
    if len(wtsi):
        log.info(f"{wtsi['resistant_phenotype'].value_counts(dropna=False).to_string()}")

        wtsi["_date_inserted_dt"] = pd.to_datetime(
            wtsi["date_inserted"], errors="coerce", utc=True
        )
        dmin = wtsi["_date_inserted_dt"].min()
        dmax = wtsi["_date_inserted_dt"].max()
        log.info(f"\nWTSI date_inserted range: {dmin} to {dmax}")
        log.info(f"WTSI unique genome IDs: {wtsi['genome_id'].nunique()}")
        log.info(f"WTSI top MLST types:")
        log.info(f"{wtsi['mlst'].value_counts().head(10).to_string()}")
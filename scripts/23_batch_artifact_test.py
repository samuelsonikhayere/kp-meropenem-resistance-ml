"""
23_batch_artifact_test.py

Test the hypothesis: African isolates fall into 2 annotation batches
based on which genes were called, NOT based on biology.
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT      = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "processed"

verif  = pd.read_csv(PROCESSED / "verified_final.csv",
                     low_memory=False, dtype={"genome_id": str})
matrix = pd.read_csv(PROCESSED / "X_features.csv",
                     low_memory=False, dtype={"genome_id": str}, index_col=0)

verif = verif.set_index("genome_id").loc[matrix.index].reset_index()
annotated = (matrix.sum(axis=1) > 0).loc[verif["genome_id"]].values

X    = matrix.loc[annotated].values
meta = verif.loc[annotated].reset_index(drop=True)
cols = matrix.columns

# Focus on African isolates
afr_mask = (meta["region"] == "Africa").values
X_afr    = X[afr_mask]
meta_afr = meta[afr_mask].reset_index(drop=True)


print("=" * 70)
print("AFRICAN ISOLATES — batch clustering hypothesis")
print("=" * 70)

# Use the 15 depleted genes as batch markers
BATCH_MARKERS = ["emrD","cpxA","parE","murA","cysB","mfd","alaS",
                 "soxR","kdpE","mdfA","patA","mdtD","nfsA","acrR","cpxR"]

marker_idx = [list(cols).index(g) for g in BATCH_MARKERS if g in cols]
marker_sub = X_afr[:, marker_idx]

# Do all African isolates have the same pattern across these markers?
unique_patterns, counts = np.unique(marker_sub, axis=0, return_counts=True)
print(f"\nDistinct patterns across {len(marker_idx)} batch marker genes:")
for pat, cnt in zip(unique_patterns, counts):
    print(f"  pattern {pat.tolist()} → {cnt} isolates")

# Cluster by total number of these markers present
marker_totals = marker_sub.sum(axis=1)
print(f"\nTotal marker genes present per isolate:")
print(pd.Series(marker_totals).value_counts().sort_index().to_string())


print("\n" + "=" * 70)
print("DO THE TWO GROUPS DIFFER BY COUNTRY, MLST, OR SOURCE?")
print("=" * 70)

# Label each isolate by which "batch" it belongs to
meta_afr["batch"] = np.where(marker_totals > 5, "Batch_A", "Batch_B")

print(f"\nBatch A (has marker genes):   {(meta_afr['batch'] == 'Batch_A').sum()}")
print(f"Batch B (lacks marker genes): {(meta_afr['batch'] == 'Batch_B').sum()}")

print(f"\nBatch A × country:")
print(pd.crosstab(meta_afr["batch"], meta_afr["isolation_country"]))
print(f"\nBatch A × MLST (top 10):")
print(pd.crosstab(meta_afr["batch"], meta_afr["mlst"]).iloc[:, :10])
print(f"\nBatch A × sequencing_center:")
print(pd.crosstab(meta_afr["batch"], meta_afr["sequencing_center"].fillna("NaN")))
print(f"\nBatch A × collection_year:")
print(pd.crosstab(meta_afr["batch"], meta_afr["collection_year"].fillna("NaN")))
print(f"\nBatch A × label:")
print(pd.crosstab(meta_afr["batch"], meta_afr["label"]))


print("\n" + "=" * 70)
print("DOES THE GLOBAL SUBSET SHOW THE SAME SPLIT?")
print("=" * 70)

# Check whether non-African isolates also split into these two patterns
marker_sub_all = X[:, marker_idx]
totals_all     = marker_sub_all.sum(axis=1)

print(f"\nMarker-gene totals across ALL 1587 annotated isolates:")
print(pd.Series(totals_all).value_counts().sort_index().to_string())

print(f"\nMarker totals by region (mean, sd):")
by_region = pd.DataFrame({
    "region": meta["region"].values,
    "marker_total": totals_all
}).groupby("region")["marker_total"].agg(["mean", "std", "count"])
print(by_region.round(3).to_string())
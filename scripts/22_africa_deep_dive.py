"""
22_africa_deep_dive.py — Understand WHY the model fails on Africa.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

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


# ── 1. What are the African isolates? ─────────────────────────────────────
africa = meta[meta["region"] == "Africa"]
print("=" * 70)
print("AFRICAN ISOLATES — characterisation")
print("=" * 70)
print(f"N = {len(africa)}")
print(f"\nLabel distribution: {africa['label'].value_counts().to_dict()}")
print(f"Resistance rate:    {africa['label'].mean():.1%}")
print(f"\nCountries: {africa['isolation_country'].value_counts().to_dict()}")
print(f"\nMLST types: {africa['mlst'].value_counts().head(10).to_dict()}")
print(f"Unique MLSTs: {africa['mlst'].nunique()}")


# ── 2. What genes are enriched in African isolates? ───────────────────────
X_africa = X[meta["region"].values == "Africa"]
X_non_afr = X[meta["region"].values != "Africa"]

gene_prevalence_africa = X_africa.mean(axis=0)
gene_prevalence_non    = X_non_afr.mean(axis=0)

gene_diff = pd.DataFrame({
    "gene": matrix.columns,
    "prev_africa": gene_prevalence_africa,
    "prev_non_africa": gene_prevalence_non,
    "diff": gene_prevalence_africa - gene_prevalence_non,
})

print("\n" + "=" * 70)
print("GENES ENRICHED IN AFRICA (top 15)")
print("=" * 70)
print(gene_diff.sort_values("diff", ascending=False).head(15).round(3).to_string(index=False))

print("\n" + "=" * 70)
print("GENES DEPLETED IN AFRICA (top 15)")
print("=" * 70)
print(gene_diff.sort_values("diff").head(15).round(3).to_string(index=False))


# ── 3. Try calibrating the model for Africa ───────────────────────────────
print("\n" + "=" * 70)
print("MODEL CALIBRATION TEST — does threshold adjustment help?")
print("=" * 70)
train_mask = meta["region"] != "Africa"
test_mask  = meta["region"] == "Africa"

clf = LogisticRegression(max_iter=3000)
clf.fit(X[train_mask.values], meta.loc[train_mask, "label"].values)
y_prob = clf.predict_proba(X[test_mask.values])[:, 1]
y_test = meta.loc[test_mask, "label"].values

print(f"Africa test set: {len(y_test)} genomes, {y_test.sum()} resistant")
print(f"Predicted probabilities (Africa test set):")
print(f"  min={y_prob.min():.3f}  median={np.median(y_prob):.3f}  max={y_prob.max():.3f}")
print(f"  fraction > 0.5: {(y_prob > 0.5).mean():.1%}")
print(f"\nIf almost all predictions are on one side of 0.5,")
print(f"the model's decision boundary is miscalibrated for this region.")


# ── 4. Also: how many African isolates are in the annotated subset? ───────
print("\n" + "=" * 70)
print("AFRICA IN THE ANNOTATED SUBSET — overall vs annotated")
print("=" * 70)
all_africa = verif[verif["region"] == "Africa"]
print(f"All verified African isolates:      {len(all_africa)}")
print(f"Annotated African isolates:         {len(africa)}")
print(f"Missing AMR gene annotations:       {len(all_africa) - len(africa)}")
print(f"\nResistance rate (all African):      {all_africa['label'].mean():.1%}")
print(f"Resistance rate (annotated African): {africa['label'].mean():.1%}")
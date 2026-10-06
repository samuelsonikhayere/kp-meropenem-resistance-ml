"""
25_africa_model_behaviour.py

Characterise what the model sees when it makes predictions on African isolates.

Questions:
- Do the African resistant isolates have KPC, NDM, OXA-48?
- Or do they resist via different mechanisms?
- What is the model's decision based on?
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

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
cols = list(matrix.columns)

# Train the global model
clf = LogisticRegression(max_iter=5000, C=1.0)
clf.fit(X, meta["label"].values)


# ── 1. Do African isolates have the classic carbapenemases? ───────────────
CARBAPENEMASE_GENES = ["KPC family", "KPC-2", "KPC-3", "NDM family",
                       "NDM-4", "OXA-48 family", "OXA-181", "OXA-232",
                       "VIM", "IMP"]

print("=" * 70)
print("CARBAPENEMASE PRESENCE — Africa vs Global")
print("=" * 70)

afr_mask = (meta["region"] == "Africa").values

print(f"\n{'Gene':<20} {'Africa (n=51)':>15} {'Non-Africa':>15}")
print("-" * 55)
for gene in CARBAPENEMASE_GENES:
    if gene not in cols:
        continue
    i = cols.index(gene)
    afr_prev = X[afr_mask, i].mean()
    non_prev = X[~afr_mask, i].mean()
    print(f"{gene:<20} {afr_prev:>14.1%} {non_prev:>14.1%}")

# Also porins
print(f"\n{'Porin genes':<20} {'Africa':>15} {'Non-Africa':>15}")
print("-" * 55)
for gene in ["ompK35", "ompK36", "omp36", "ompA", "ompK37"]:
    if gene not in cols:
        continue
    i = cols.index(gene)
    afr_prev = X[afr_mask, i].mean()
    non_prev = X[~afr_mask, i].mean()
    print(f"{gene:<20} {afr_prev:>14.1%} {non_prev:>14.1%}")


# ── 2. What does the model predict for African resistant isolates? ────────
print("\n" + "=" * 70)
print("MODEL BEHAVIOUR ON AFRICAN ISOLATES")
print("=" * 70)

# Get African rows only
afr_X = X[afr_mask]
afr_meta = meta[afr_mask].reset_index(drop=True)
afr_prob = clf.predict_proba(afr_X)[:, 1]
afr_meta["predicted_prob"] = afr_prob

# Split by actual label
resistant = afr_meta[afr_meta["label"] == 1]
susceptible = afr_meta[afr_meta["label"] == 0]

print(f"\nAfrican RESISTANT isolates (n={len(resistant)}):")
print(f"  Predicted probability range: {resistant['predicted_prob'].min():.3f} to {resistant['predicted_prob'].max():.3f}")
print(f"  Mean predicted probability:  {resistant['predicted_prob'].mean():.3f}")
print(f"  Fraction predicted > 0.5:    {(resistant['predicted_prob'] > 0.5).mean():.1%}")

print(f"\nAfrican SUSCEPTIBLE isolates (n={len(susceptible)}):")
print(f"  Predicted probability range: {susceptible['predicted_prob'].min():.3f} to {susceptible['predicted_prob'].max():.3f}")
print(f"  Mean predicted probability:  {susceptible['predicted_prob'].mean():.3f}")
print(f"  Fraction predicted > 0.5:    {(susceptible['predicted_prob'] > 0.5).mean():.1%}")

print("\n** If the resistant isolates are predicted as susceptible (or vice versa),")
print("   the model is miscalibrated. If both distributions overlap,")
print("   the model can't distinguish them.")


# ── 3. Feature contributions for the African-resistant isolates ───────────
print("\n" + "=" * 70)
print("WHAT FEATURES DO AFRICAN-RESISTANT ISOLATES HAVE?")
print("=" * 70)

# Average gene presence in African resistant isolates
afr_res_idx = afr_meta["label"].values == 1
afr_res_genes = afr_X[afr_res_idx].mean(axis=0)

# Compare to non-African resistant
non_afr = meta["region"].values != "Africa"
non_afr_res_idx = (meta["label"].values == 1) & non_afr
non_afr_res_genes = X[non_afr_res_idx].mean(axis=0)

# Only show genes present in at least one of the two
mask = (afr_res_genes > 0.3) | (non_afr_res_genes > 0.3)
diff_df = pd.DataFrame({
    "gene": [cols[i] for i in range(len(cols)) if mask[i]],
    "afr_res": afr_res_genes[mask],
    "non_afr_res": non_afr_res_genes[mask],
}).assign(diff=lambda d: d["afr_res"] - d["non_afr_res"])

print("\nGenes MORE common in African-resistant isolates:")
print(diff_df.sort_values("diff", ascending=False).head(15).round(3).to_string(index=False))

print("\nGenes MORE common in non-African-resistant isolates:")
print(diff_df.sort_values("diff").head(15).round(3).to_string(index=False))
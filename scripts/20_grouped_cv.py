"""
20_grouped_cv.py — Test whether AUROC 0.971 survives when clonally
related isolates are kept together in train/test splits.

Two grouping strategies:
  A) GroupKFold by MLST sequence type  (lineage-level splitting)
  B) GroupKFold by region               (geography-level splitting)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

ROOT      = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "processed"

verif  = pd.read_csv(PROCESSED / "verified_final.csv",
                     low_memory=False, dtype={"genome_id": str})
matrix = pd.read_csv(PROCESSED / "X_features.csv",
                     low_memory=False, dtype={"genome_id": str}, index_col=0)

# Align verif to the matrix's index order
verif = verif.set_index("genome_id").loc[matrix.index].reset_index()

# Keep only annotated genomes
annotated = matrix.sum(axis=1) > 0         # boolean Series, index=genome_id
annotated = annotated.loc[verif["genome_id"]].values   # FIX: convert to numpy array

X    = matrix.loc[annotated].values
meta = verif.loc[annotated].reset_index(drop=True)
y    = meta["label"].values

groups_mlst = meta["mlst"].fillna("UNKNOWN").astype(str).values

print(f"Annotated subset: {len(X)} genomes")
print(f"Resistant: {y.sum()} ({y.mean():.1%})")
print(f"Unique MLST types: {len(set(groups_mlst))}")
print(f"MLST 'UNKNOWN' count: {(groups_mlst == 'UNKNOWN').sum()}")


def run_cv(X, y, groups, splitter, label):
    aucs = []
    for train_idx, test_idx in splitter.split(X, y, groups):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[train_idx], y[train_idx])
        y_prob = clf.predict_proba(X[test_idx])[:, 1]
        if len(np.unique(y[test_idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[test_idx], y_prob))
    print(f"\n{label}")
    print(f"  Folds scored: {len(aucs)}")
    print(f"  AUROC mean:   {np.mean(aucs):.3f}")
    print(f"  AUROC SD:     {np.std(aucs):.3f}")
    print(f"  Fold values:  {[round(a, 3) for a in aucs]}")
    return aucs


print("\n" + "=" * 70)
print("BASELINE — random 5-fold CV")
print("=" * 70)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
run_cv(X, y, groups_mlst, skf, "StratifiedKFold (baseline)")


print("\n" + "=" * 70)
print("TEST 1 — GroupKFold by MLST")
print("=" * 70)
n_mlst    = len(set(groups_mlst))
n_splits  = min(5, n_mlst)
gkf       = GroupKFold(n_splits=n_splits)
run_cv(X, y, groups_mlst, gkf, f"GroupKFold by MLST (n_splits={n_splits})")


print("\n" + "=" * 70)
print("TEST 2 — GroupKFold by REGION")
print("=" * 70)
groups_region = meta["region"].fillna("UNKNOWN").astype(str).values
n_region      = len(set(groups_region))
n_splits_r    = min(5, n_region)
gkf_r         = GroupKFold(n_splits=n_splits_r)
run_cv(X, y, groups_region, gkf_r,
       f"GroupKFold by Region (n_splits={n_splits_r})")


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Interpretation:
  - If GroupKFold by MLST stays near 0.97 → signal survives lineage separation.
  - If it drops below 0.75                 → substantial phylogenetic leakage.
  - If GroupKFold by Region drops          → geographic generalisation is poor.
""")
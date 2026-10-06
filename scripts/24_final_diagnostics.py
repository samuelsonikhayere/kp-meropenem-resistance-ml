"""
24_final_diagnostics.py — Two final checks for the paper.

1. Deposit dates for Batch A vs Batch B African isolates.
2. Feature importance of the random-CV model — are batch markers top-ranked?
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

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


# ── 1. Deposit dates ──────────────────────────────────────────────────────
print("=" * 70)
print("DEPOSIT DATE BY BATCH (African isolates)")
print("=" * 70)
afr_mask = (meta["region"] == "Africa").values
afr = meta[afr_mask].copy()
afr_X = X[afr_mask]

BATCH_MARKERS = ["emrD","cpxA","parE","murA","cysB","mfd","alaS",
                 "soxR","kdpE","mdfA","patA","mdtD","nfsA","acrR","cpxR"]
marker_idx = [cols.index(g) for g in BATCH_MARKERS if g in cols]
afr["marker_total"] = afr_X[:, marker_idx].sum(axis=1)
afr["batch"] = np.where(afr["marker_total"] > 5, "Batch_A", "Batch_B")

for batch in ["Batch_A", "Batch_B"]:
    sub = afr[afr["batch"] == batch]
    print(f"\n{batch} (n={len(sub)}):")
    print(f"  date_inserted range: {sub['date_inserted'].min()} to {sub['date_inserted'].max()}")
    print(f"  completion_date range: {sub['completion_date'].min()} to {sub['completion_date'].max()}")


# ── 2. Feature importance ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TOP 20 FEATURES BY |COEF| — random-CV logistic regression")
print("=" * 70)

# Fit once on all data (for coefficient inspection only)
clf = LogisticRegression(max_iter=5000, C=1.0)
clf.fit(X, meta["label"].values)

coef_df = pd.DataFrame({
    "gene": cols,
    "coef": clf.coef_[0],
    "abs_coef": np.abs(clf.coef_[0]),
}).sort_values("abs_coef", ascending=False)

print(coef_df.head(20).round(4).to_string(index=False))

# Check: are batch markers in the top 20?
top20_genes = coef_df.head(20)["gene"].tolist()
batch_in_top = [g for g in BATCH_MARKERS if g in top20_genes]
print(f"\nBatch marker genes in top 20: {len(batch_in_top)}/15")
print(f"  {batch_in_top}")


# ── 3. Batch-marker-only model performance ────────────────────────────────
print("\n" + "=" * 70)
print("AUROC USING ONLY THE 15 BATCH MARKERS")
print("=" * 70)
X_batch = X[:, marker_idx]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = []
for tr, te in skf.split(X_batch, meta["label"].values):
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_batch[tr], meta["label"].values[tr])
    from sklearn.metrics import roc_auc_score
    aucs.append(roc_auc_score(meta["label"].values[te],
                              clf.predict_proba(X_batch[te])[:, 1]))
print(f"AUROC with only 15 batch markers: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
print(f"AUROC with all 827 genes:         0.971 ± 0.009")
print(f"Fraction of signal from 15 markers: {np.mean(aucs) / 0.971:.1%}")
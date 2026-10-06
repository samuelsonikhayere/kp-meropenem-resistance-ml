"""
26_africa_holdout_predictions.py

CORRECT version of the African prediction test:
Train on non-Africa, predict on Africa (proper held-out evaluation).
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

# ── Train on non-Africa, test on Africa ───────────────────────────────────
train_mask = (meta["region"] != "Africa").values
test_mask  = (meta["region"] == "Africa").values

X_train = X[train_mask]
y_train = meta.loc[train_mask, "label"].values
X_test  = X[test_mask]
y_test  = meta.loc[test_mask, "label"].values

clf = LogisticRegression(max_iter=5000, C=1.0)
clf.fit(X_train, y_train)
y_prob = clf.predict_proba(X_test)[:, 1]

print("=" * 70)
print("AFRICA HELD-OUT EVALUATION (proper)")
print("=" * 70)
print(f"Train n = {len(X_train)}  positives = {y_train.sum()} ({y_train.mean():.1%})")
print(f"Test  n = {len(X_test)}   positives = {y_test.sum()} ({y_test.mean():.1%})")
print(f"\nAUROC: {roc_auc_score(y_test, y_prob):.3f}")
print(f"AUPRC: {average_precision_score(y_test, y_prob):.3f}")

# Look at prediction distribution by true label
print(f"\nPredicted probabilities by true label:")
for lab in [0, 1]:
    mask = y_test == lab
    p = y_prob[mask]
    print(f"  label={lab} (n={mask.sum()}): "
          f"min={p.min():.3f}  median={np.median(p):.3f}  max={p.max():.3f}")

# How many African isolates does the model get right?
pred_binary = (y_prob > 0.5).astype(int)
print(f"\nPredictions >0.5: {(pred_binary == 1).sum()} / {len(y_test)}")
print(f"Correct classification: {(pred_binary == y_test).mean():.1%}")
print(f"Accuracy of majority class baseline: {max(y_test.mean(), 1 - y_test.mean()):.1%}")


# ── Same evaluation, using only the batch markers as features ─────────────
print("\n" + "=" * 70)
print("COMPARISON: batch markers alone, Africa held out")
print("=" * 70)
BATCH_MARKERS = ["emrD","cpxA","parE","murA","cysB","mfd","alaS",
                 "soxR","kdpE","mdfA","patA","mdtD","nfsA","acrR","cpxR"]
cols = list(matrix.columns)
marker_idx = [cols.index(g) for g in BATCH_MARKERS if g in cols]

clf2 = LogisticRegression(max_iter=3000)
clf2.fit(X_train[:, marker_idx], y_train)
prob2 = clf2.predict_proba(X_test[:, marker_idx])[:, 1]
print(f"AUROC with 15 batch markers only: {roc_auc_score(y_test, prob2):.3f}")


# ── The comparison that matters ───────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Model trained on non-Africa, tested on Africa:")
print(f"  All 827 genes:     AUROC = {roc_auc_score(y_test, y_prob):.3f}")
print(f"  15 batch markers:  AUROC = {roc_auc_score(y_test, prob2):.3f}")


# ── What does the honest model predict for African-resistant isolates? ───
print("\n" + "=" * 70)
print("PREDICTIONS FOR AFRICAN-RESISTANT ISOLATES (held-out model)")
print("=" * 70)
afr_meta = meta[test_mask].reset_index(drop=True)
afr_meta["predicted"] = y_prob

res = afr_meta[afr_meta["label"] == 1]
sus = afr_meta[afr_meta["label"] == 0]
print(f"\nResistant Africans (n={len(res)}):")
print(f"  Mean predicted prob: {res['predicted'].mean():.3f}")
print(f"  Fraction > 0.5:      {(res['predicted'] > 0.5).mean():.1%}")
print(f"\nSusceptible Africans (n={len(sus)}):")
print(f"  Mean predicted prob: {sus['predicted'].mean():.3f}")
print(f"  Fraction < 0.5:      {(sus['predicted'] < 0.5).mean():.1%}")
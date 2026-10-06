"""
21_region_holdout.py — Leave-one-region-out CV, reporting per-region AUROC.

Tells us which regions the model transfers to and which it doesn't.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

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

print(f"Annotated subset: {len(X)}")


def evaluate_holdout(held_region):
    """Train on everything except held_region, test on held_region."""
    train_mask = meta["region"] != held_region
    test_mask  = meta["region"] == held_region

    if test_mask.sum() < 10:
        return None

    X_train, y_train = X[train_mask.values], meta.loc[train_mask, "label"].values
    X_test,  y_test  = X[test_mask.values],  meta.loc[test_mask,  "label"].values

    if len(np.unique(y_test)) < 2:
        return None

    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]

    return {
        "region":     held_region,
        "n_train":    int(train_mask.sum()),
        "n_test":     int(test_mask.sum()),
        "prev_test":  float(y_test.mean()),
        "prev_train": float(y_train.mean()),
        "AUROC":      roc_auc_score(y_test, y_prob),
        "AUPRC":      average_precision_score(y_test, y_prob),
    }


print("\n" + "=" * 80)
print("LEAVE-ONE-REGION-OUT EVALUATION")
print("=" * 80)

results = []
for region in sorted(meta["region"].unique()):
    r = evaluate_holdout(region)
    if r:
        results.append(r)

df_res = pd.DataFrame(results).sort_values("AUROC", ascending=False)
print(df_res.round(3).to_string(index=False))


# ── Compare to a "global" baseline: random 5-fold on same subset ──────────
from sklearn.model_selection import StratifiedKFold
skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y    = meta["label"].values
aucs = []
for tr, te in skf.split(X, y):
    clf = LogisticRegression(max_iter=3000)
    clf.fit(X[tr], y[tr])
    aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
print(f"\nRandom 5-fold reference AUROC:  {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")


# ── Also test the honest African question ─────────────────────────────────
print("\n" + "=" * 80)
print("AFRICA HOLDOUT — the paper's headline number")
print("=" * 80)
africa_result = evaluate_holdout("Africa")
if africa_result:
    for k, v in africa_result.items():
        print(f"  {k:12s}: {v}")

# Also: train on Africa only (tiny), test on rest
print("\n" + "=" * 80)
print("TRAIN AFRICA-ONLY, TEST ON NON-AFRICA")
print("=" * 80)
train_mask = meta["region"] == "Africa"
test_mask  = meta["region"] != "Africa"
if train_mask.sum() >= 20:
    X_tr, y_tr = X[train_mask.values], meta.loc[train_mask, "label"].values
    X_te, y_te = X[test_mask.values],  meta.loc[test_mask,  "label"].values
    if len(np.unique(y_tr)) >= 2:
        clf = LogisticRegression(max_iter=3000)
        clf.fit(X_tr, y_tr)
        y_prob = clf.predict_proba(X_te)[:, 1]
        print(f"  n_train (Africa):   {train_mask.sum()}")
        print(f"  n_test  (non-Afr):  {test_mask.sum()}")
        print(f"  AUROC on non-Africa: {roc_auc_score(y_te, y_prob):.3f}")

df_res.to_csv(PROCESSED / "region_holdout_results.csv", index=False)
print(f"\n✓ Saved: processed/region_holdout_results.csv")
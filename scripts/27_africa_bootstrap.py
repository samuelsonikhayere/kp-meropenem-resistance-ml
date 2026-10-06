"""
27_africa_bootstrap.py

Bootstrap 95% CI for the African held-out AUROC (n=51 is small).
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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

train_mask = (meta["region"] != "Africa").values
test_mask  = (meta["region"] == "Africa").values

clf = LogisticRegression(max_iter=5000, C=1.0)
clf.fit(X[train_mask], meta.loc[train_mask, "label"].values)
y_prob = clf.predict_proba(X[test_mask])[:, 1]
y_test = meta.loc[test_mask, "label"].values

point_auc = roc_auc_score(y_test, y_prob)
print(f"Africa held-out AUROC: {point_auc:.3f}")

# Bootstrap
rng = np.random.default_rng(42)
boot_aucs = []
for _ in range(2000):
    idx = rng.integers(0, len(y_test), len(y_test))
    if len(np.unique(y_test[idx])) < 2:
        continue
    boot_aucs.append(roc_auc_score(y_test[idx], y_prob[idx]))

boot_aucs = np.array(boot_aucs)
print(f"Bootstrap 95% CI: [{np.percentile(boot_aucs, 2.5):.3f}, {np.percentile(boot_aucs, 97.5):.3f}]")
print(f"Bootstrap mean:   {boot_aucs.mean():.3f}")
print(f"Fraction > 0.5:   {(boot_aucs > 0.5).mean():.1%}")
print(f"Fraction > 0.7:   {(boot_aucs > 0.7).mean():.1%}")
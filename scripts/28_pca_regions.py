"""
28_pca_regions.py — PCA of the 827-gene matrix, coloured by region.

Shows whether African isolates cluster separately from the training pool.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

ROOT      = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "processed"
RESULTS   = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

verif  = pd.read_csv(PROCESSED / "verified_final.csv",
                     low_memory=False, dtype={"genome_id": str})
matrix = pd.read_csv(PROCESSED / "X_features.csv",
                     low_memory=False, dtype={"genome_id": str}, index_col=0)
verif = verif.set_index("genome_id").loc[matrix.index].reset_index()
annotated = (matrix.sum(axis=1) > 0).loc[verif["genome_id"]].values

X    = matrix.loc[annotated].values
meta = verif.loc[annotated].reset_index(drop=True)

# PCA
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)

plt.figure(figsize=(10, 8))
for region, colour in zip(
    ["North America", "Asia", "Europe", "Oceania", "Africa",
     "South America", "Unknown"],
    ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red",
     "tab:brown", "tab:gray"],
):
    mask = meta["region"].values == region
    if mask.sum() == 0:
        continue
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                label=f"{region} (n={mask.sum()})",
                alpha=0.6, s=30, c=colour)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.title("PCA of AMR gene presence/absence matrix — by region")
plt.legend(loc="best", fontsize=9)
plt.tight_layout()
plt.savefig(RESULTS / "pca_by_region.png", dpi=150)
print(f"✓ Saved: results/pca_by_region.png")

# Also colour by label
plt.figure(figsize=(10, 8))
for label, colour, name in [(0, "tab:blue", "Susceptible"),
                             (1, "tab:red", "Resistant")]:
    mask = meta["label"].values == label
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                label=f"{name} (n={mask.sum()})",
                alpha=0.6, s=30, c=colour)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.title("PCA of AMR gene presence/absence matrix — by phenotype")
plt.legend(loc="best", fontsize=9)
plt.tight_layout()
plt.savefig(RESULTS / "pca_by_label.png", dpi=150)
print(f"✓ Saved: results/pca_by_label.png")
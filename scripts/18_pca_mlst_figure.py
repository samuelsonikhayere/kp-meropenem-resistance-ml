"""
18_pca_mlst_figure.py — Supplementary figure: PCA by MLST lineage.

Confirms feature-space structure supports the MLST-grouped CV design.
"""

from __future__ import annotations
from importlib.machinery import SourceFileLoader
from pathlib import Path as _P

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("18_pca_mlst_figure")

table = pd.read_csv(setup.PROCESSED / "06_analysis_table.csv",
                    low_memory=False, dtype={"genome_id": str})

assert len(table) == 3694, f"Expected 3694 annotated, got {len(table)}"

feat_cols = setup.feature_columns(table)
X = table[feat_cols].values.astype("float32")
mlst = table["mlst"].fillna("UNKNOWN").astype(str)

pca = PCA(n_components=2, random_state=setup.RANDOM_SEED)
X_2d = pca.fit_transform(X)

top_mlst = mlst.value_counts().head(10).index.tolist()
colors = plt.cm.tab10(np.linspace(0, 1, len(top_mlst)))

fig, ax = plt.subplots(figsize=(7.2, 5.6))

mask_other = ~mlst.isin(top_mlst)
ax.scatter(X_2d[mask_other, 0], X_2d[mask_other, 1],
           c="#d9d9d9", s=8, alpha=0.5, edgecolors="none",
           label=f"Other MLST ({int(mask_other.sum())})")

for i, m in enumerate(top_mlst):
    mask = mlst == m
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=[colors[i]], s=18, alpha=0.85, edgecolors="none",
               label=f"{m} (n={int(mask.sum())})")

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title("Feature space structure by MLST lineage", fontsize=10)
ax.legend(loc="best", fontsize=7, frameon=False, ncol=2)
fig.tight_layout()
fig.savefig(setup.FIGURES / "figureS_pca_mlst.pdf", bbox_inches="tight")
fig.savefig(setup.FIGURES / "figureS_pca_mlst.png", bbox_inches="tight", dpi=300)
plt.close(fig)
log.info("Saved: figures/figureS_pca_mlst.pdf / .png")
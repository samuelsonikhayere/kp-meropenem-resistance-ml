"""
20_pca_region_figure.py — Figure 2: PCA of feature space by region.

Loads the exact Batch 3 analysis table (3,694 genomes × 377 features),
runs PCA, and colours points by region.

Asserts:
  - 3,694 genomes (annotated primary cohort)
  - 79 African isolates

Outputs:
  figures/figure2_pca_region.pdf
  figures/figure2_pca_region.png
"""

from __future__ import annotations
from importlib.machinery import SourceFileLoader
from pathlib import Path as _P

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("20_pca_region_figure")


# ─────────────────────────────────────────────────────────────────────────
# Load analysis table
# ─────────────────────────────────────────────────────────────────────────
table = pd.read_csv(setup.PROCESSED / "06_analysis_table.csv",
                    low_memory=False, dtype={"genome_id": str})

assert len(table) == 3694, f"Expected 3694 annotated genomes, got {len(table)}"
africa_n = int((table["region"] == "Africa").sum())
assert africa_n == 79, f"Expected 79 African isolates, got {africa_n}"

feat_cols = setup.feature_columns(table)
X = table[feat_cols].values.astype("float32")
region = table["region"].values

log.info(f"Cohort: {len(table)} genomes × {len(feat_cols)} features")
log.info(f"African isolates: {africa_n}")


# ─────────────────────────────────────────────────────────────────────────
# PCA
# ─────────────────────────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=setup.RANDOM_SEED)
X_2d = pca.fit_transform(X)
evr = pca.explained_variance_ratio_
log.info(f"Explained variance — PC1: {evr[0]:.3f}, PC2: {evr[1]:.3f}")


# ─────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":  300,
})

# Order: draw large regions first, then small ones so they are visible on top
REGION_ORDER = [
    ("North America", "#3b6fb5", 12),
    ("Asia",          "#e08a3c", 14),
    ("Europe",        "#4a934a", 16),
    ("Unknown",       "#8c8c8c", 14),
    ("Oceania",       "#9b6fb5", 20),
    ("Africa",        "#d44a4a", 26),
    ("South America", "#7a5c3b", 32),
]

fig, ax = plt.subplots(figsize=(7.5, 5.6))

for name, colour, size in REGION_ORDER:
    mask = region == name
    if mask.sum() == 0:
        continue
    ax.scatter(
        X_2d[mask, 0], X_2d[mask, 1],
        c=colour, s=size, alpha=0.80,
        edgecolors="#222", linewidths=0.25,
        label=f"{name} (n={int(mask.sum())})",
    )

ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% variance)")
ax.set_title("Feature space structure by geographic region",
             fontsize=10.5, pad=8)

ax.legend(
    loc="best", fontsize=8,
    frameon=False, ncol=1,
    markerscale=1.4,
)

fig.tight_layout()
fig.savefig(setup.FIGURES / "figure2_pca_region.pdf", bbox_inches="tight")
fig.savefig(setup.FIGURES / "figure2_pca_region.png", bbox_inches="tight", dpi=300)
plt.close(fig)
log.info("Saved: figures/figure2_pca_region.pdf / .png")


# ─────────────────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────────────────
setup.write_result("20_pca_region_figure", {
    "analysis_name":   "pca_by_region",
    "dataset_version": "api_v1",
    "n_genomes":       int(len(table)),
    "n_features":      int(len(feat_cols)),
    "african_isolates": africa_n,
    "pc1_variance":    float(evr[0]),
    "pc2_variance":    float(evr[1]),
    "software_version": setup.software_versions(),
})
log.info("Done.")
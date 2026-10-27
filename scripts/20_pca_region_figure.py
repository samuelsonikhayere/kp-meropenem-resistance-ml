"""20: Figure 2, PCA of the AMR feature matrix coloured by region."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from fig_style import *
df, sc = pca_scores()
fig, ax = plt.subplots(figsize=(6.2, 4.6))
for r in ["North America", "Asia", "Europe", "Oceania", "Not recorded", "South America", "Africa"]:
    m = (df.region == r).values
    ax.scatter(sc.PC1[m], sc.PC2[m], s=9 if r not in ("Africa", "South America") else 16, c=REGION_COL[r], alpha=.55 if r != "Africa" else .9,
               edgecolors="none", label=f"{r} (n = {m.sum():,})", zorder=3 if r in ("Africa", "South America") else 2)
ax.set_xlabel(f"PC1 ({100 * sc.var1[0]:.1f}% of variance)"); ax.set_ylabel(f"PC2 ({100 * sc.var2[0]:.1f}% of variance)")
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, .5), markerscale=1.8)
save(fig, "Figure2_pca_region")

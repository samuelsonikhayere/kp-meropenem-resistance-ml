"""18: Figure S1, the same PCA coloured by the eight most common sequence types."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from fig_style import *
df, sc = pca_scores(); st = df.mlst.fillna("Not recorded").astype(str).str.replace("MLST.klebsiella.", "ST", regex=False)
top = st[st != "Not recorded"].value_counts().head(8).index.tolist()
pal = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#000000"]
fig, ax = plt.subplots(figsize=(6.2, 4.6))
o = ~st.isin(top).values; ax.scatter(sc.PC1[o], sc.PC2[o], s=7, c="#D0D0D0", edgecolors="none", label=f"Other (n = {o.sum():,})")
for t, c in zip(top, pal):
    m = (st == t).values; ax.scatter(sc.PC1[m], sc.PC2[m], s=10, c=c, alpha=.8, edgecolors="none", label=f"{t} (n = {m.sum():,})")
ax.set_xlabel(f"PC1 ({100 * sc.var1[0]:.1f}% of variance)"); ax.set_ylabel(f"PC2 ({100 * sc.var2[0]:.1f}% of variance)")
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, .5), markerscale=1.8)
save(fig, "FigureS1_pca_mlst")

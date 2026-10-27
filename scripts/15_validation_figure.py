"""15: Figure 3, discrimination under the three CV-style schemes (bootstrap 95% CIs from table S4)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from fig_style import *
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

ci = pd.read_csv(S.RESULTS / "tableS4_bootstrap_cis.csv"); t4 = pd.read_csv(S.RESULTS / "table4_validation_schemes.csv")
schemes = ["Random 5-fold", "MLST-grouped 5-fold", "Temporal holdout"]; lab = ["Random\n5-fold", "Lineage (MLST)\ngrouped", "Temporal\nholdout"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
for ax, key, L in zip(axs, ["AUROC", "AUPRC"], "AB"):
    for j, m in enumerate(S.MODELS):
        for i, sc in enumerate(schemes):
            x = i + (j - .5) * .28
            if key == "AUROC":
                nm = {"Random 5-fold": "Random stratified 5-fold", "MLST-grouped 5-fold": "MLST-grouped 5-fold", "Temporal holdout": "Temporal (train <2016, test 2016 onward)"}[sc]
                r = t4[(t4.Model == m) & (t4.Scheme == nm)].iloc[0]; sd = r["AUROC fold SD"]; sd = 0 if pd.isna(sd) else sd
                ax.errorbar(x, r["AUROC (mean of folds)"], yerr=sd, fmt="o", color=MODEL_COL[m], capsize=3 if sd else 0, ms=5, lw=1.2, label=MODEL_LABEL[m] if i == 0 else None)
            else:
                nm = {"Random 5-fold": "Random stratified 5-fold", "MLST-grouped 5-fold": "MLST-grouped 5-fold", "Temporal holdout": "Temporal (train <2016, test 2016 onward)"}[sc]
                r = t4[(t4.Model == m) & (t4.Scheme == nm)].iloc[0]; sd = r["AUPRC fold SD"]; sd = 0 if pd.isna(sd) else sd
                ax.errorbar(x, r["AUPRC (mean of folds)"], yerr=sd, fmt="o", color=MODEL_COL[m], capsize=3 if sd else 0, ms=5, lw=1.2, label=MODEL_LABEL[m] if i == 0 else None)
    ax.set_xticks(range(3)); ax.set_xticklabels(lab); ax.set_ylabel(key); ax.set_xlim(-.6, 2.6); ax.set_ylim(.88, 1.0); ax.grid(axis="y", color="#E5E5E5", lw=.6); letter(ax, L)
axs[0].legend(frameon=False, loc="lower left")
fig.tight_layout(); save(fig, "Figure3_validation_schemes")

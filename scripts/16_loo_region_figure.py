"""16: Figure 4, leave-one-region-out AUROC with bootstrap CIs (A) and the effect of adding local isolates (B)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from fig_style import *
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

ci = pd.read_csv(S.RESULTS / "tableS4_bootstrap_cis.csv"); lc = pd.read_csv(S.RESULTS / "tableS10_learning_curve_local_data.csv"); t6 = pd.read_csv(S.RESULTS / "table6_region_holdout.csv")
regs = t6.Region.tolist()[::-1]
fig, (a, b) = plt.subplots(1, 2, figsize=(8.0, 3.8), gridspec_kw={"width_ratios": [1.45, 1]})
for j, m in enumerate(S.MODELS):
    for i, r in enumerate(regs):
        d = ci[(ci.Model == m) & (ci.Scheme == f"Held-out region: {r}")].iloc[0]; y = i + (j - .5) * .3
        a.errorbar(d.AUROC, y, xerr=[[d.AUROC - d.CI_low], [d.CI_high - d.AUROC]], fmt="o", color=MODEL_COL[m], capsize=2.5, ms=4.5, lw=1.1, label=MODEL_LABEL[m] if i == 0 else None)
a.axvline(.5, color="#999", ls=":", lw=1)
lab = [f"{r}\nn = {int(t6.set_index('Region').loc[r, 'n'])}, {t6.set_index('Region').loc[r, 'Resistant_pct']:.0f}% resistant" for r in regs]
a.set_yticks(range(len(regs))); a.set_yticklabels(lab, fontsize=7.5); a.set_xlabel("AUROC on the held-out region"); a.set_xlim(.2, 1.02); a.set_ylim(-.6, len(regs) - .4)
a.legend(frameon=False, loc="lower left"); letter(a, "A", -.38)
for r, c in [("Europe", REGION_COL["Europe"]), ("Oceania", REGION_COL["Oceania"])]:
    d = lc[lc.Region == r]; b.errorbar(d["Local isolates added"], d["AUROC mean"], yerr=d["AUROC SD"].fillna(0), marker="o", color=c, capsize=2.5, ms=4.5, lw=1.2, label=r)
b.set_xlabel("Local isolates added to training"); b.set_ylabel("AUROC (logistic regression)"); b.set_ylim(.5, 1.0); b.legend(frameon=False, loc="lower right"); b.grid(axis="y", color="#E5E5E5", lw=.6); letter(b, "B")
fig.tight_layout(); save(fig, "Figure4_leave_one_region_out")

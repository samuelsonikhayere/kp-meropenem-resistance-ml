"""13: Figures 5, 7 and S2 from the pooled out-of-fold predictions (ROC, precision-recall, calibration, decision curve, regional calibration shift, confusion matrix)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from fig_style import *
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

oof = pd.read_csv(S.RESULTS / "oof_predictions.csv"); y = oof.label.values; df = S.load_table(); f = S.feature_columns(df)
X = df[f].values; ci = [f.index(c) for c in S.carbapenemase_columns(f)]; rule = (X[:, ci].sum(1) > 0).astype(int)
sens = rule[y == 1].mean(); spec = 1 - rule[y == 0].mean()
# Figure 5
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.5))
for m in S.MODELS:
    p = oof[f"{m}__random"].values; fpr, tpr, _ = roc_curve(y, p); pr, rc, _ = precision_recall_curve(y, p)
    a.plot(fpr, tpr, color=MODEL_COL[m], lw=1.6, label=f"{MODEL_LABEL[m]} (AUROC {S.compute_metrics(y, p)['auroc']:.3f})")
    b.plot(rc, pr, color=MODEL_COL[m], lw=1.6, label=f"{MODEL_LABEL[m]} (AUPRC {S.compute_metrics(y, p)['auprc']:.3f})")
a.plot([0, 1], [0, 1], color="#999", ls=":", lw=1); a.plot(1 - spec, sens, "s", color=C["rule"], ms=5, label="Carbapenemase rule")
b.axhline(y.mean(), color="#999", ls=":", lw=1); b.plot(sens, rule[y == 1].sum() / rule.sum(), "s", color=C["rule"], ms=5, label="Carbapenemase rule")
a.set_xlabel("False positive rate"); a.set_ylabel("True positive rate"); b.set_xlabel("Recall"); b.set_ylabel("Precision"); b.set_ylim(.4, 1.02)
for ax, L in zip((a, b), "AB"): ax.legend(frameon=False, loc="lower right" if L == "A" else "lower left", fontsize=7.5); letter(ax, L)
fig.tight_layout(); save(fig, "Figure5_roc_pr")
# Figure 7
fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.5)); a, b, c = axs
a.plot([0, 1], [0, 1], color="#999", ls=":", lw=1)
for m in S.MODELS:
    p = oof[f"{m}__random"].values; q = pd.qcut(p, 10, duplicates="drop"); g = pd.DataFrame({"p": p, "y": y}).groupby(q, observed=True).mean()
    a.plot(g.p, g.y, "o-", color=MODEL_COL[m], ms=4, lw=1.2, label=MODEL_LABEL[m])
a.set_xlabel("Mean predicted probability"); a.set_ylabel("Observed fraction resistant"); a.legend(frameon=False, loc="upper left")
th = np.linspace(.01, .99, 99); nb = lambda yt, pr_, t_: (((pr_ >= t_) & (yt == 1)).sum() - ((pr_ >= t_) & (yt == 0)).sum() * t_ / (1 - t_)) / len(yt)
for m in S.MODELS: b.plot(th, [nb(y, oof[f"{m}__random"].values, t_) for t_ in th], color=MODEL_COL[m], lw=1.5, label=MODEL_LABEL[m])
b.plot(th, [y.mean() - (1 - y.mean()) * t_ / (1 - t_) for t_ in th], color="#777", ls="--", lw=1.1, label="Treat all"); b.axhline(0, color="k", lw=.8, label="Treat none")
b.set_ylim(-.1, .45); b.set_xlim(0, 1); b.set_xlabel("Threshold probability"); b.set_ylabel("Net benefit"); b.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower center", bbox_to_anchor=(.5, 1.0))
e = pd.read_csv(S.RESULTS / "tableS11_recalibration.csv").iloc[::-1].reset_index(drop=True)
for i, r in e.iterrows():
    c.plot([r["Observed prevalence"], r["Mean predicted risk"]], [i, i], color="#BBB", lw=1.2, zorder=1)
c.scatter(e["Observed prevalence"], range(len(e)), color="k", s=22, zorder=3, label="Observed"); c.scatter(e["Mean predicted risk"], range(len(e)), color=C["lr"], s=22, zorder=3, label="Predicted (held-out)")
c.set_yticks(range(len(e))); c.set_yticklabels(e.Region, fontsize=7.5); c.set_xlabel("Resistance prevalence"); c.set_xlim(0, 1); c.legend(frameon=False, loc="lower center", bbox_to_anchor=(.4, 1.0), ncol=2, fontsize=7.5)
for ax, L in zip(axs, "ABC"): letter(ax, L, -.2 if L != "C" else -.4)
fig.tight_layout(); save(fig, "Figure7_calibration_dca")
# Figure S2
cm = confusion_matrix(y, (oof["logistic_regression__random"] >= .5).astype(int)); fig, ax = plt.subplots(figsize=(3.6, 3.2))
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2): ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=11, color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Susceptible", "Resistant"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["Susceptible", "Resistant"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Observed"); [s.set_visible(True) for s in ax.spines.values()]
fig.tight_layout(); save(fig, "FigureS2_confusion_matrix")
tn, fp, fn, tp = cm.ravel(); S.write_result("13_model_diagnostics", {"confusion_LR_random_0.5": dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), sensitivity=tp / (tp + fn), specificity=tn / (tn + fp)),
    "carbapenemase_rule": dict(sensitivity=float(sens), specificity=float(spec))})

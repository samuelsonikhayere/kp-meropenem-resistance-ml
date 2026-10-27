"""17: Figure 6, the 15 strongest features in each model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from fig_style import *
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

t = pd.read_csv(S.RESULTS / "table7_features_logreg.csv")
l = t.sort_values("coefficient", key=abs, ascending=False).head(15).iloc[::-1]; x = t.sort_values("xgb_importance", ascending=False).head(15).iloc[::-1]
fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 4.4))
col = lambda d: [C["xgb"] if c else C["grey"] for c in d.carbapenemase_product]
a.barh(range(15), l.coefficient, color=[C["xgb"] if c else (C["lr"] if v > 0 else "#7FB2D6") for c, v in zip(l.carbapenemase_product, l.coefficient)], height=.7)
a.set_yticks(range(15)); a.set_yticklabels([short(n) for n in l.feature], fontsize=7); a.axvline(0, color="k", lw=.8); a.set_xlabel("Logistic regression coefficient"); letter(a, "A", -.9)
b.barh(range(15), x.xgb_importance, color=col(x), height=.7); b.set_yticks(range(15)); b.set_yticklabels([short(n) for n in x.feature], fontsize=7); b.set_xlabel("XGBoost importance (gain share)"); letter(b, "B", -.9)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=C["xgb"], label="Carbapenemase product"), Patch(color=C["lr"], label="Other, raises predicted risk"), Patch(color="#7FB2D6", label="Other, lowers predicted risk"), Patch(color=C["grey"], label="Other product (XGBoost)")],
           loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.04))
fig.tight_layout(rect=(0, .03, 1, 1)); save(fig, "Figure6_feature_importance")

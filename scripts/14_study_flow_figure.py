"""14: Figure 1, cohort selection and analysis flow. Numbers are read from the cascade table, never typed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from fig_style import *
from matplotlib.patches import FancyBboxPatch
cas = pd.read_csv(S.RESULTS / "table2_filter_cascade.csv"); full = S.load_table(True)
g = lambda k: int(cas[cas.stage.str.contains(k, regex=False)].iloc[0].records)
raw, merged, phen, binary, ver, final = g("Raw meropenem"), g("inner merge"), g("missing phenotype"), g("R/S"), g("verified-method"), g("final cohort")
n_ann = int(full.has_annotation.sum())
fig, ax = plt.subplots(figsize=(7.0, 9.0)); ax.set_xlim(0, 10); ax.set_ylim(0, 21); ax.axis("off")
def box(x, y, w, h, text, fc="#F2F6FA", ec="#33566E", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, fontweight="bold" if bold else "normal", linespacing=1.35)
def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#33566E", lw=1, shrinkA=0, shrinkB=0))
mx, mw, ex, ew, h, eh, step = 0.4, 4.6, 5.9, 3.7, 1.2, 1.0, 2.6
cx = mx + mw / 2
main = [(f"BV-BRC meropenem records\nn = {raw:,}", None),
        (f"Joined to genome metadata\nn = {merged:,}", f"Lost at the join\nn = {raw - merged:,}"),
        (f"Phenotype recorded\nn = {phen:,}", f"Phenotype missing\nn = {merged - phen:,}"),
        (f"Resistant or susceptible\nn = {binary:,}", f"Intermediate or\ndose dependent, n = {phen - binary:,}"),
        (f"Laboratory verified method\nn = {ver:,}", f"Not laboratory verified\nn = {binary - ver:,}"),
        (f"Verified cohort, one record per genome\nn = {final:,}", f"Duplicate genome records\nn = {ver - final:,}")]
top = 19.3; ys = []
for i, (t, e) in enumerate(main):
    y = top - i * step; ys.append(y); box(mx, y, mw, h, t, bold=(i == len(main) - 1))
    if i:
        arrow(cx, ys[i - 1], cx, y + h)
        box(ex, y + h + (step - h - eh) / 2, ew, eh, e, fc="#FBF3EA", ec="#9A5B1B")
        ym = y + h + (step - h) / 2; arrow(cx, ym, ex, ym)
y = ys[-1] - step; box(mx, y, mw, h, f"Annotated cohort used for modelling\nn = {n_ann:,}")
arrow(cx, ys[-1], cx, y + h)
box(ex, y + h + (step - h - eh) / 2, ew, eh, f"No AMR annotation\nn = {final - n_ann:,}", fc="#FBF3EA", ec="#9A5B1B")
ym = y + h + (step - h) / 2; arrow(cx, ym, ex, ym)
names = ["Random 5-fold", "Lineage (MLST)\ngrouped", "Temporal\nholdout", "Leave-one-\nregion-out"]
ybar = y - 0.9; ax.plot([1.45, 8.55], [ybar, ybar], color="#33566E", lw=1); ax.plot([cx, cx], [y, ybar], color="#33566E", lw=1)
for i, t in enumerate(names):
    x = 0.4 + i * 2.35; box(x, ybar - 1.9, 2.15, 1.4, t, fc="#EEF6F0", ec="#2F6B4F"); arrow(x + 1.075, ybar, x + 1.075, ybar - 0.5)
ax.text(5, ybar - 2.4, "Validation analyses", ha="center", fontsize=8.5, color="#2F6B4F", style="italic")
save(fig, "Figure1_study_flow")

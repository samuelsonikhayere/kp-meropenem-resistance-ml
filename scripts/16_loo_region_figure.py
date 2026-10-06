"""
16_loo_region_figure.py — Figure 4: leave-one-region-out AUROC.

Horizontal bar chart, one bar per region, LR and XGBoost side by side.
Regions ordered by LR performance (best at top).
"""

from __future__ import annotations
from importlib.machinery import SourceFileLoader
from pathlib import Path as _P

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("16_loo_region_figure")

lr  = pd.read_csv(setup.RESULTS / "table5_region_holdout__logistic_regression.csv")
xgb = pd.read_csv(setup.RESULTS / "table5_region_holdout__xgboost.csv")

both = lr[["region", "n_test", "auroc"]].rename(columns={"auroc": "lr_auroc"})
both = both.merge(xgb[["region", "auroc"]].rename(columns={"auroc": "xgb_auroc"}),
                  on="region", how="outer")

boot = pd.read_csv(setup.RESULTS / "tableS4_bootstrap_cis.csv")
both = both.merge(boot[["region", "auroc_ci_lo", "auroc_ci_hi"]],
                  on="region", how="left")

africa_row = both[both["region"] == "Africa"]
assert len(africa_row) == 1 and africa_row.iloc[0]["n_test"] == 79, \
    f"Expected 79 African test isolates, got {africa_row.iloc[0]['n_test']}"

both = both.sort_values("lr_auroc", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(7.0, 4.8))
y = np.arange(len(both))
bar_h = 0.36

lr_lo = (both["lr_auroc"] - both["auroc_ci_lo"]).fillna(0)
lr_hi = (both["auroc_ci_hi"] - both["lr_auroc"]).fillna(0)
lr_err = np.array([lr_lo, lr_hi])

ax.barh(y + bar_h/2, both["lr_auroc"], height=bar_h,
        xerr=lr_err, capsize=3,
        color="#3b6fb5", edgecolor="#222", linewidth=0.5,
        label="Logistic regression")

ax.barh(y - bar_h/2, both["xgb_auroc"], height=bar_h,
        color="#c1573b", edgecolor="#222", linewidth=0.5,
        label="XGBoost")

ylabels = [f"{r}  (n={int(n)})" for r, n in zip(both["region"], both["n_test"])]
ax.set_yticks(y)
ax.set_yticklabels(ylabels, fontsize=9)
ax.set_xlabel("Leave-one-region-out AUROC")
ax.set_xlim(0.4, 1.02)
ax.axvline(0.5, color="grey", linestyle=":", linewidth=0.6, label="Chance")
ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)

for yi, (lr_a, xgb_a) in enumerate(zip(both["lr_auroc"], both["xgb_auroc"])):
    ax.text(lr_a + 0.012, yi + bar_h/2, f"{lr_a:.3f}",
            va="center", fontsize=8)
    ax.text(xgb_a + 0.012, yi - bar_h/2, f"{xgb_a:.3f}",
            va="center", fontsize=8)

ax.legend(loc="lower right", fontsize=8.5, frameon=False)
ax.set_title("Heterogeneous geographic generalisability of meropenem "
             "resistance prediction", fontsize=10, pad=8)
fig.tight_layout()
fig.savefig(setup.FIGURES / "figure4_loo_region.pdf", bbox_inches="tight")
fig.savefig(setup.FIGURES / "figure4_loo_region.png", bbox_inches="tight", dpi=300)
plt.close(fig)
log.info("Saved: figures/figure4_loo_region.pdf / .png")
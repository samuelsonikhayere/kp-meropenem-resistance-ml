"""
17_feature_importance_figure.py — Figure 6: top features.

Two side-by-side panels:
  A. Logistic regression — top 15 by |coefficient|
  B. XGBoost           — top 15 by importance
"""

from __future__ import annotations
import textwrap
from importlib.machinery import SourceFileLoader
from pathlib import Path as _P

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("17_feature_importance_figure")

lr  = pd.read_csv(setup.RESULTS / "table7_features_logreg.csv")
xgb = pd.read_csv(setup.RESULTS / "table7_features_xgboost.csv")


def short_label(s, width=48):
    s = s.replace("(EC ", "(EC ").strip()
    return "\n".join(textwrap.wrap(s, width=width))[:140]


def plot_panel(ax, df, value_col, title, color, top_n=15):
    top = df.head(top_n).iloc[::-1]
    y = np.arange(len(top))
    ax.barh(y, top[value_col],
            color=color, edgecolor="#222", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([short_label(s) for s in top["feature"]], fontsize=7)
    ax.set_title(title, fontsize=9.5)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8)


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

plot_panel(
    axes[0],
    lr.sort_values("abs_coefficient", ascending=False),
    "abs_coefficient",
    "Logistic regression — top 15 by |coefficient|",
    "#3b6fb5",
)
axes[0].set_xlabel("|Coefficient|")

plot_panel(
    axes[1],
    xgb,
    "importance",
    "XGBoost — top 15 by importance",
    "#c1573b",
)
axes[1].set_xlabel("Importance")

fig.suptitle("Feature importance identifies established carbapenem-resistance "
             "determinants", fontsize=10.5, fontweight="bold", y=1.00)
fig.tight_layout()
fig.savefig(setup.FIGURES / "figure6_feature_importance.pdf", bbox_inches="tight")
fig.savefig(setup.FIGURES / "figure6_feature_importance.png",
            bbox_inches="tight", dpi=300)
plt.close(fig)
log.info("Saved: figures/figure6_feature_importance.pdf / .png")
"""
15_validation_figure.py — Figure 3: validation performance across schemes.

Three separate panels (one per validation question):
  A. Random 5-fold     — IID-like performance
  B. MLST GroupKFold   — lineage-independent performance
  C. Temporal ≥2016    — temporal generalisation

Each panel shows LR and XGBoost side by side.
"""

from __future__ import annotations
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path as _P

import numpy as np
import matplotlib.pyplot as plt

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("15_validation_figure")

results = json.loads((setup.RESULTS / "08_model_results.json").read_text(encoding="utf-8"))

def get(model, scheme):
    return results.get(f"{model}__{scheme}", {})

panels = [
    ("Random 5-fold CV",         "IID-like performance",             "random"),
    ("MLST GroupKFold",          "Lineage-independent performance",  "mlst"),
    ("Temporal holdout ≥2016",   "Temporal generalisation",          "temporal"),
]

fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
colors = {"logistic_regression": "#3b6fb5", "xgboost": "#c1573b"}
labels = {"logistic_regression": "Logistic regression", "xgboost": "XGBoost"}

for ax, (title, subtitle, scheme) in zip(axes, panels):
    aurocs, sds, names = [], [], []
    for model in ("logistic_regression", "xgboost"):
        r = get(model, scheme)
        if not r:
            continue
        if scheme == "temporal":
            aurocs.append(r.get("auroc", np.nan))
            sds.append(0)
        else:
            aurocs.append(r.get("auroc_mean", np.nan))
            sds.append(r.get("auroc_sd", 0))
        names.append(model)

    x = np.arange(len(names))
    bars = ax.bar(x, aurocs, yerr=sds, capsize=4,
                  color=[colors[n] for n in names],
                  edgecolor="#333", linewidth=0.6, width=0.6)

    for bar, a in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, a + 0.02,
                f"{a:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([labels[n] for n in names], fontsize=8.5)
    ax.set_title(f"{title}\n{subtitle}", fontsize=9)
    ax.set_ylim(0.5, 1.02)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.6)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

axes[0].set_ylabel("AUROC")

fig.suptitle("Model performance across three validation questions",
             fontsize=10.5, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(setup.FIGURES / "figure3_validation_schemes.pdf", bbox_inches="tight")
fig.savefig(setup.FIGURES / "figure3_validation_schemes.png", bbox_inches="tight", dpi=300)
plt.close(fig)
log.info("Saved: figures/figure3_validation_schemes.pdf / .png")
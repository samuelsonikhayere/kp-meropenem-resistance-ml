"""
13_model_diagnostics.py — Calibration, decision curve, confusion matrices.

For the primary model (logistic regression), evaluated on:
  - Random 5-fold CV
  - Africa holdout

Produces figures for the supplementary and the primary manuscript.
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from importlib.machinery import SourceFileLoader
from pathlib import Path as _P

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             confusion_matrix, roc_auc_score,
                             average_precision_score)

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("13_model_diagnostics")

RANDOM_SEED = setup.RANDOM_SEED
LR_PARAMS   = dict(max_iter=5000, C=1.0, solver="lbfgs", random_state=RANDOM_SEED)

# Publication-style plot settings
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       9,
    "axes.labelsize":  9,
    "axes.titlesize":  10,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "figure.dpi":       150,
})

# ─────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────
table = pd.read_csv(setup.PROCESSED / "06_analysis_table.csv",
                    low_memory=False, dtype={"genome_id": str})

META = {"genome_id", "label", "region", "region_sensitivity",
        "isolation_country", "sequencing_center", "mlst",
        "genome_quality", "collection_year", "method_tier", "has_annotation"}
features = [c for c in table.columns if c not in META]
X = table[features].values.astype("float32")
y = table["label"].values.astype(int)
regions = table["region"].values


# ─────────────────────────────────────────────────────────────────────────
# 1. Random CV — pooled predictions
# ─────────────────────────────────────────────────────────────────────────
log.info("Random 5-fold CV — pooling test predictions...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
p_random = np.zeros(len(y))
for tr, te in skf.split(X, y):
    m = LogisticRegression(**LR_PARAMS)
    m.fit(X[tr], y[tr])
    p_random[te] = m.predict_proba(X[te])[:, 1]

# ─────────────────────────────────────────────────────────────────────────
# 2. Africa holdout — train on non-Africa, test on Africa
# ─────────────────────────────────────────────────────────────────────────
log.info("Africa holdout...")
africa_mask = regions == "Africa"
X_train_afr, y_train_afr = X[~africa_mask], y[~africa_mask]
X_test_afr,  y_test_afr  = X[africa_mask],  y[africa_mask]

m_afr = LogisticRegression(**LR_PARAMS)
m_afr.fit(X_train_afr, y_train_afr)
p_africa = m_afr.predict_proba(X_test_afr)[:, 1]

log.info(f"  Africa holdout AUROC: {roc_auc_score(y_test_afr, p_africa):.3f}")


# ─────────────────────────────────────────────────────────────────────────
# Figure 1 — ROC curves (random CV pooled + Africa holdout)
# ─────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 4.5))

fpr, tpr, _ = roc_curve(y, p_random)
auc_r = roc_auc_score(y, p_random)
ax.plot(fpr, tpr, color="tab:blue", lw=1.8,
        label=f"Random 5-fold CV (pooled, AUROC={auc_r:.3f})")

if len(np.unique(y_test_afr)) > 1:
    fpr_a, tpr_a, _ = roc_curve(y_test_afr, p_africa)
    auc_a = roc_auc_score(y_test_afr, p_africa)
    ax.plot(fpr_a, tpr_a, color="tab:red", lw=1.8,
            label=f"Africa holdout (AUROC={auc_a:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
ax.set_xlabel("False positive rate (1 − specificity)")
ax.set_ylabel("True positive rate (sensitivity)")
ax.set_title("ROC — meropenem resistance")
ax.legend(loc="lower right", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(setup.FIGURES / "figure_roc.pdf")
fig.savefig(setup.FIGURES / "figure_roc.png")
plt.close(fig)
log.info("Saved: figures/figure_roc.pdf / .png")


# ─────────────────────────────────────────────────────────────────────────
# Figure 2 — Precision-recall curves
# ─────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 4.5))

prec, rec, _ = precision_recall_curve(y, p_random)
ap_r = average_precision_score(y, p_random)
ax.plot(rec, prec, color="tab:blue", lw=1.8,
        label=f"Random 5-fold CV (pooled, AUPRC={ap_r:.3f})")

if len(np.unique(y_test_afr)) > 1:
    prec_a, rec_a, _ = precision_recall_curve(y_test_afr, p_africa)
    ap_a = average_precision_score(y_test_afr, p_africa)
    ax.plot(rec_a, prec_a, color="tab:red", lw=1.8,
            label=f"Africa holdout (AUPRC={ap_a:.3f})")

ax.axhline(y.mean(), color="grey", linestyle="--", lw=0.8,
           label=f"Baseline prevalence ({y.mean():.2f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-recall — meropenem resistance")
ax.legend(loc="best", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(setup.FIGURES / "figure_pr.pdf")
fig.savefig(setup.FIGURES / "figure_pr.png")
plt.close(fig)
log.info("Saved: figures/figure_pr.pdf / .png")


# ─────────────────────────────────────────────────────────────────────────
# Figure 3 — Calibration plot (random CV pooled + Africa holdout)
# ─────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 4.5))

frac_pos, mean_pred = calibration_curve(y, p_random, n_bins=10, strategy="quantile")
ax.plot(mean_pred, frac_pos, marker="o", color="tab:blue", lw=1.5,
        markersize=4, label="Random 5-fold CV (pooled)")

if len(np.unique(y_test_afr)) > 1:
    try:
        frac_a, mean_a = calibration_curve(y_test_afr, p_africa,
                                           n_bins=5, strategy="quantile")
        ax.plot(mean_a, frac_a, marker="s", color="tab:red", lw=1.5,
                markersize=4, label="Africa holdout")
    except ValueError:
        pass

ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed fraction resistant")
ax.set_title("Calibration — meropenem resistance")
ax.legend(loc="best", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(setup.FIGURES / "figure_calibration.pdf")
fig.savefig(setup.FIGURES / "figure_calibration.png")
plt.close(fig)
log.info("Saved: figures/figure_calibration.pdf / .png")


# ─────────────────────────────────────────────────────────────────────────
# Figure 4 — Decision curve analysis (random CV)
# ─────────────────────────────────────────────────────────────────────────
def net_benefit(y_true, y_prob, thresholds):
    N = len(y_true)
    nb = []
    for t in thresholds:
        pred = y_prob >= t
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        nb.append((tp / N) - (fp / N) * (t / (1 - t)))
    return np.array(nb)


thresholds = np.linspace(0.01, 0.99, 99)
nb_model = net_benefit(y, p_random, thresholds)
nb_all   = net_benefit(y, np.ones_like(p_random), thresholds)
prevalence = y.mean()

fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.plot(thresholds, nb_model, color="tab:blue", lw=1.8, label="Model")
ax.plot(thresholds, nb_all, color="grey", lw=1.0, linestyle="--",
        label="Treat all")
ax.axhline(0, color="black", lw=0.8, linestyle=":",
           label="Treat none")
ax.axvline(prevalence, color="grey", lw=0.6, linestyle=":",
           label=f"Prevalence = {prevalence:.2f}")
ax.set_xlabel("Threshold probability")
ax.set_ylabel("Net benefit")
ax.set_title("Decision curve — meropenem resistance")
ax.legend(loc="best", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(setup.FIGURES / "figure_dca.pdf")
fig.savefig(setup.FIGURES / "figure_dca.png")
plt.close(fig)
log.info("Saved: figures/figure_dca.pdf / .png")


# ─────────────────────────────────────────────────────────────────────────
# Figure 5 — Confusion matrix (threshold 0.5) for random CV pooled
# ─────────────────────────────────────────────────────────────────────────
cm = confusion_matrix(y, (p_random >= 0.5).astype(int))
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(cm, cmap="Blues", aspect="equal")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=11)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred S", "Pred R"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["True S", "True R"])
ax.set_title("Confusion matrix (threshold 0.5, random CV pooled)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(setup.FIGURES / "figure_confusion.pdf")
fig.savefig(setup.FIGURES / "figure_confusion.png")
plt.close(fig)
log.info("Saved: figures/figure_confusion.pdf / .png")


# ─────────────────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────────────────
setup.write_result("13_model_diagnostics", {
    "analysis_name":     "model_diagnostics",
    "dataset_version":   "api_v1",
    "n":                 int(len(y)),
    "auroc_random_pooled": float(auc_r),
    "auroc_africa_holdout": float(roc_auc_score(y_test_afr, p_africa))
                              if len(np.unique(y_test_afr)) > 1 else None,
    "confusion_matrix_random": cm.tolist(),
    "timestamp":         datetime.now(timezone.utc).isoformat(),
    "software_version":  setup.software_versions(),
})
log.info("Done.")
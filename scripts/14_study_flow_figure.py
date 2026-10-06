"""
14_study_flow_figure.py — Figure 1: cohort selection and analysis flow diagram.

Reads numbers from results/table2_filter_cascade.csv and processed/06_analysis_table.csv.
Renders a box-and-arrow study flow diagram (not CONSORT).
"""

from __future__ import annotations
from importlib.machinery import SourceFileLoader
from pathlib import Path as _P

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("14_study_flow_figure")

# ── Load numbers ─────────────────────────────────────────────────────────
cascade = pd.read_csv(setup.RESULTS / "table2_filter_cascade.csv")

def val(stage_substr):
    row = cascade[cascade["stage"].str.contains(stage_substr, na=False)]
    return int(row.iloc[0]["records"]) if len(row) and pd.notna(row.iloc[0]["records"]) else None

def uniq(stage_substr):
    row = cascade[cascade["stage"].str.contains(stage_substr, na=False)]
    return int(row.iloc[0]["unique_genomes"]) if len(row) and pd.notna(row.iloc[0]["unique_genomes"]) else None

primary = pd.read_csv(setup.PROCESSED / "06_analysis_table.csv", low_memory=False)
full    = pd.read_csv(setup.PROCESSED / "06_analysis_table_full.csv", low_memory=False)

n_amr_records     = val("Raw meropenem AMR records")
n_after_merge     = val("After inner merge")
n_after_phenotype = val("After dropping missing phenotype")
n_after_binary    = val("After keeping only R/S")
n_after_verified  = val("After verified-method filter")
n_after_dedup     = val("After evidence-hierarchy")
n_annotated       = len(primary)
n_full_cohort     = len(full)
n_no_annotation   = n_full_cohort - n_annotated
n_features        = len(setup.feature_columns(primary))

log.info(f"Flow numbers: AMR={n_amr_records}, merged={n_after_merge}, "
         f"phenotype={n_after_phenotype}, binary={n_after_binary}, "
         f"verified={n_after_verified}, dedup={n_after_dedup}, "
         f"annotated={n_annotated}, full={n_full_cohort}, feats={n_features}")

# Assert expected numbers
assert n_full_cohort == 3930, f"Expected 3930 verified cohort, got {n_full_cohort}"
assert n_annotated == 3694,   f"Expected 3694 annotated, got {n_annotated}"

# ── Draw figure ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "figure.dpi": 300,
})

fig, ax = plt.subplots(figsize=(7.5, 9))
ax.set_xlim(0, 10); ax.set_ylim(0, 15); ax.axis("off")

steps = [
    ("Raw BV-BRC meropenem AMR records",
     f"n = {n_amr_records:,}", "#eef4fb"),
    ("Genome metadata merged on genome_id",
     f"n = {n_after_merge:,} records; "
     f"{uniq('After inner merge') or ''} unique genome IDs", "#eef4fb"),
    ("Missing phenotype excluded",
     f"n = {n_after_phenotype:,} records", "#f6f9fd"),
    ("Intermediate / SDD phenotypes excluded",
     f"n = {n_after_binary:,} records", "#f6f9fd"),
    ("Verified laboratory typing method required",
     f"n = {n_after_verified:,} records", "#f6f9fd"),
    ("Evidence-hierarchy deduplication (one row per genome)",
     f"n = {n_after_dedup:,} verified isolates", "#eaf5ea"),
    ("Verified cohort (analysis denominator)",
     f"n = {n_full_cohort:,} isolates\n"
     f"1,579 resistant / 2,351 susceptible", "#eaf5ea"),
    ("Annotated feature matrix (primary modelling cohort)",
     f"n = {n_annotated:,} isolates × {n_features} AMR products\n"
     f"{n_no_annotation} verified isolates excluded (no AMR annotation)", "#fdf2e6"),
]

y_positions = [14.2, 12.6, 11.3, 10.2, 9.1, 8.0, 6.5, 4.6]
box_w, box_h = 7.0, 1.1
x_center = 5

for (title, sub, col), y in zip(steps, y_positions):
    box = FancyBboxPatch(
        (x_center - box_w/2, y - box_h/2), box_w, box_h,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        linewidth=0.8, edgecolor="#333", facecolor=col)
    ax.add_patch(box)
    ax.text(x_center, y + 0.20, title, ha="center", va="center",
            fontsize=9.5, fontweight="bold")
    ax.text(x_center, y - 0.25, sub, ha="center", va="center",
            fontsize=8.5, color="#333")

for y_top, y_bot in zip(y_positions[:-1], y_positions[1:]):
    arrow = FancyArrowPatch(
        (x_center, y_top - box_h/2 - 0.02),
        (x_center, y_bot + box_h/2 + 0.02),
        arrowstyle="-|>", mutation_scale=12,
        linewidth=0.9, color="#333")
    ax.add_patch(arrow)

ax.text(x_center, 3.3, "Primary analyses (Batch 3)", ha="center",
        fontsize=9.5, fontweight="bold")

analysis_items = [
    "Random 5-fold cross-validation",
    "MLST-grouped 5-fold cross-validation",
    "Leave-one-region-out validation (7 regions)",
    "Temporal holdout: train <2016, test ≥2016",
    "2,000-iteration bootstrap CIs per region",
    "Baselines: majority class, annotation availability,\n"
    "            11 batch-marker products",
]
for i, txt in enumerate(analysis_items):
    ax.text(x_center, 2.8 - i * 0.35, f"•  {txt}",
            ha="left", va="top", fontsize=8.5, color="#222",
            family="DejaVu Sans")

fig.suptitle("Study cohort selection and analysis flow",
             fontsize=11, fontweight="bold", y=0.98)
fig.tight_layout()
fig.savefig(setup.FIGURES / "figure1_study_flow.pdf", bbox_inches="tight")
fig.savefig(setup.FIGURES / "figure1_study_flow.png", bbox_inches="tight", dpi=300)
plt.close(fig)
log.info("Saved: figures/figure1_study_flow.pdf / .png")
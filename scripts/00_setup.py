"""
00_setup.py — Project environment validation and path configuration.

Every other script imports from this module to guarantee consistent
paths, seeds, and logging.
"""

from __future__ import annotations
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

# ─────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

RAW        = ROOT / "raw_data"
PROCESSED  = ROOT / "processed"
RESULTS    = ROOT / "results"
FIGURES    = ROOT / "figures"
LOGS       = ROOT / "logs"
PROVENANCE = ROOT / "provenance"
ENV        = ROOT / "env"

for d in (PROCESSED, RESULTS, FIGURES, LOGS, PROVENANCE, ENV):
    d.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────
# Domain constants — single source of truth across all scripts
# ─────────────────────────────────────────────────────────────────────────

# Verified laboratory typing methods, ordered by evidence tier
VERIFIED_METHOD_TIERS: dict[str, int] = {
    # Tier 1 — quantitative / standardised
    "MIC":              1,
    "Broth dilution":   1,
    "Agar dilution":    1,
    "Etest":            1,
    # Tier 2 — automated AST platforms
    "VITEK":            2,
    "VITEK 2":          2,
    "Phoenix":          2,
    "MicroScan":        2,
    "Microscan":        2,
    "Sensititre":       2,
    # Tier 3 — qualitative
    "Disk diffusion":   3,
}

# Phenotypes to exclude before binarisation
EXCLUDED_PHENOTYPES = {"Intermediate", "Susceptible-dose dependent"}

# Phenotypes to keep (binary)
BINARY_PHENOTYPES = {"Resistant", "Susceptible"}

# Continent / sub-region mapping (primary analysis)
# Caribbean is a SEPARATE region per FINAL-1 decision
REGION_MAP: dict[str, str] = {
    # ── Africa ──────────────────────────────────────────────────────────
    **{c: "Africa" for c in [
        "Algeria","Angola","Benin","Botswana","Burkina Faso","Burundi",
        "Cabo Verde","Cameroon","Central African Republic","Chad","Comoros",
        "Democratic Republic of the Congo","Congo","Republic of the Congo",
        "Djibouti","Egypt","Equatorial Guinea","Eritrea","Eswatini",
        "Ethiopia","Gabon","Gambia","Ghana","Guinea","Guinea-Bissau",
        "Cote d'Ivoire","Ivory Coast","Kenya","Lesotho","Liberia","Libya",
        "Madagascar","Malawi","Mali","Mauritania","Mauritius","Morocco",
        "Mozambique","Namibia","Niger","Nigeria","Rwanda",
        "Sao Tome and Principe","Senegal","Sierra Leone","Somalia",
        "South Africa","South Sudan","Sudan","Tanzania","Togo","Tunisia",
        "Uganda","Zambia","Zimbabwe",
    ]},

    # ── Asia (incl. Middle East) ────────────────────────────────────────
    **{c: "Asia" for c in [
        "China","India","Japan","South Korea","Korea","Taiwan",
        "Thailand","Vietnam","Singapore","Malaysia","Indonesia",
        "Philippines","Pakistan","Bangladesh","Sri Lanka","Nepal",
        "Iran","Iraq","Israel","Saudi Arabia","United Arab Emirates",
        "Qatar","Kuwait","Oman","Jordan","Lebanon","Turkey",
        "Russia","Kazakhstan","Uzbekistan","Afghanistan",
    ]},

    # ── Europe ──────────────────────────────────────────────────────────
    **{c: "Europe" for c in [
        "United Kingdom","Ireland","France","Germany","Italy","Spain",
        "Portugal","Netherlands","Belgium","Switzerland","Austria",
        "Sweden","Norway","Denmark","Finland","Iceland","Poland",
        "Czech Republic","Czechia","Slovakia","Hungary","Romania",
        "Bulgaria","Greece","Croatia","Serbia","Slovenia","Estonia",
        "Latvia","Lithuania","Ukraine","Belarus","Moldova",
    ]},

    # ── North America (continental only) ────────────────────────────────
    **{c: "North America" for c in [
        "USA","United States","Canada","Mexico","Honduras",
    ]},

    # ── Caribbean (separate region) ─────────────────────────────────────
    **{c: "Caribbean" for c in [
        "Barbados","Trinidad","Trinidad and Tobago","Dominica",
        "Saint Kitts","Haiti","Belize","Saint Lucia","Saint Vincent",
        "Cayman Islands","Grenada","Bermuda","Antigua",
        "Dominican Republic","Jamaica","Bahamas","Cuba",
        "Puerto Rico","Guadeloupe","Martinique",
    ]},

    # ── South America ───────────────────────────────────────────────────
    **{c: "South America" for c in [
        "Brazil","Argentina","Chile","Colombia","Peru","Venezuela",
        "Ecuador","Bolivia","Uruguay","Paraguay",
    ]},

    # ── Oceania ─────────────────────────────────────────────────────────
    **{c: "Oceania" for c in [
        "Australia","New Zealand","Guam",
    ]},
}


# ─────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────
# Analysis metadata columns — single source of truth
# ─────────────────────────────────────────────────────────────────────────
# Columns in the processed analysis tables that are NOT features.
# Any column outside this list is treated as a feature by downstream scripts.
ANALYSIS_METADATA_COLUMNS = [
    "genome_id",
    "label",
    "region",
    "region_sensitivity",
    "isolation_country",
    "sequencing_center",
    "mlst",
    "genome_quality",
    "collection_year",
    "method_tier",
    "has_annotation",
]


def feature_columns(df) -> list:
    """Return the list of feature (non-metadata) column names in an analysis table."""
    return [c for c in df.columns if c not in ANALYSIS_METADATA_COLUMNS]

def get_logger(name: str) -> logging.Logger:
    """Create a logger that writes to both console and logs/<name>.log."""
    logger = logging.getLogger(name)
    if logger.handlers:              # already configured
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(LOGS / f"{name}.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# ─────────────────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────────────────
def software_versions() -> dict:
    """Capture exact versions of every relevant package for provenance."""
    return {
        "python":     sys.version.split()[0],
        "platform":   platform.platform(),
        "pandas":     pd.__version__,
        "numpy":      np.__version__,
        "sklearn":    sklearn.__version__,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


def write_result(name: str, payload: dict) -> Path:
    """Append a machine-readable result object to results/all_results.json."""
    import json
    results_file = RESULTS / "all_results.json"
    existing = []
    if results_file.exists():
        try:
            existing = json.loads(results_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
    existing.append(payload)
    results_file.write_text(
        json.dumps(existing, indent=2, default=str),
        encoding="utf-8",
    )
    return results_file


if __name__ == "__main__":
    log = get_logger("00_setup")
    log.info("Pipeline setup module loaded.")
    log.info(f"Project root: {ROOT}")
    log.info(f"Software versions: {software_versions()}")
    log.info(f"REGION_MAP defines {len(set(REGION_MAP.values()))} regions.")
    log.info(f"Verified typing methods: {len(VERIFIED_METHOD_TIERS)}")
    log.info(f"RANDOM_SEED = {RANDOM_SEED}")
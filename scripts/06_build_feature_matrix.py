"""
06_build_feature_matrix.py — Build genome × AMR determinant binary matrix.

Inputs  : processed/sp_gene_raw.csv      (from 05 — long-format AMR annotations)
          processed/04_with_regions.csv  (cohort definition)
Outputs : processed/06_feature_matrix.csv        (primary; annotated genomes only)
          processed/06_feature_matrix_full.csv   (sensitivity; all genomes, zero-filled)
          processed/06_analysis_table.csv        (features + labels + region, primary)
          processed/06_analysis_table_full.csv   (features + labels + region, sensitivity)
          results/tableS2_feature_matrix_summary.csv
          provenance/06_build_feature_matrix.json
          logs/06_build_feature_matrix.log

Decision rationale (locked):
  - BV-BRC's sp_gene endpoint populates "product" (100%) but not "gene" (0%).
  - The sp_gene_ref table returns gene names only for some accession-style
    source_ids, and none for the numeric source_ids used by CARD/PATRIC.
  - Therefore the "product" string is used directly as the feature identifier.
  - Products are gene-family-level (e.g. "SHV family"), which is the standard
    granularity for AMR genotype-to-phenotype prediction.
  - Binary collapse via max aggregation (present = 1, absent = 0).
  - Primary analysis: annotated genomes only.
  - Sensitivity analysis: full cohort with explicit has_annotation flag
    (do NOT treat missing annotation as confirmed absence).
"""

from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from importlib.machinery import SourceFileLoader
from pathlib import Path as _P

import pandas as pd

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("06_build_feature_matrix")


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_write_csv(df: pd.DataFrame, path) -> None:
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        log.error(f"Could not write: {path}")
        log.error("The file is open in another program (Excel?). Close it and re-run.")
        raise


# ─────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────
sp_gene = pd.read_csv(setup.PROCESSED / "sp_gene_raw.csv",
                      low_memory=False, dtype={"genome_id": str})
sp_gene["genome_id"] = sp_gene["genome_id"].astype(str)

cohort = pd.read_csv(setup.PROCESSED / "04_with_regions.csv",
                     low_memory=False, dtype={"genome_id": str})
cohort["genome_id"] = cohort["genome_id"].astype(str)

log.info(f"sp_gene rows:      {len(sp_gene)}")
log.info(f"cohort rows:       {len(cohort)}")


# ─────────────────────────────────────────────────────────────────────────
# Build feature identifier from product
# ─────────────────────────────────────────────────────────────────────────
sp_gene["feature"] = sp_gene["product"].astype(str).str.strip()

before = len(sp_gene)
sp_gene = sp_gene[
    sp_gene["feature"].notna()
    & (sp_gene["feature"] != "")
    & (sp_gene["feature"].str.lower() != "nan")
]
log.info(f"Dropped {before - len(sp_gene)} records with missing/invalid product")
log.info(f"Records with usable features: {len(sp_gene)}")
log.info(f"Unique features:              {sp_gene['feature'].nunique()}")


# ─────────────────────────────────────────────────────────────────────────
# Build binary genome × feature matrix
# ─────────────────────────────────────────────────────────────────────────
matrix = (sp_gene
          .assign(value=1)
          .pivot_table(index="genome_id",
                       columns="feature",
                       values="value",
                       aggfunc="max",
                       fill_value=0)
          .astype("int8"))

log.info(f"Matrix shape (annotated genomes only): {matrix.shape}")


# ─────────────────────────────────────────────────────────────────────────
# Primary analysis table — annotated genomes only
# ─────────────────────────────────────────────────────────────────────────
annotated_ids = set(matrix.index)

# Metadata columns to carry through
meta_cols = ["genome_id", "label", "region", "region_sensitivity",
             "isolation_country", "sequencing_center", "mlst",
             "genome_quality", "collection_year", "method_tier"]

cohort_annotated = (cohort[cohort["genome_id"].isin(annotated_ids)]
                    .set_index("genome_id")
                    .reindex(matrix.index))
cohort_annotated = cohort_annotated.reset_index()

keep_meta = [c for c in meta_cols if c in cohort_annotated.columns]

primary_table = pd.concat(
    [
        cohort_annotated[keep_meta].reset_index(drop=True),
        matrix.reset_index(drop=True),
    ],
    axis=1,
)
log.info(f"Primary analysis table: {primary_table.shape}")


# ─────────────────────────────────────────────────────────────────────────
# Sensitivity analysis table — full cohort, zero-fill absent
# ─────────────────────────────────────────────────────────────────────────
full_matrix = matrix.reindex(cohort["genome_id"].values, fill_value=0)
full_matrix.index.name = "genome_id"

cohort_full = (cohort.set_index("genome_id")
               .reindex(full_matrix.index)
               .reset_index())
cohort_full["has_annotation"] = (cohort_full["genome_id"]
                                 .isin(annotated_ids).astype(int))

keep_meta_full = ["genome_id", "label", "region", "region_sensitivity",
                  "isolation_country", "sequencing_center", "mlst",
                  "genome_quality", "collection_year", "method_tier",
                  "has_annotation"]
keep_meta_full = [c for c in keep_meta_full if c in cohort_full.columns]

full_table = pd.concat(
    [
        cohort_full[keep_meta_full].reset_index(drop=True),
        full_matrix.reset_index(drop=True),
    ],
    axis=1,
)
log.info(f"Sensitivity analysis table: {full_table.shape}")


# ─────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────
paths = {
    "feature_matrix":       setup.PROCESSED / "06_feature_matrix.csv",
    "feature_matrix_full":  setup.PROCESSED / "06_feature_matrix_full.csv",
    "analysis_table":       setup.PROCESSED / "06_analysis_table.csv",
    "analysis_table_full":  setup.PROCESSED / "06_analysis_table_full.csv",
}
safe_write_csv(matrix, paths["feature_matrix"])
safe_write_csv(full_matrix, paths["feature_matrix_full"])
safe_write_csv(primary_table, paths["analysis_table"])
safe_write_csv(full_table, paths["analysis_table_full"])

for k, p in paths.items():
    log.info(f"Saved: {p.relative_to(setup.ROOT)}  ({p.stat().st_size / 1e6:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────
# Table S2 — Feature matrix summary
# ─────────────────────────────────────────────────────────────────────────
gene_prevalence = matrix.sum(axis=0).sort_values(ascending=False)
n_genomes = matrix.shape[0]

summary_rows = [
    {"metric": "Verified cohort size",                   "value": int(len(cohort))},
    {"metric": "Annotated genomes (primary)",            "value": int(n_genomes)},
    {"metric": "Genomes without annotation (sensitivity)", "value": int(len(cohort) - n_genomes)},
    {"metric": "Unique AMR products (features)",         "value": int(matrix.shape[1])},
    {"metric": "Median features per genome",             "value": float(matrix.sum(axis=1).median())},
    {"metric": "Mean features per genome",               "value": float(round(matrix.sum(axis=1).mean(), 2))},
    {"metric": "Features in >90% of genomes",            "value": int((matrix.mean(axis=0) > 0.9).sum())},
    {"metric": "Features in <10% of genomes",            "value": int((matrix.mean(axis=0) < 0.1).sum())},
    {"metric": "Most prevalent feature",                 "value": str(gene_prevalence.index[0])[:60]},
    {"metric": "Most prevalent feature prevalence",      "value": f"{gene_prevalence.iloc[0] / n_genomes:.1%}"},
    {"metric": "Carbapenemase products detected",
     "value": int(sum(1 for f in matrix.columns
                      if any(k in f.lower() for k in
                             ["kpc", "ndm", "oxa-48", "oxa-181", "oxa-232",
                              "vim", "imp", "carbapenem"])))},
]
summary_df = pd.DataFrame(summary_rows)
safe_write_csv(summary_df, setup.RESULTS / "tableS2_feature_matrix_summary.csv")
log.info("Saved: results/tableS2_feature_matrix_summary.csv")
log.info("\n" + summary_df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────────────────
prov = {
    "script":    "06_build_feature_matrix.py",
    "run_at":    datetime.now(timezone.utc).isoformat(),
    "software":  setup.software_versions(),
    "decisions": {
        "B1_property_filter":       "Antibiotic Resistance (via sp_gene keyword)",
        "B2_unresolved_names":      "dropped",
        "B3_binarisation":          "max aggregation → 0/1",
        "feature_identifier":       "sp_gene.product (sp_gene_ref.gene was empty)",
        "primary_scope":            "annotated genomes only",
        "sensitivity_scope":        "full cohort with has_annotation flag",
    },
    "counts": {
        "cohort_size":        int(len(cohort)),
        "sp_gene_raw_rows":   int(len(sp_gene)),
        "annotated_genomes":  int(matrix.shape[0]),
        "features":           int(matrix.shape[1]),
        "genomes_zero_annot": int(len(cohort) - matrix.shape[0]),
    },
    "outputs": {k: {"path": str(v.relative_to(setup.ROOT)),
                    "sha256": sha256_file(v)}
                for k, v in paths.items()},
}
(setup.PROVENANCE / "06_build_feature_matrix.json").write_text(
    json.dumps(prov, indent=2), encoding="utf-8")

setup.write_result("06_build_feature_matrix", {
    "analysis_name":     "feature_matrix_construction",
    "dataset_version":   "api_v1",
    "n_cohort":          int(len(cohort)),
    "n_annotated":       int(matrix.shape[0]),
    "n_features":        int(matrix.shape[1]),
    "timestamp":         datetime.now(timezone.utc).isoformat(),
    "software_version":  setup.software_versions(),
})
log.info("Done.")
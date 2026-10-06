"""
05_download_features.py — Download AMR gene annotations from BV-BRC API.

Downloads sp_gene (filtered to Antibiotic Resistance) and resolves
gene names via sp_gene_ref, for every genome_id in the verified cohort.

Input  : processed/04_with_regions.csv   (defines the genome_id set)
Output : processed/sp_gene_raw.csv       (raw long-format AMR annotations)
         processed/sp_gene_ref.csv       (resolved gene names)
         provenance/05_download_features.json
         logs/05_download_features.log

Design:
  - Batched queries (100 genome_ids per request) to avoid rate limits
  - Checkpoint after every batch — resume on interruption
  - SHA256 of both output files for the manuscript
  - Retry logic with exponential backoff on 5xx/429

Runtime estimate: 15-30 minutes for 3,930 genomes.
"""

from __future__ import annotations
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from importlib.machinery import SourceFileLoader
from pathlib import Path as _P

import pandas as pd
import requests
from tqdm import tqdm

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("05_download_features")

# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────
BASE             = "https://www.bv-brc.org/api"
HEADERS          = {"http_accept": "text/csv"}
TIMEOUT          = 180
BATCH_SIZE       = 100       # genomes per sp_gene request
REF_BATCH_SIZE   = 100       # source_ids per sp_gene_ref request
SLEEP_BETWEEN    = 1.0       # seconds between requests
MAX_RETRIES      = 3

SP_GENE_CHECKPOINT = setup.PROCESSED / "_sp_gene_checkpoint.csv"
SP_REF_CHECKPOINT  = setup.PROCESSED / "_sp_gene_ref_checkpoint.csv"


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def query_with_retry(url: str, retries: int = MAX_RETRIES) -> pd.DataFrame | None:
    """GET a URL, retry on 5xx/429, return DataFrame or None."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=HEADERS, timeout=TIMEOUT)
            if r.status_code == 200:
                if not r.text.strip():
                    return pd.DataFrame()
                return pd.read_csv(io.StringIO(r.text), low_memory=False)
            elif r.status_code in (429, 500, 502, 503, 504):
                wait = 5 * (attempt + 1)
                log.warning(f"  HTTP {r.status_code}, retrying in {wait}s "
                            f"(attempt {attempt + 1}/{retries})")
                time.sleep(wait)
            else:
                log.error(f"  HTTP {r.status_code}: {r.text[:200]}")
                return None
        except requests.RequestException as e:
            log.warning(f"  Request error: {e} (attempt {attempt + 1}/{retries})")
            time.sleep(2 + attempt)
    return None


def query_sp_gene(genome_ids: list[str]) -> pd.DataFrame | None:
    """Query sp_gene for a batch, filtered to AMR."""
    id_list = ",".join(str(g) for g in genome_ids)
    rql = (
        f"and(in(genome_id,({id_list})),keyword(Antibiotic Resistance))"
        f"&limit(50000)"
        f"&select(genome_id,gene,product,source,source_id,classification)"
    )
    return query_with_retry(f"{BASE}/sp_gene/?{rql}")


def query_sp_gene_ref(source_ids: list[str]) -> pd.DataFrame | None:
    """Resolve gene names via sp_gene_ref."""
    id_list = ",".join(str(s) for s in source_ids)
    rql = (
        f"in(source_id,({id_list}))"
        f"&limit(50000)"
        f"&select(source_id,gene,source,product)"
    )
    return query_with_retry(f"{BASE}/sp_gene_ref/?{rql}")


# ─────────────────────────────────────────────────────────────────────────
# Load cohort
# ─────────────────────────────────────────────────────────────────────────
cohort = pd.read_csv(setup.PROCESSED / "04_with_regions.csv",
                     low_memory=False, dtype={"genome_id": str})
cohort["genome_id"] = cohort["genome_id"].astype(str)
all_ids = cohort["genome_id"].dropna().unique().tolist()
log.info(f"Verified cohort: {len(all_ids)} genomes")


# ─────────────────────────────────────────────────────────────────────────
# Phase 1 — sp_gene download
# ─────────────────────────────────────────────────────────────────────────
if SP_GENE_CHECKPOINT.exists():
    existing = pd.read_csv(SP_GENE_CHECKPOINT,
                           low_memory=False, dtype={"genome_id": str})
    existing["genome_id"] = existing["genome_id"].astype(str)
    done_ids = set(existing["genome_id"].unique())
    log.info(f"Resuming sp_gene: {len(done_ids)} genomes already present")
    sp_gene_frames = [existing]
else:
    done_ids = set()
    sp_gene_frames = []

remaining = [g for g in all_ids if g not in done_ids]
log.info(f"Genomes remaining for sp_gene: {len(remaining)}")

batches = [remaining[i:i + BATCH_SIZE]
           for i in range(0, len(remaining), BATCH_SIZE)]
log.info(f"Batches to process: {len(batches)}")

for i, batch in enumerate(tqdm(batches, desc="sp_gene batches")):
    df = query_sp_gene(batch)
    if df is not None and len(df):
        df["genome_id"] = df["genome_id"].astype(str)
        sp_gene_frames.append(df)

    # Checkpoint after each batch
    if sp_gene_frames:
        pd.concat(sp_gene_frames, ignore_index=True).to_csv(
            SP_GENE_CHECKPOINT, index=False)

    time.sleep(SLEEP_BETWEEN)


# Concatenate and deduplicate
sp_gene_raw = pd.concat(sp_gene_frames, ignore_index=True).drop_duplicates()
sp_gene_raw.to_csv(setup.PROCESSED / "sp_gene_raw.csv", index=False)
log.info(f"sp_gene_raw: {sp_gene_raw.shape}")
log.info(f"Unique genomes with annotations: {sp_gene_raw['genome_id'].nunique()}")
log.info(f"Genomes with NO annotation:      "
         f"{len(all_ids) - sp_gene_raw['genome_id'].nunique()}")


# ─────────────────────────────────────────────────────────────────────────
# Phase 2 — sp_gene_ref resolution
# ─────────────────────────────────────────────────────────────────────────
if SP_REF_CHECKPOINT.exists():
    ref_existing = pd.read_csv(SP_REF_CHECKPOINT, low_memory=False)
    resolved_ids = set(ref_existing["source_id"].dropna().astype(str).unique())
    log.info(f"Resuming sp_gene_ref: {len(resolved_ids)} source_ids already resolved")
    ref_frames = [ref_existing]
else:
    resolved_ids = set()
    ref_frames = []

source_ids = (sp_gene_raw["source_id"]
              .dropna()
              .astype(str)
              .unique()
              .tolist())
log.info(f"Total unique source_ids to resolve: {len(source_ids)}")

source_ids_remaining = [s for s in source_ids if s not in resolved_ids]
log.info(f"source_ids remaining: {len(source_ids_remaining)}")

ref_batches = [source_ids_remaining[i:i + REF_BATCH_SIZE]
               for i in range(0, len(source_ids_remaining), REF_BATCH_SIZE)]
log.info(f"Ref batches to process: {len(ref_batches)}")

for i, batch in enumerate(tqdm(ref_batches, desc="sp_gene_ref batches")):
    df = query_sp_gene_ref(batch)
    if df is not None and len(df):
        ref_frames.append(df)

    if ref_frames:
        pd.concat(ref_frames, ignore_index=True).to_csv(
            SP_REF_CHECKPOINT, index=False)

    time.sleep(SLEEP_BETWEEN)


# Concatenate and deduplicate
sp_gene_ref = pd.concat(ref_frames, ignore_index=True).drop_duplicates()
sp_gene_ref.columns = [c.strip().replace(" ", "_").lower()
                       for c in sp_gene_ref.columns]
sp_gene_ref.to_csv(setup.PROCESSED / "sp_gene_ref.csv", index=False)
log.info(f"sp_gene_ref: {sp_gene_ref.shape}")


# ─────────────────────────────────────────────────────────────────────────
# Cleanup checkpoints
# ─────────────────────────────────────────────────────────────────────────
for cp in (SP_GENE_CHECKPOINT, SP_REF_CHECKPOINT):
    if cp.exists():
        cp.unlink()


# ─────────────────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────────────────
prov = {
    "script":    "05_download_features.py",
    "run_at":    datetime.now(timezone.utc).isoformat(),
    "software":  setup.software_versions(),
    "api_base":  BASE,
    "batches": {
        "sp_gene_batch_size":     BATCH_SIZE,
        "sp_gene_ref_batch_size": REF_BATCH_SIZE,
        "sleep_between_seconds":  SLEEP_BETWEEN,
    },
    "counts": {
        "cohort_genomes":             len(all_ids),
        "sp_gene_rows":               int(len(sp_gene_raw)),
        "genomes_with_annotation":    int(sp_gene_raw["genome_id"].nunique()),
        "genomes_without_annotation": int(len(all_ids) - sp_gene_raw["genome_id"].nunique()),
        "unique_source_ids":          int(len(source_ids)),
        "sp_gene_ref_rows":           int(len(sp_gene_ref)),
    },
    "outputs": {
        "sp_gene_raw": {
            "path":   str((setup.PROCESSED / "sp_gene_raw.csv").relative_to(setup.ROOT)),
            "sha256": sha256_file(setup.PROCESSED / "sp_gene_raw.csv"),
        },
        "sp_gene_ref": {
            "path":   str((setup.PROCESSED / "sp_gene_ref.csv").relative_to(setup.ROOT)),
            "sha256": sha256_file(setup.PROCESSED / "sp_gene_ref.csv"),
        },
    },
}
(setup.PROVENANCE / "05_download_features.json").write_text(
    json.dumps(prov, indent=2), encoding="utf-8")
log.info("Provenance written.")

setup.write_result("05_download_features", {
    "analysis_name":     "feature_download",
    "dataset_version":   "api_v1",
    "n_cohort_genomes":  int(len(all_ids)),
    "n_with_annotation": int(sp_gene_raw["genome_id"].nunique()),
    "n_without_annotation": int(len(all_ids) - sp_gene_raw["genome_id"].nunique()),
    "n_sp_gene_rows":    int(len(sp_gene_raw)),
    "n_sp_gene_ref_rows": int(len(sp_gene_ref)),
    "timestamp":         datetime.now(timezone.utc).isoformat(),
    "software_version":  setup.software_versions(),
})
log.info("Done.")
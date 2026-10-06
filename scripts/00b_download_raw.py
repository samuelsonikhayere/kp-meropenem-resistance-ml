"""
00b_download_raw.py — Fresh download of BV-BRC genome metadata and AMR
phenotype tables, via the public RQL API, with resume and checksum.

Outputs:
  raw_data/BVBRC_genome.csv          (all K. pneumoniae genomes, all columns)
  raw_data/BVBRC_genome_amr.csv      (all meropenem records, all columns)
  provenance/00b_download_raw.json   (queries, counts, checksums, timing)

Resume behaviour:
  Checkpoints after each batch. If the script is interrupted and re-run,
  it resumes from the last completed offset for each table.

API notes:
  - Genome endpoint   : https://www.bv-brc.org/api/genome/
  - genome_amr endpoint: https://www.bv-brc.org/api/genome_amr/
  - RQL filter for K. pneumoniae: eq(species,Klebsiella pneumoniae)
    (or taxon_id,573 — both work; species name is more readable)
  - RQL filter for meropenem: eq(antibiotic,meropenem)
  - The API caps `limit` at 25,000 per request.
  - CSV responses are requested via the http_accept query parameter.
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

setup = SourceFileLoader("setup", str(_P(__file__).parent / "00_setup.py")).load_module()
log = setup.get_logger("00b_download_raw")

# ─────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────
BASE_URL = "https://www.bv-brc.org/api"

# Per your D1 choice: 5,000 per page, 1 second pause between requests
PAGE_SIZE = 5_000
SLEEP_SECONDS = 1.0

# Hard cap to prevent runaway loops in case of API misbehaviour
MAX_PAGES_PER_TABLE = 100

# API endpoint configs — one entry per table
TABLES = {
    "genome": {
        "endpoint": "genome",
        "rql_filter": 'eq(species,Klebsiella pneumoniae)',
        "output_csv": setup.RAW / "BVBRC_genome.csv",
        "provenance_key": "genome_metadata",
        "description": "All K. pneumoniae genomes, all columns",
    },
    "genome_amr": {
        "endpoint": "genome_amr",
        "rql_filter": 'and(eq(taxon_id,573),eq(antibiotic,meropenem))',  # CHANGED
        "output_csv": setup.RAW / "BVBRC_genome_amr.csv",
        "provenance_key": "amr_phenotypes",
        "description": "All meropenem AMR records for K. pneumoniae, all columns",
    },
}

# Checkpoint directory for resume support
CHECKPOINT_DIR = setup.PROCESSED / "_download_checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────
def sha256_file(path) -> str:
    """Streaming SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_query(endpoint: str, rql_filter: str, offset: int, limit: int) -> str:
    """Build the RQL query URL fragment for a given table, offset, and limit."""
    # URL-encode the filter components individually (per API docs)
    # but keep RQL operators unencoded.
    import urllib.parse
    encoded_filter = rql_filter.replace(" ", "%20")
    return (
        f"{endpoint}/?"
        f"{encoded_filter}"
        f"&limit({limit},{offset})"
        f"&http_accept=text/csv"
    )


def download_table(name: str, config: dict) -> dict:
    """
    Download a single table with pagination, resume, and provenance.
    Returns a dict summarising the download.
    """
    endpoint = config["endpoint"]
    rql      = config["rql_filter"]
    out_csv  = config["output_csv"]
    checkpoint = CHECKPOINT_DIR / f"{name}_checkpoint.json"

    # ── Resume: load previous state if it exists ─────────────────────────
    offset = 0
    frames: list[pd.DataFrame] = []
    pages_fetched = 0
    resumed_from = 0

    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        offset = state.get("offset", 0)
        resumed_from = offset
        log.info(f"[{name}] Resuming from offset {offset}")

        # Re-read any already-saved partial CSV
        partial = CHECKPOINT_DIR / f"{name}_partial.csv"
        if partial.exists():
            frames.append(pd.read_csv(partial, low_memory=False))
            log.info(f"[{name}] Loaded {len(frames[0])} rows from partial checkpoint")

    # ── Paginated fetch loop ─────────────────────────────────────────────
    total_rows = sum(len(f) for f in frames)
    t0 = time.time()

    for page in range(MAX_PAGES_PER_TABLE):
        query = build_query(endpoint, rql, offset, PAGE_SIZE)
        url   = f"{BASE_URL}/{query}"

        log.info(f"[{name}] Page {page + 1}: offset={offset}")
        try:
            r = requests.get(url, timeout=180)
        except requests.RequestException as e:
            log.error(f"[{name}] Network error on page {page + 1}: {e}")
            log.info(f"[{name}] Saving checkpoint and exiting for resume.")
            _save_checkpoint(name, offset, frames)
            raise

        if r.status_code != 200:
            log.error(f"[{name}] HTTP {r.status_code}: {r.text[:300]}")
            _save_checkpoint(name, offset, frames)
            raise RuntimeError(f"API returned HTTP {r.status_code}")

        if not r.text.strip():
            log.info(f"[{name}] Empty response — end of table reached.")
            break

        df_page = pd.read_csv(io.StringIO(r.text), low_memory=False)
        n_page = len(df_page)

        if n_page == 0:
            log.info(f"[{name}] Zero rows returned — end of table reached.")
            break

        frames.append(df_page)
        total_rows += n_page
        pages_fetched += 1

        log.info(f"[{name}] Page {page + 1}: {n_page} rows (cumulative {total_rows})")

        # If fewer rows than the page size, we've reached the end
        if n_page < PAGE_SIZE:
            log.info(f"[{name}] Received fewer than {PAGE_SIZE} rows — end of table.")
            break

        offset += PAGE_SIZE

        # Checkpoint after every page
        _save_checkpoint(name, offset, frames)

        # Be polite
        time.sleep(SLEEP_SECONDS)

    elapsed = time.time() - t0

    # ── Concatenate and save final CSV ───────────────────────────────────
    if not frames:
        raise RuntimeError(f"[{name}] No data downloaded.")

    final = pd.concat(frames, ignore_index=True)

    # Deduplicate defensively (pagination overlap can rarely duplicate rows)
    n_before_dedup = len(final)
    final = final.drop_duplicates()
    n_dropped = n_before_dedup - len(final)
    if n_dropped:
        log.warning(f"[{name}] Dropped {n_dropped} duplicate rows after concatenation.")

    final.to_csv(out_csv, index=False)
    log.info(f"[{name}] Saved: {out_csv.relative_to(setup.ROOT)}  ({final.shape})")

    # ── Checksum ─────────────────────────────────────────────────────────
    checksum = sha256_file(out_csv)
    log.info(f"[{name}] SHA256: {checksum}")

    # ── Cleanup checkpoint on success ────────────────────────────────────
    if checkpoint.exists():
        checkpoint.unlink()
    partial = CHECKPOINT_DIR / f"{name}_partial.csv"
    if partial.exists():
        partial.unlink()

    return {
        "rows":            int(len(final)),
        "columns":         int(final.shape[1]),
        "pages_fetched":   int(pages_fetched),
        "resumed_from":    int(resumed_from),
        "elapsed_seconds": round(elapsed, 1),
        "sha256":          checksum,
        "output":          str(out_csv.relative_to(setup.ROOT)),
    }


def _save_checkpoint(name: str, offset: int, frames: list[pd.DataFrame]) -> None:
    """Write current offset and a partial CSV so a crashed run can resume."""
    state = {
        "offset":    offset,
        "saved_at":  datetime.now(timezone.utc).isoformat(),
    }
    (CHECKPOINT_DIR / f"{name}_checkpoint.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            CHECKPOINT_DIR / f"{name}_partial.csv", index=False
        )


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 70)
    log.info("BV-BRC download — fresh, paginated, resumable")
    log.info(f"Page size: {PAGE_SIZE}, sleep: {SLEEP_SECONDS}s")
    log.info("=" * 70)

    provenance = {
        "script":      "00b_download_raw.py",
        "run_at":      datetime.now(timezone.utc).isoformat(),
        "software":    setup.software_versions(),
        "api_base":    BASE_URL,
        "page_size":   PAGE_SIZE,
        "sleep_seconds": SLEEP_SECONDS,
        "tables":      {},
    }

    for name, config in TABLES.items():
        log.info("")
        log.info(f"Downloading table: {name} — {config['description']}")
        log.info(f"  Endpoint: {config['endpoint']}")
        log.info(f"  Filter:   {config['rql_filter']}")

        try:
            summary = download_table(name, config)
            provenance["tables"][config["provenance_key"]] = {
                "endpoint":        config["endpoint"],
                "rql_filter":      config["rql_filter"],
                "output_csv":      summary["output"],
                "rows":            summary["rows"],
                "columns":         summary["columns"],
                "pages_fetched":   summary["pages_fetched"],
                "resumed_from":    summary["resumed_from"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "sha256":          summary["sha256"],
            }
        except Exception as e:
            log.error(f"FAILED: {name}: {e}")
            provenance["tables"][config["provenance_key"]] = {"error": str(e)}
            raise

    provenance["completed_at"] = datetime.now(timezone.utc).isoformat()
    prov_path = setup.PROVENANCE / "00b_download_raw.json"
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    log.info("")
    log.info(f"Provenance written: {prov_path.relative_to(setup.ROOT)}")

    # Emit machine-readable result object
    setup.write_result("00b_download_raw", {
        "analysis_name":   "bvbrc_download",
        "dataset_version": "v1",
        "timestamp":       provenance["completed_at"],
        "software_version": setup.software_versions(),
        "tables":          provenance["tables"],
    })

    log.info("Download complete.")
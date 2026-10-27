"""04: assign continental regions from isolation_country and refresh both analysis tables.
Unmapped countries stop the script so nothing is silently dropped. Missing country = 'Not recorded'."""
import importlib.util, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
setup = importlib.import_module("00_setup")
from importlib import import_module
import pandas as pd
S = setup
log = S.get_logger("04_region_assignment")

def assign(df):
    c = df["isolation_country"].astype("string").str.strip()
    reg = c.map(S.REGION_MAP)
    bad = sorted(c[c.notna() & reg.isna()].unique())
    if bad:
        raise SystemExit(f"Unmapped countries, add them to REGION_MAP: {bad}")
    df["region"] = reg.fillna(S.NOT_RECORDED).astype(object)
    df["is_caribbean"] = c.isin(S.CARIBBEAN).astype(int)
    df["region_caribbean_separate"] = df["region"].where(df["is_caribbean"] == 0, "Caribbean")
    df.loc[df["region"] == S.NOT_RECORDED, "region_caribbean_separate"] = S.NOT_RECORDED
    df.drop(columns=[x for x in ["region_sensitivity"] if x in df], inplace=True)
    return df

for f in ["04_with_regions.csv", "06_analysis_table.csv", "06_analysis_table_full.csv"]:
    p = S.PROCESSED / f
    df = assign(pd.read_csv(p, low_memory=False, dtype={"genome_id": str}))
    df.to_csv(p, index=False)
    log.info(f"{f}: {len(df)} rows; " + str(df["region"].value_counts().to_dict()))

full = pd.read_csv(S.PROCESSED / "06_analysis_table_full.csv", low_memory=False, dtype={"genome_id": str})
ann = full[full.has_annotation == 1]
rows = []
for r in S.ordered_regions(full.region):
    a, b = full[full.region == r], ann[ann.region == r]
    rows.append(dict(Region=r, Verified_n=len(a), Verified_resistant=int(a.label.sum()),
                     Verified_resistant_pct=round(100 * a.label.mean(), 1), Annotated_n=len(b),
                     Annotated_pct_of_cohort=round(100 * len(b) / len(ann), 1),
                     Annotated_resistant=int(b.label.sum()), Annotated_resistant_pct=round(100 * b.label.mean(), 1)))
S.save_csv(pd.DataFrame(rows), "table3_region_distribution.csv")
audit = full.groupby(["region", full.isolation_country.fillna("(no country)")]).size().reset_index(name="n")
S.save_csv(audit, "mapping_audit.csv")
log.info("Caribbean isolates (verified): %d, WTSI share: %d" % (full.is_caribbean.sum(),
         ((full.is_caribbean == 1) & (full.sequencing_center == "WTSI")).sum()))

"""07: Table 1 (cohort, overall and by region), Table S1 (single-centre effect), Table S2 (feature matrix summary)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import importlib; S = importlib.import_module("00_setup")
import numpy as np, pandas as pd
log = S.get_logger("07_cohort_characteristics")
full = S.load_table(True); ann = full[full.has_annotation == 1]; f = S.feature_columns(full)
yr = pd.to_numeric(full.collection_year, errors="coerce")
ctr = full.sequencing_center.fillna("Not recorded")
pct = lambda a, b: f"{a} ({100 * a / b:.1f}%)"
n = len(full)
overall = [("Verified isolates, n", n), ("Annotated isolates (modelling cohort), n", len(ann)),
 ("Resistant, n (%)", pct(int(full.label.sum()), n)), ("Susceptible, n (%)", pct(int((1 - full.label).sum()), n)),
 ("Countries recorded, n", full.isolation_country.nunique()), ("Country not recorded, n (%)", pct(int(full.isolation_country.isna().sum()), n)),
 ("Collection year recorded, n (%)", pct(int(yr.notna().sum()), n)),
 ("Collection year, median (range)", f"{int(yr.median())} ({int(yr.min())} to {int(yr.max())})"),
 ("Sequencing centre recorded, n (%)", pct(int((ctr != 'Not recorded').sum()), n)),
 ("Sequence type recorded, n (%)", pct(int(full.mlst.notna().sum()), n)),
 ("Genome quality 'Good', n (% of recorded)", pct(int((full.genome_quality == 'Good').sum()), int(full.genome_quality.notna().sum()))),
 ("Tier 1 (MIC based) phenotype, n (%)", pct(int((full.method_tier == 1).sum()), n)),
 ("Tier 3 (disk diffusion) phenotype, n (%)", pct(int((full.method_tier == 3).sum()), n)),
 ("Annotated, n (%)", pct(len(ann), n))]
S.save_csv(pd.DataFrame(overall, columns=["Characteristic", "Value"]), "table1_overall.csv")
rows = []
for r in S.ordered_regions(full.region):
    a = full[full.region == r]; b = ann[ann.region == r]; y_ = yr[a.index]
    rows.append({"Region": r, "Verified n": len(a), "Annotated n": len(b), "Annotated % of modelling cohort": round(100 * len(b) / len(ann), 1),
                 "Resistant % (annotated)": round(100 * b.label.mean(), 1), "Median collection year": y_.median(), "Year recorded %": round(100 * y_.notna().mean(), 1),
                 "Countries": a.isolation_country.nunique(), "Distinct sequence types": a.mlst.nunique()})
S.save_csv(pd.DataFrame(rows), "table1_by_region.csv")
rows = []
for r in S.ordered_regions(full.region):
    a = full[full.region == r]; c = ctr[a.index]; vc = c.value_counts(); top = vc.index[0]
    named = vc.drop("Not recorded", errors="ignore"); tn = named.index[0] if len(named) else "none"
    m = (c == tn).values
    rows.append({"Region": r, "Verified n": len(a), "Centre not recorded %": round(100 * (c == "Not recorded").mean(), 1),
                 "Largest named centre": tn, "Centre n": int(m.sum()), "Centre % of region": round(100 * m.mean(), 1),
                 "Resistant % in centre": round(100 * a.label[m].mean(), 1) if m.any() else None,
                 "Resistant % elsewhere in region": round(100 * a.label[~m].mean(), 1) if (~m).any() else None})
S.save_csv(pd.DataFrame(rows), "tableS1_single_source_effect.csv")
X = full.loc[full.has_annotation == 1, f].values; cc = S.carbapenemase_columns(f)
fs = [("Verified cohort size", n), ("Annotated genomes (modelling cohort)", len(ann)), ("Genomes without annotation", n - len(ann)),
 ("Unique AMR products (features)", len(f)), ("Median features per genome", float(np.median(X.sum(1)))), ("Mean features per genome", round(float(X.sum(1).mean()), 1)),
 ("Features in more than 90% of genomes", int((X.mean(0) > .9).sum())), ("Features in fewer than 10% of genomes", int((X.mean(0) < .1).sum())),
 ("Carbapenemase product strings", len(cc))]
S.save_csv(pd.DataFrame(fs, columns=["Metric", "Value"]), "tableS2_feature_matrix_summary.csv")
S.write_result("07_cohort_characteristics", {"overall": overall, "carbapenemase_products": cc})
for t in ["table1_overall", "table1_by_region", "tableS1_single_source_effect", "tableS2_feature_matrix_summary"]:
    log.info("\n" + pd.read_csv(S.RESULTS / f"{t}.csv").to_string())

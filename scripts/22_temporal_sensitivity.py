"""22: sensitivity of the temporal holdout to the cut-off year, and comparison of isolates with and without a collection year.
The main temporal analysis (script 08) uses train < 2016, test >= 2016. Here the same models are refitted at
every collection year that appears in the data (the earliest year is skipped because nothing precedes it),
however small the training or test set. Years with no isolates are not listed.
Cut-offs where a model cannot be fitted or AUROC is undefined (one class only) are kept as rows with a note. Isolates with no collection year are excluded, as in script 08."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import importlib; S = importlib.import_module("00_setup")
import numpy as np, pandas as pd
log = S.get_logger("22_temporal_sensitivity")

df = S.load_table(); feats = S.feature_columns(df)
assert len(feats) == S.N_FEATURES_EXPECTED, len(feats)
X = df[feats].values.astype("float32"); y = df["label"].values.astype(int)
year = pd.to_numeric(df["collection_year"], errors="coerce").values
ok = ~np.isnan(year)
yrs = sorted(set(int(v) for v in year[ok]))
CUTOFFS = yrs[1:]   # every year that appears in the data; the earliest year has no isolates before it

rows = []
for cut in CUTOFFS:
    tr = np.where(ok & (year < cut))[0]; te = np.where(ok & (year >= cut))[0]
    base = dict(Cutoff=cut, n_train=len(tr), n_test=len(te),
                n_train_resistant=int(y[tr].sum()), n_test_resistant=int(y[te].sum()),
                train_resistant_pct=round(100 * y[tr].mean(), 1) if len(tr) else None,
                test_resistant_pct=round(100 * y[te].mean(), 1) if len(te) else None)
    if len(tr) == 0 or len(te) == 0 or len(set(y[tr])) < 2:
        for m in S.MODELS: rows.append({**base, "Model": m, "AUROC": None, "CI_low": None, "CI_high": None, "AUPRC": None, "Note": "model cannot be fitted (training set empty or one class)"})
        log.info(f"cut-off {cut}: not fitted (train {len(tr)}, test {len(te)})"); continue
    for m in S.MODELS:
        p = S.make_model(m).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        if len(set(y[te])) < 2:
            rows.append({**base, "Model": m, "AUROC": None, "CI_low": None, "CI_high": None, "AUPRC": None, "Note": "AUROC undefined (test set has one class)"}); continue
        met = S.compute_metrics(y[te], p); lo, hi = S.bootstrap_auc(y[te], p)
        rows.append({**base, "Model": m, "AUROC": round(met["auroc"], 3), "CI_low": round(lo, 3), "CI_high": round(hi, 3),
                     "AUPRC": round(met["auprc"], 3), "Note": ""})
        log.info(f"cut-off {cut} {m}: AUROC {met['auroc']:.3f} ({lo:.3f} to {hi:.3f}), train {len(tr)}, test {len(te)}")
T = pd.DataFrame(rows); S.save_csv(T, "tableS12_temporal_cutoff_sensitivity.csv")

# isolates with and without a collection year (the temporal analysis applies only to the dated subset)
reg = df["region"].values; known = ok
rows = []
for name, mask in (("Collection year recorded", known), ("Collection year missing", ~known)):
    r = dict(Group=name, n=int(mask.sum()), resistant_pct=round(100 * y[mask].mean(), 1))
    vc = pd.Series(reg[mask]).value_counts(normalize=True) * 100
    for g in S.ordered_regions(set(reg)): r[f"{g} (%)"] = round(float(vc.get(g, 0)), 1)
    rows.append(r)
G = pd.DataFrame(rows); S.save_csv(G, "tableS12b_dated_vs_undated.csv")
log.info("\n" + G.to_string(index=False))
json.dump({"cutoffs": T.to_dict("records"), "dated_vs_undated": G.to_dict("records")},
          open(S.RESULTS / "temporal_sensitivity.json", "w"), indent=1)
print(T.to_string(index=False)); print(G.to_string(index=False))
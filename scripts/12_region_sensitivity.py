"""12: sensitivity of leave-one-region-out results to the region definition."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import importlib, json; S = importlib.import_module("00_setup")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

log = S.get_logger("12_region_sensitivity")
df = S.load_table(); f = S.feature_columns(df); X = df[f].values.astype("float32"); y = df.label.values.astype(int)
def loro(mask, labels, tag):
    out = []
    for r in S.ordered_regions(labels[mask]) if tag != "caribbean_separate" else sorted(set(labels[mask])):
        te = mask & (labels == r); tr = mask & (labels != r); row = dict(Mapping=tag, Region=r, n=int(te.sum()), Resistant_pct=round(100 * y[te].mean(), 1))
        for m in S.MODELS:
            p = S.make_model(m).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            row[f"{m} AUROC"] = round(roc_auc_score(y[te], p), 3) if len(set(y[te])) == 2 else None
        out.append(row)
    return out
reg = df.region.values; res = []
allm = np.ones(len(df), bool)
res += loro(allm, reg, "primary (continental)")
res += loro(allm, df.region_caribbean_separate.values, "caribbean_separate")
res += loro(~(df.is_caribbean == 1).values, reg, "Caribbean isolates removed")
res += loro((df.region != S.NOT_RECORDED).values, reg, "Not recorded removed")
t = pd.DataFrame(res); S.save_csv(t, "tableS5_loo_region_sensitivity.csv")
S.write_result("12_region_sensitivity", res); log.info("\n" + t.to_string())

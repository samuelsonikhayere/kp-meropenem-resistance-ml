"""09: 95% bootstrap CIs (2,000 resamples) on out-of-fold predictions; lineage-level (cluster) bootstrap for MLST."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import importlib, json; S = importlib.import_module("00_setup")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

log = S.get_logger("09_bootstrap_ci")
df = S.load_table(); oof = pd.read_csv(S.RESULTS / "oof_predictions.csv"); y = oof.label.values
st = df["mlst"].fillna("UNKNOWN").astype(str).values
rows = []
def add(model, scheme, mask, key, groups=None, note=""):
    m = np.asarray(mask); p = oof.loc[m, key].values
    lo, hi = S.bootstrap_auc(y[m], p, None if groups is None else groups[m])
    rows.append(dict(Model=model, Scheme=scheme, n=int(m.sum()), AUROC=roc_auc_score(y[m], p), CI_low=lo, CI_high=hi, Note=note))
allm = np.ones(len(y), bool)
for m in S.MODELS:
    add(m, "Random 5-fold", allm, f"{m}__random")
    add(m, "MLST-grouped 5-fold", allm, f"{m}__mlst", note="resampled isolates")
    add(m, "MLST-grouped 5-fold (lineage bootstrap)", allm, f"{m}__mlst", groups=st, note="resampled sequence types")
    t = oof[f"{m}__temporal"].notna().values
    add(m, "Temporal holdout", t, f"{m}__temporal")
    for r in S.ordered_regions(oof.region):
        add(m, f"Held-out region: {r}", (oof.region == r).values, f"{m}__loro")
out = pd.DataFrame(rows); S.save_csv(out.round(3), "tableS4_bootstrap_cis.csv")
S.write_result("09_bootstrap_ci", out.round(4).to_dict("records")); log.info("\n" + out.round(3).to_string())

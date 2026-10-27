"""11: baselines. A model has to beat all of these to count as learning resistance biology."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import importlib, json; S = importlib.import_module("00_setup")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold
log = S.get_logger("11_baseline_models")
full = S.load_table(True); df = S.load_table(); f = S.feature_columns(df)
X = df[f].values.astype("float32"); y = df.label.values.astype(int)
rows = []
# B0 majority class
rows.append(dict(Baseline="Majority class", AUROC=0.5, SD=None, Note="constant prediction"))
# B1 annotation availability (does having a matrix at all predict the label?) on the full verified cohort
a = full.has_annotation.values.reshape(-1, 1).astype(float); yy = full.label.values.astype(int); sc = []
for tr, te in StratifiedKFold(5, shuffle=True, random_state=S.RANDOM_SEED).split(a, yy):
    sc.append(roc_auc_score(yy[te], S.make_model("logistic_regression").fit(a[tr], yy[tr]).predict_proba(a[te])[:, 1]))
rows.append(dict(Baseline="Annotation availability only", AUROC=np.mean(sc), SD=np.std(sc, ddof=1), Note="LR on a yes/no flag, random 5-fold, verified cohort n=%d" % len(yy)))
# B2 batch-marker products only, 5-fold CV
MARK = ["DNA gyrase subunit A (EC 5.99.1.3)","DNA-directed RNA polymerase beta subunit (EC 2.7.7.6)",
 "DNA-directed RNA polymerase beta' subunit (EC 2.7.7.6)","Transcription termination factor Rho","Translation elongation factor Tu",
 "Multidrug efflux pump EmrD (of MFS type)","Multidrug resistance regulator EmrR (MprA)",
 "Multidrug efflux system AcrAB-TolC, membrane fusion component AcrA","Multiple antibiotic resistance protein MarA",
 "Multiple antibiotic resistance protein MarR","Multidrug efflux system AcrAD-TolC, inner-membrane proton/drug antiporter AcrD (RND type)"]
# The sulfonamide resistance product was removed from the batch-marker set. It is a resistance product
# present in only some genomes, so it can carry resistance signal. It is reported as its own baseline below.
SULF = "Dihydropteroate synthase type-2 (EC 2.5.1.15) @ Sulfonamide resistance protein"
pm = [m for m in MARK if m in f]; idx = [f.index(m) for m in pm]
log.info(f"Batch markers listed {len(MARK)}, present in the feature matrix {len(pm)}; not present: {[m for m in MARK if m not in f]}")
def cv(Xs):
    s = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=S.RANDOM_SEED).split(Xs, y):
        s.append(roc_auc_score(y[te], S.make_model("logistic_regression").fit(Xs[tr], y[tr]).predict_proba(Xs[te])[:, 1]))
    return float(np.mean(s)), float(np.std(s, ddof=1))
m_, s_ = cv(X[:, idx]); rows.append(dict(Baseline=f"Batch-marker products only ({len(pm)} present of {len(MARK)})", AUROC=m_, SD=s_, Note="LR, random 5-fold"))
if SULF in f:
    m_, s_ = cv(X[:, [f.index(SULF)]]); rows.append(dict(Baseline="Sulfonamide resistance product alone", AUROC=m_, SD=s_, Note="LR, random 5-fold; resistance product, not a batch marker"))
afr = (df.region == "Africa").values
pa = S.make_model("logistic_regression").fit(X[~afr][:, idx], y[~afr]).predict_proba(X[afr][:, idx])[:, 1]
lo_, hi_ = S.bootstrap_auc(y[afr], pa)
rows.append(dict(Baseline="Batch-marker products only, African holdout", AUROC=roc_auc_score(y[afr], pa), SD=None, CI_low=lo_, CI_high=hi_, Note=f"trained on other regions, n={int(afr.sum())}; 95% CI from 2,000 bootstrap resamples"))
# B3 one-line rule: any carbapenemase product present => resistant (no training, so valid in every region)
cc = S.carbapenemase_columns(f); ci = [f.index(c) for c in cc]
rule = (X[:, ci].sum(1) > 0).astype(float)
lo_, hi_ = S.bootstrap_auc(y, rule)
rows.append(dict(Baseline=f"Carbapenemase presence rule ({len(cc)} products)", AUROC=roc_auc_score(y, rule), SD=None, CI_low=lo_, CI_high=hi_, Note="no training; all annotated isolates; 95% CI from 2,000 bootstrap resamples"))
# B4 LR on carbapenemase products only, random CV
m_, s_ = cv(X[:, ci]); rows.append(dict(Baseline="LR on carbapenemase products only", AUROC=m_, SD=s_, Note="random 5-fold"))
# rule by region
rr = []
tp = lambda yt, rl: (float(((rl == 1) & (yt == 1)).sum() / max((yt == 1).sum(), 1)), float(((rl == 0) & (yt == 0)).sum() / max((yt == 0).sum(), 1)))
for r in S.ordered_regions(df.region):
    mk = (df.region == r).values; se, sp = tp(y[mk], rule[mk])
    rr.append(dict(Region=r, n=int(mk.sum()), Resistant_pct=round(100 * y[mk].mean(), 1), Rule_sensitivity=round(se, 3),
                   Rule_specificity=round(sp, 3), Rule_AUROC=round(roc_auc_score(y[mk], rule[mk]), 3) if len(set(y[mk])) == 2 else None))
oof = pd.read_csv(S.RESULTS / "oof_predictions.csv")
for m in S.MODELS:
    for d in rr:
        mk = (oof.region == d["Region"]).values
        d[f"{m} held-out AUROC"] = round(roc_auc_score(y[mk], oof.loc[mk, f"{m}__loro"]), 3)
t = pd.DataFrame(rows); S.save_csv(t.round(3), "tableS3_baselines.csv"); S.save_csv(pd.DataFrame(rr), "tableS3b_carbapenemase_rule_by_region.csv")
S.write_result("11_baseline_models", {"baselines": t.round(4).to_dict("records"), "rule_by_region": rr,
                                      "carbapenemase_products": cc, "batch_markers_present": pm})
log.info("\n" + t.round(3).to_string()); log.info("\n" + pd.DataFrame(rr).to_string())
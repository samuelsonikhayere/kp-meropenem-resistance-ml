"""08: four validation schemes for both models. Saves metrics, fold SDs and out-of-fold predictions."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import importlib; S = importlib.import_module("00_setup")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
log = S.get_logger("08_model_evaluation")

df = S.load_table(); feats = S.feature_columns(df)
assert len(feats) == S.N_FEATURES_EXPECTED, len(feats)
X = df[feats].values.astype("float32"); y = df["label"].values.astype(int)
mlst = df["mlst"].fillna("UNKNOWN").astype(str).values
year = pd.to_numeric(df["collection_year"], errors="coerce").values
region = df["region"].values
oof = pd.DataFrame({"genome_id": df.genome_id, "region": region, "label": y})
res = {}

def fit_pred(m, tr, te):
    mod = S.make_model(m).fit(X[tr], y[tr]); return mod.predict_proba(X[te])[:, 1]

def run_cv(m, splits, key):
    p = np.full(len(y), np.nan); folds = []; pr = []
    for tr, te in splits:
        p[te] = fit_pred(m, tr, te)
        if len(set(y[te])) == 2: folds.append(roc_auc_score(y[te], p[te])); pr.append(average_precision_score(y[te], p[te]))
    met = S.compute_metrics(y, p)
    met.update(fold_aurocs=folds, auroc_sd=float(np.std(folds, ddof=1)), auroc_mean_of_folds=float(np.mean(folds)), auprc_mean_of_folds=float(np.mean(pr)), auprc_sd=float(np.std(pr, ddof=1)))
    met.update({"calibration": S.calibration_stats(y, p)})
    res[f"{m}__{key}"] = met; oof[f"{m}__{key}"] = p
    log.info(f"{m} {key}: AUROC {met['auroc']:.3f} (fold SD {met['auroc_sd']:.3f})")

skf = list(StratifiedKFold(5, shuffle=True, random_state=S.RANDOM_SEED).split(X, y))
gkf = list(GroupKFold(5).split(X, y, groups=mlst))
for m in S.MODELS:
    run_cv(m, skf, "random")
    run_cv(m, gkf, "mlst")
    # temporal: train before 2016, test 2016 onward, isolates without a year are excluded
    ok = ~np.isnan(year); tr = np.where(ok & (year < 2016))[0]; te = np.where(ok & (year >= 2016))[0]
    p = fit_pred(m, tr, te); met = S.compute_metrics(y[te], p)
    met.update(n_train=int(len(tr)), n_test=int(len(te)), n_excluded_missing_year=int((~ok).sum()),
               train_prevalence=float(y[tr].mean()), calibration=S.calibration_stats(y[te], p))
    res[f"{m}__temporal"] = met; oof[f"{m}__temporal"] = np.nan; oof.loc[te, f"{m}__temporal"] = p
    log.info(f"{m} temporal: AUROC {met['auroc']:.3f} (train {len(tr)}, test {len(te)})")
    # leave-one-region-out
    loro = {}
    for r in S.ordered_regions(region):
        te = np.where(region == r)[0]; tr = np.where(region != r)[0]
        p = fit_pred(m, tr, te); met = S.compute_metrics(y[te], p)
        met.update(n_train=int(len(tr)), train_prevalence=float(y[tr].mean()), calibration=S.calibration_stats(y[te], p))
        loro[r] = met; oof.loc[te, f"{m}__loro"] = p
        log.info(f"{m} LORO {r}: AUROC {met.get('auroc', float('nan')):.3f} (n={met['n']})")
    res[f"{m}__loro"] = loro

json.dump(res, open(S.RESULTS / "08_model_results.json", "w"), indent=2)
oof.to_csv(S.RESULTS / "oof_predictions.csv", index=False)

names = {"random": "Random stratified 5-fold", "mlst": "MLST-grouped 5-fold", "temporal": "Temporal (train <2016, test 2016 onward)"}
rows = []
for m in S.MODELS:
    for k, lab in names.items():
        r = res[f"{m}__{k}"]
        cv = "auroc_mean_of_folds" in r
        rows.append({"Model": m, "Scheme": lab, "AUROC (mean of folds)": r["auroc_mean_of_folds"] if cv else r["auroc"], "AUROC fold SD": r.get("auroc_sd"),
                     "AUPRC (mean of folds)": r["auprc_mean_of_folds"] if cv else r["auprc"], "AUPRC fold SD": r.get("auprc_sd"),
                     "AUROC (pooled predictions)": r["auroc"], "Accuracy (pooled)": r["accuracy"], "F1 (pooled)": r["f1"], "n": r["n"]})
S.save_csv(pd.DataFrame(rows).round(3), "table4_validation_schemes.csv")
S.save_csv(pd.DataFrame([{"Model": m, **{k: res[f"{m}__temporal"][k] for k in
    ["n_train","n_test","n_excluded_missing_year","train_prevalence","prevalence","auroc","auprc","accuracy","f1"]}}
    for m in S.MODELS]).round(3), "table5_temporal_holdout.csv")
rows = []
for r in S.ordered_regions(region):
    row = {"Region": r}
    for m in S.MODELS:
        d = res[f"{m}__loro"][r]
        row.update({f"{m} AUROC": d.get("auroc"), f"{m} AUPRC": d.get("auprc")})
    row.update(n=d["n"], Resistant_pct=round(100 * d["prevalence"], 1), Train_prevalence_pct=round(100 * d["train_prevalence"], 1))
    rows.append(row)
S.save_csv(pd.DataFrame(rows).round(3), "table6_region_holdout.csv")
S.write_result("08_model_evaluation", {"versions": S.software_versions(), "summary": rows})

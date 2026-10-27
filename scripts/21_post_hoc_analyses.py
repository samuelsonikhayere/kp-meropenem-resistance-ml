"""21: post hoc analyses added after the main results (all labelled post hoc in the manuscript).
A. generalisation gap per region  B. where Europe and Oceania fail  C. leave-one-centre-out
D. local-data learning curve      E. does recalibration repair the shifted regions?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import importlib, json; S = importlib.import_module("00_setup")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
log = S.get_logger("21_post_hoc_analyses")
df = S.load_table(); f = S.feature_columns(df); X = df[f].values.astype("float32"); y = df.label.values.astype(int)
oof = pd.read_csv(S.RESULTS / "oof_predictions.csv"); reg = df.region.values
ci = [f.index(c) for c in S.carbapenemase_columns(f)]; carb = (X[:, ci].sum(1) > 0)
auc = lambda yt, p: roc_auc_score(yt, p) if len(set(yt)) == 2 else np.nan
out = {}

# A. gap = random-CV AUROC inside the region minus held-out AUROC
rows = []
for r in S.ordered_regions(reg):
    mk = reg == r; row = {"Region": r, "n": int(mk.sum())}
    for m in S.MODELS:
        w = auc(y[mk], oof.loc[mk, f"{m}__random"]); h = auc(y[mk], oof.loc[mk, f"{m}__loro"])
        row.update({f"{m} within-region AUROC": w, f"{m} held-out AUROC": h, f"{m} gap": w - h})
    rows.append(row)
A = pd.DataFrame(rows).round(3); S.save_csv(A, "tableS6_generalisation_gap.csv"); out["A_gap"] = A.to_dict("records")

# B. error analysis: for each shifted region, performance split by carbapenemase presence (LR held-out predictions)
rows = []
for r in ["Europe", "Oceania"]:
    for flag, lab in [(True, "carbapenemase present"), (False, "no carbapenemase")]:
        mk = (reg == r) & (carb == flag)
        if mk.sum() == 0: continue
        row = {"Region": r, "Group": lab, "n": int(mk.sum()), "Resistant_n": int(y[mk].sum()), "Resistant_pct": round(100 * y[mk].mean(), 1)}
        for m in S.MODELS:
            p = oof.loc[mk, f"{m}__loro"].values
            row[f"{m} mean predicted risk"] = round(float(p.mean()), 3)
            row[f"{m} AUROC"] = round(float(auc(y[mk], p)), 3) if len(set(y[mk])) == 2 else None
        rows.append(row)
B = pd.DataFrame(rows); S.save_csv(B, "tableS7_error_analysis_by_carbapenemase.csv"); out["B_error"] = B.to_dict("records")
st = df.mlst.fillna("UNKNOWN").astype(str)
top = []
for r in ["Europe", "Oceania"]:
    mk = reg == r; vc = st[mk].value_counts().head(5)
    for s_, n in vc.items():
        k = mk & (st == s_).values
        top.append({"Region": r, "ST": s_, "n": int(n), "Resistant_pct": round(100 * y[k].mean(), 1),
                    "Carbapenemase_pct": round(100 * carb[k].mean(), 1),
                    "LR mean predicted risk": round(float(oof.loc[k, "logistic_regression__loro"].mean()), 3)})
T = pd.DataFrame(top); S.save_csv(T, "tableS8_top_sequence_types_shifted_regions.csv"); out["B_st"] = top

# C. leave-one-centre-out for the largest recorded centres
ctr = df.sequencing_center.fillna("Not recorded").replace({"Houston Methodist Hospital System": "Houston Methodist Hospital"})
rows = []
for c in [k for k, v in ctr.value_counts().items() if v >= 100 and k != "Not recorded"]:
    te = (ctr == c).values; tr = ~te; row = {"Centre": c, "n": int(te.sum()), "Resistant_pct": round(100 * y[te].mean(), 1)}
    for m in S.MODELS:
        row[f"{m} AUROC"] = round(float(auc(y[te], S.make_model(m).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])), 3)
    rows.append(row)
C = pd.DataFrame(rows); S.save_csv(C, "tableS9_leave_one_centre_out.csv"); out["C_centre"] = rows

# D. learning curve: add k isolates from the target region to training, test on the remainder (20 repeats)
rows = []; rng = np.random.default_rng(S.RANDOM_SEED)
for r in ["Europe", "Oceania"]:
    idx = np.where(reg == r)[0]
    for k in [x for x in [0, 25, 50, 100, 150] if x < len(idx) - 20]:
        vals = []
        for rep in range(20 if k else 1):
            pick = rng.choice(idx, k, replace=False) if k else np.array([], int)
            te = np.setdiff1d(idx, pick); tr = np.concatenate([np.where(reg != r)[0], pick])
            if len(set(y[te])) < 2: continue
            vals.append(auc(y[te], S.make_model("logistic_regression").fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]))
        rows.append({"Region": r, "Local isolates added": k, "AUROC mean": round(float(np.mean(vals)), 3), "AUROC SD": round(float(np.std(vals)), 3) if len(vals) > 1 else None})
D = pd.DataFrame(rows); S.save_csv(D, "tableS10_learning_curve_local_data.csv"); out["D_learning"] = rows

# E. recalibration: shift the intercept so mean predicted risk equals the true local prevalence (uses local labels, so a diagnostic only)
rows = []
for r in S.ordered_regions(reg):
    mk = reg == r; p = np.clip(oof.loc[mk, "logistic_regression__loro"].values, 1e-6, 1 - 1e-6); z = np.log(p / (1 - p))
    from scipy.optimize import brentq
    d = brentq(lambda a: (1 / (1 + np.exp(-(z + a)))).mean() - y[mk].mean(), -15, 15); q = 1 / (1 + np.exp(-(z + d)))
    rows.append({"Region": r, "Mean predicted risk": round(p.mean(), 3), "Observed prevalence": round(y[mk].mean(), 3),
                 "Brier before": round(float(np.mean((p - y[mk]) ** 2)), 3), "Brier after intercept shift": round(float(np.mean((q - y[mk]) ** 2)), 3),
                 "AUROC (unchanged)": round(float(auc(y[mk], p)), 3), "Intercept shift": round(float(d), 2)})
E = pd.DataFrame(rows); S.save_csv(E, "tableS11_recalibration.csv"); out["E_recal"] = rows
S.write_result("21_post_hoc_analyses", out)
for n, t in zip("A B T C D E".split(), [A, B, T, C, D, E]): log.info(f"\n{n}\n" + t.to_string())

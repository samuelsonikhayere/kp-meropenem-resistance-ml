"""10: feature importance on the full annotated cohort (logistic regression coefficients, XGBoost gain)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import importlib, json; S = importlib.import_module("00_setup")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

log = S.get_logger("10_feature_importance")
df = S.load_table(); f = S.feature_columns(df); X = df[f].values.astype("float32"); y = df.label.values.astype(int)
carb = set(S.carbapenemase_columns(f))
lr = S.make_model("logistic_regression").fit(X, y); xg = S.make_model("xgboost").fit(X, y)
t = pd.DataFrame({"feature": f, "coefficient": lr.coef_[0], "odds_ratio": np.exp(lr.coef_[0]),
                  "xgb_importance": xg.feature_importances_, "prevalence_pct": 100 * X.mean(0),
                  "carbapenemase_product": [c in carb for c in f]})
t["abs_coef"] = t.coefficient.abs()
t["lr_rank"] = t.abs_coef.rank(ascending=False).astype(int); t["xgb_rank"] = t.xgb_importance.rank(ascending=False).astype(int)
t = t.sort_values("abs_coef", ascending=False)
S.save_csv(t.drop(columns="abs_coef").round(4), "table7_features_logreg.csv")
S.save_csv(t.sort_values("xgb_importance", ascending=False).round(4), "table7_features_xgboost.csv")
S.write_result("10_feature_importance", {"n_features": len(f), "carbapenemase_features": sorted(carb),
    "top15_lr": t.head(15).feature.tolist(), "top15_xgb": t.sort_values("xgb_importance", ascending=False).head(15).feature.tolist(),
    "overlap_top15": len(set(t.head(15).feature) & set(t.sort_values("xgb_importance", ascending=False).head(15).feature))})
log.info(t.head(15)[["feature","coefficient","xgb_importance","xgb_rank"]].to_string())

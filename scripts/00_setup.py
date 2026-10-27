"""00_setup.py: paths, seeds, constants and helpers shared by every script."""
from __future__ import annotations
import json, logging, platform, re, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent.parent
RAW, PROCESSED, RESULTS = ROOT / "raw_data", ROOT / "processed", ROOT / "results"
FIGURES, LOGS, PROVENANCE, ENV = ROOT / "figures", ROOT / "logs", ROOT / "provenance", ROOT / "env"
for d in (PROCESSED, RESULTS, FIGURES, LOGS, PROVENANCE, ENV):
    d.mkdir(exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

VERIFIED_METHOD_TIERS = {
    "MIC": 1, "Broth dilution": 1, "Agar dilution": 1, "Etest": 1,
    "VITEK": 2, "VITEK 2": 2, "Phoenix": 2, "MicroScan": 2, "Microscan": 2, "Sensititre": 2,
    "Disk diffusion": 3,
}
EXCLUDED_PHENOTYPES = {"Intermediate", "Susceptible-dose dependent"}
BINARY_PHENOTYPES = {"Resistant", "Susceptible"}

# Continental regions. Central America and the Caribbean belong to North America
# because the country was recorded for every one of these isolates.
_AFRICA = ["Algeria","Angola","Benin","Botswana","Burkina Faso","Burundi","Cabo Verde","Cameroon",
 "Central African Republic","Chad","Comoros","Democratic Republic of the Congo","Congo","Republic of the Congo",
 "Djibouti","Egypt","Equatorial Guinea","Eritrea","Eswatini","Ethiopia","Gabon","Gambia","Ghana","Guinea",
 "Guinea-Bissau","Cote d'Ivoire","Ivory Coast","Kenya","Lesotho","Liberia","Libya","Madagascar","Malawi","Mali",
 "Mauritania","Mauritius","Morocco","Mozambique","Namibia","Niger","Nigeria","Rwanda","Sao Tome and Principe",
 "Senegal","Sierra Leone","Somalia","South Africa","South Sudan","Sudan","Tanzania","Togo","Tunisia","Uganda",
 "Zambia","Zimbabwe"]
_ASIA = ["China","India","Japan","South Korea","Korea","Taiwan","Thailand","Vietnam","Singapore","Malaysia",
 "Indonesia","Philippines","Pakistan","Bangladesh","Sri Lanka","Nepal","Iran","Iraq","Israel","Saudi Arabia",
 "United Arab Emirates","Qatar","Kuwait","Oman","Jordan","Lebanon","Turkey","Russia","Kazakhstan","Uzbekistan",
 "Afghanistan"]
_EUROPE = ["United Kingdom","Ireland","France","Germany","Italy","Spain","Portugal","Netherlands","Belgium",
 "Switzerland","Austria","Sweden","Norway","Denmark","Finland","Iceland","Poland","Czech Republic","Czechia",
 "Slovakia","Hungary","Romania","Bulgaria","Greece","Croatia","Serbia","Slovenia","Estonia","Latvia","Lithuania",
 "Ukraine","Belarus","Moldova"]
CARIBBEAN = ["Barbados","Trinidad","Trinidad and Tobago","Dominica","Saint Kitts","Haiti","Saint Lucia",
 "Saint Vincent","Cayman Islands","Grenada","Bermuda","Antigua","Dominican Republic","Jamaica","Bahamas","Cuba",
 "Puerto Rico","Guadeloupe","Martinique"]
_NORTH_AMERICA = ["USA","United States","Canada","Mexico","Honduras","Belize","Guatemala","Costa Rica","Panama",
 "El Salvador","Nicaragua"] + CARIBBEAN
_SOUTH_AMERICA = ["Brazil","Argentina","Chile","Colombia","Peru","Venezuela","Ecuador","Bolivia","Uruguay","Paraguay"]
_OCEANIA = ["Australia","New Zealand","Guam"]
REGION_MAP = {}
for _r, _l in [("Africa",_AFRICA),("Asia",_ASIA),("Europe",_EUROPE),("North America",_NORTH_AMERICA),
               ("South America",_SOUTH_AMERICA),("Oceania",_OCEANIA)]:
    REGION_MAP.update({c: _r for c in _l})
NOT_RECORDED = "Not recorded"
REGION_ORDER = ["North America","Asia","Europe","Oceania","Africa","South America",NOT_RECORDED]

ANALYSIS_METADATA_COLUMNS = ["genome_id","label","region","region_caribbean_separate","is_caribbean",
    "region_sensitivity","isolation_country","sequencing_center","mlst","genome_quality","collection_year",
    "method_tier","has_annotation"]
N_FEATURES_EXPECTED = 377

def feature_columns(df):
    return [c for c in df.columns if c not in ANALYSIS_METADATA_COLUMNS]

LR_PARAMS = dict(max_iter=5000, C=1.0, solver="lbfgs", random_state=RANDOM_SEED)
XGB_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8,
                  colsample_bytree=0.8, random_state=RANDOM_SEED, eval_metric="logloss", n_jobs=4)

def make_model(name):
    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**LR_PARAMS)
    from xgboost import XGBClassifier
    return XGBClassifier(**XGB_PARAMS)
MODELS = ["logistic_regression", "xgboost"]

# Carbapenemase products, matched on the feature (product) name. Plain "Imp" outer membrane protein
# (LptD) is not a carbapenemase, so IMP is matched only as the metallo-beta-lactamase family.
CARBAPENEMASE_REGEX = r"=> (KPC|NDM|VIM|IMP) family|carbapenem-hydrolyzing"
def carbapenemase_columns(cols):
    rx = re.compile(CARBAPENEMASE_REGEX)
    return [c for c in cols if rx.search(c)]

def get_logger(name):
    lg = logging.getLogger(name)
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    f = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(LOGS / f"{name}.log", mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(f); lg.addHandler(h)
    return lg

def software_versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform(), "pandas": pd.__version__,
           "numpy": np.__version__, "sklearn": sklearn.__version__}
    try:
        import xgboost; out["xgboost"] = xgboost.__version__
    except ImportError:
        out["xgboost"] = None
    out["captured_at"] = datetime.now(timezone.utc).isoformat()
    return out

def write_result(name, payload):
    """Store a result under its analysis name in results/all_results.json (re-runs replace, never duplicate)."""
    f = RESULTS / "all_results.json"
    cur = {}
    if f.exists():
        try:
            cur = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(cur, list):
                cur = {}
        except json.JSONDecodeError:
            cur = {}
    cur[name] = payload
    f.write_text(json.dumps(cur, indent=2, default=str), encoding="utf-8")

def add_caribbean_columns(df):
    """Add is_caribbean and region_caribbean_separate when they are missing.
    Script 06 rebuilds the analysis tables without these two columns, so they are derived again here
    from isolation_country, with the same rules as script 04."""
    if "is_caribbean" in df.columns and "region_caribbean_separate" in df.columns:
        return df
    c = df["isolation_country"].astype("string").str.strip()
    df["is_caribbean"] = c.isin(CARIBBEAN).astype(int)
    df["region_caribbean_separate"] = df["region"].where(df["is_caribbean"] == 0, "Caribbean")
    df.loc[df["region"] == NOT_RECORDED, "region_caribbean_separate"] = NOT_RECORDED
    return df

def load_table(full=False):
    df = pd.read_csv(PROCESSED / ("06_analysis_table_full.csv" if full else "06_analysis_table.csv"),
                     low_memory=False, dtype={"genome_id": str})
    return add_caribbean_columns(df)

def ordered_regions(present):
    return [r for r in REGION_ORDER if r in set(present)]

def compute_metrics(y, p, thr=0.5):
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
    y = np.asarray(y); p = np.asarray(p)
    out = {"n": int(len(y)), "prevalence": float(y.mean())}
    if len(set(y)) < 2:
        return out
    pred = (p >= thr).astype(int)
    out.update(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
               accuracy=float(accuracy_score(y, pred)), f1=float(f1_score(y, pred)))
    return out

def calibration_stats(y, p):
    """Calibration intercept and slope from a logistic recalibration on the logit of the predictions."""
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y); p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    if len(set(y)) < 2:
        return {"intercept": None, "slope": None, "brier": float(np.mean((p - y) ** 2))}
    z = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return {"intercept": float(m.intercept_[0]), "slope": float(m.coef_[0, 0]), "brier": float(np.mean((p - y) ** 2))}

def bootstrap_auc(y, p, groups=None, n_boot=2000, seed=RANDOM_SEED):
    """Percentile 95% CI. With groups, whole groups are resampled (cluster bootstrap)."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); p = np.asarray(p); rng = np.random.default_rng(seed); vals = []
    if groups is None:
        n = len(y)
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            if len(set(y[i])) == 2:
                vals.append(roc_auc_score(y[i], p[i]))
    else:
        g = pd.Series(groups).fillna("UNKNOWN").astype(str).values
        idx = {k: np.where(g == k)[0] for k in np.unique(g)}; keys = list(idx)
        for _ in range(n_boot):
            i = np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])
            if len(set(y[i])) == 2:
                vals.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def save_csv(df, name, **kw):
    df.to_csv(RESULTS / name, index=False, **kw)

if __name__ == "__main__":
    log = get_logger("00_setup")
    log.info(f"Root {ROOT}"); log.info(f"Versions {software_versions()}")
    log.info(f"{len(set(REGION_MAP.values()))} regions defined; seed {RANDOM_SEED}")

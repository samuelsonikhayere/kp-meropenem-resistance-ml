"""Shared figure style: white background, sans-serif, colour-blind-safe palette, panel letters, 300 dpi PNG + PDF."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import importlib; S = importlib.import_module("00_setup")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"lr": "#0072B2", "xgb": "#D55E00", "rule": "#555555", "grey": "#BBBBBB", "dark": "#222222", "green": "#009E73"}
REGION_COL = {"North America": "#0072B2", "Asia": "#E69F00", "Europe": "#009E73", "Oceania": "#CC79A7",
              "Africa": "#D55E00", "South America": "#56B4E9", "Not recorded": "#999999"}
MODEL_LABEL = {"logistic_regression": "Logistic regression", "xgboost": "XGBoost"}
MODEL_COL = {"logistic_regression": C["lr"], "xgboost": C["xgb"]}

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42})

def letter(ax, s, x=-0.14, y=1.05):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(S.FIGURES / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

def pca_scores():
    f = S.RESULTS / "pca_scores.csv"
    df = S.load_table()
    if f.exists():
        return df, pd.read_csv(f)
    X = df[S.feature_columns(df)].values.astype(float); Xc = X - X.mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False); var = s ** 2 / (s ** 2).sum()
    sc = pd.DataFrame({"PC1": U[:, 0] * s[0], "PC2": U[:, 1] * s[1]}); sc["var1"] = var[0]; sc["var2"] = var[1]
    sc.to_csv(f, index=False); return df, sc

def short(name, n=48):
    if "=>" in name:
        name = name.split("=>")[-1].strip()
        name = name.replace(", carbapenem-hydrolyzing", "").replace(", some variants carbapenem-hydrolyzing", "")
        return name
    name = name.split(" (EC")[0]
    return name if len(name) <= n else name[: n - 1] + "…"

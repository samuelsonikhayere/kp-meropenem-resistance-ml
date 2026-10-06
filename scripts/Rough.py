import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

cohort = pd.read_csv(
    Path("processed/04_with_regions.csv"),
    low_memory=False,
    dtype={"genome_id": str},
)

# THE KEY TEST: Can region alone predict resistance?
X = cohort[["region", "isolation_source", "laboratory_typing_method"]].copy()
y = cohort["label"]

pipe = Pipeline([
    ("onehot", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), slice(0, 3))
    ], remainder="drop")),
    ("clf", LogisticRegression(max_iter=5000, C=1.0)),
])

scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
print(f"AUROC predicting resistance from REGION + SOURCE + METHOD only: {scores.mean():.3f} ± {scores.std():.3f}")
print("Interpretation: if this is > 0.65, your confounding is severe.")
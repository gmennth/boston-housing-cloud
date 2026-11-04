import json, joblib, numpy as np, pandas as pd, sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]

df = pd.read_csv(ROOT / "data" / "boston.csv")

# --- Contrôle qualité des données (ajouté) ---
print("Dimensions :", df.shape)
print("Doublons :", df.duplicated().sum())
print("Valeurs manquantes par colonne :")
print(df.isna().sum())
df = df.drop_duplicates()
# ---------------------------------------------

X, y = df[FEATURES], df["MEDV"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

def make(model):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()), ("model", model)])

candidates = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
}

fitted, results = {}, {}
for name, mdl in candidates.items():
    pipe = make(mdl)
    cv_r2 = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="r2").mean()
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    fitted[name] = pipe
    results[name] = {"cv_r2": round(float(cv_r2), 4),
                     "r2": round(float(r2_score(y_te, pred)), 4),
                     "rmse": round(float(np.sqrt(mean_squared_error(y_te, pred))), 4),
                     "mae": round(float(mean_absolute_error(y_te, pred)), 4)}
    print(name, results[name])

best_name = max(results, key=lambda k: results[k]["r2"])
best = fitted[best_name]

imp = permutation_importance(best, X_te, y_te, n_repeats=10, random_state=42)
importance = dict(zip(FEATURES, [round(float(v), 4) for v in imp.importances_mean]))

(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(best, ROOT / "models" / "model.pkl")

metrics = {"best_model": best_name,
           "metrics": {k: results[best_name][k] for k in ("rmse", "mae", "r2")},
           "all_models": results, "importance": importance,
           "sklearn_version": sklearn.__version__}
(ROOT / "models" / "metrics.json").write_text(json.dumps(metrics, indent=2))

ranges = {c: {"min": float(X[c].min()), "max": float(X[c].max()), "median": float(X[c].median())}
          for c in FEATURES}
(ROOT / "models" / "feature_ranges.json").write_text(json.dumps(ranges, indent=2))

print("Meilleur modèle :", best_name, metrics["metrics"])
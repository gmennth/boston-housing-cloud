import json, joblib, numpy as np, pandas as pd, sklearn
import matplotlib
matplotlib.use("Agg")  # pas de fenêtre : on enregistre directement les images
import matplotlib.pyplot as plt
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

# Réglages du GradientBoosting : à remplacer par ceux de ml/tune.py si le gain est réel
GB_PARAMS = {"random_state": 42}

df = pd.read_csv(ROOT / "data" / "boston.csv")

# --- Contrôle qualité des données ---
print("Dimensions :", df.shape)
print("Doublons :", df.duplicated().sum())
print("Valeurs manquantes par colonne :")
print(df.isna().sum())
df = df.drop_duplicates()
# ------------------------------------

X, y = df[FEATURES], df["MEDV"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

def make(model):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()), ("model", model)])

candidates = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(**GB_PARAMS),
}

fitted, results = {}, {}
for name, mdl in candidates.items():
    pipe = make(mdl)
    scores = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="r2")  # 5 mini-examens sur l'apprentissage
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    fitted[name] = pipe
    results[name] = {"cv_r2": round(float(scores.mean()), 4),
                     "cv_std": round(float(scores.std()), 4),
                     "r2": round(float(r2_score(y_te, pred)), 4),
                     "rmse": round(float(np.sqrt(mean_squared_error(y_te, pred))), 4),
                     "mae": round(float(mean_absolute_error(y_te, pred)), 4)}
    print(name, results[name])

# Choix du meilleur modèle sur la validation croisée (le test reste réservé à la mesure finale)
best_name = max(results, key=lambda k: results[k]["cv_r2"])
best = fitted[best_name]
pred_best = best.predict(X_te)

imp = permutation_importance(best, X_te, y_te, n_repeats=10, random_state=42)
importance = dict(zip(FEATURES, [round(float(v), 4) for v in imp.importances_mean]))

# --- Sauvegardes (mêmes fichiers et mêmes clés que la v0) ---
(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(best, ROOT / "models" / "model.pkl")

metrics = {"best_model": best_name,
           "metrics": {k: results[best_name][k] for k in ("rmse", "mae", "r2")},
           "all_models": results, "importance": importance,
           "selection_criterion": "cv_r2",
           "sklearn_version": sklearn.__version__}
(ROOT / "models" / "metrics.json").write_text(json.dumps(metrics, indent=2))

ranges = {c: {"min": float(X[c].min()), "max": float(X[c].max()), "median": float(X[c].median())}
          for c in FEATURES}
(ROOT / "models" / "feature_ranges.json").write_text(json.dumps(ranges, indent=2))

# --- Graphiques pour l'oral (ml/figures/) ---
fig_dir = ROOT / "ml" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
rmse_b, r2_b = metrics["metrics"]["rmse"], metrics["metrics"]["r2"]

# 1. Prix réel contre prix prédit
plt.figure(figsize=(6, 6))
plt.scatter(y_te, pred_best, alpha=0.7)
lim = [min(y_te.min(), pred_best.min()) - 1, max(y_te.max(), pred_best.max()) + 1]
plt.plot(lim, lim, "r--", label="Prédiction parfaite")
plt.xlabel("Prix réel (milliers de dollars)"); plt.ylabel("Prix prédit (milliers de dollars)")
plt.title(f"{best_name} : R² = {r2_b}, RMSE = {rmse_b}"); plt.legend(); plt.grid(alpha=0.3)
plt.savefig(fig_dir / "prix_reel_vs_predit.png", dpi=150, bbox_inches="tight"); plt.close()

# 2. Résidus (erreurs) en fonction du prix prédit
plt.figure(figsize=(7, 4.5))
plt.scatter(pred_best, y_te - pred_best, alpha=0.7)
plt.axhline(0, color="r", linestyle="--")
plt.xlabel("Prix prédit (milliers de dollars)"); plt.ylabel("Erreur = réel - prédit")
plt.title("Résidus du meilleur modèle"); plt.grid(alpha=0.3)
plt.savefig(fig_dir / "residus.png", dpi=150, bbox_inches="tight"); plt.close()

# 3. Importance des variables
imp_s = pd.Series(importance).sort_values()
plt.figure(figsize=(7, 5))
imp_s.plot(kind="barh")
plt.xlabel("Baisse de R² quand la variable est mélangée"); plt.title("Importance des variables (permutation)")
plt.savefig(fig_dir / "importance_variables.png", dpi=150, bbox_inches="tight"); plt.close()

# 4. Comparaison des modèles (RMSE sur le test)
cmp_s = pd.Series({k: v["rmse"] for k, v in results.items()}).sort_values(ascending=False)
plt.figure(figsize=(7, 4))
cmp_s.plot(kind="barh")
plt.xlabel("RMSE sur le test (milliers de dollars, plus petit = mieux)"); plt.title("Comparaison des modèles")
plt.savefig(fig_dir / "comparaison_modeles.png", dpi=150, bbox_inches="tight"); plt.close()

print("Meilleur modele (choisi sur la validation croisee) :", best_name, metrics["metrics"])
print("Graphiques enregistres dans", fig_dir)

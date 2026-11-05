"""Réglage des hyperparamètres du GradientBoosting (GridSearchCV).

Ne modifie AUCUN fichier du modèle : il affiche seulement si le réglage améliore le résultat.
Si le gain est réel, copiez la ligne GB_PARAMS affichée dans ml/train.py puis relancez train.py.
"""
import json
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]

df = pd.read_csv(ROOT / "data" / "boston.csv").drop_duplicates()
X, y = df[FEATURES], df["MEDV"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)  # même séparation que train.py

def make(model):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()), ("model", model)])

def evaluer(pipe):
    pred = pipe.predict(X_te)
    return {"r2": round(float(r2_score(y_te, pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_te, pred))), 4),
            "mae": round(float(mean_absolute_error(y_te, pred)), 4)}

# Modèle de départ (réglages par défaut)
defaut = make(GradientBoostingRegressor(random_state=42)).fit(X_tr, y_tr)

# Grille : 3 x 2 x 3 x 2 = 36 combinaisons, chacune évaluée sur 5 plis (180 entraînements)
grille = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [2, 3, 4],
    "model__subsample": [0.8, 1.0],
}
recherche = GridSearchCV(make(GradientBoostingRegressor(random_state=42)), grille, cv=5,
                         scoring="neg_root_mean_squared_error", n_jobs=-1)
recherche.fit(X_tr, y_tr)  # uniquement sur l'apprentissage : le test n'est pas touché

res_defaut = evaluer(defaut)
res_regle = evaluer(recherche.best_estimator_)
params = {k.replace("model__", ""): v for k, v in recherche.best_params_.items()}

print("Meilleurs réglages :", params)
print("RMSE en validation croisée (réglé) :", round(-recherche.best_score_, 4))
print("Test, réglages par défaut :", res_defaut)
print("Test, réglages optimisés  :", res_regle)
gain = res_defaut["rmse"] - res_regle["rmse"]
print(f"Gain de RMSE sur le test : {gain:.4f} ({100 * gain / res_defaut['rmse']:.1f} %)")
if gain > 0.05:
    print("\nGain réel : dans ml/train.py, remplacez la ligne GB_PARAMS par :")
    print("GB_PARAMS =", {"random_state": 42, **params})
    print("puis relancez : python ml/train.py")
else:
    print("\nGain négligeable : gardez les réglages par défaut (modèle plus simple, presque aussi bon).")

(ROOT / "ml" / "tuning_results.json").write_text(json.dumps(
    {"best_params": params, "test_default": res_defaut, "test_tuned": res_regle}, indent=2))

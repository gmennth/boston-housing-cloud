from pathlib import Path
import json
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel 

# Chemin principal du projet
ROOT = Path(__file__).resolve().parent.parent

# Liste des 13 variables utilisées par le modèle
FEATURES = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX",
    "RM", "AGE", "DIS", "RAD", "TAX",
    "PTRATIO", "B", "LSTAT"
]

# Chemins des fichiers du modèle
MODEL_PATH = ROOT / "models" / "model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"
RANGES_PATH = ROOT / "models" / "feature_ranges.json"

# Charger le modèle de Machine Learning
model = joblib.load(MODEL_PATH)

# Charger les performances du modèle
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

# Charger les plages de valeurs des variables
feature_ranges = json.loads(RANGES_PATH.read_text(encoding="utf-8"))

# Création de l'application FastAPI
app = FastAPI(
    title="Boston Housing API",
    description="API de prédiction des prix des maisons",
    version="1.0.0"
)

# Vérifier si l'API fonctionne
@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "L'API fonctionne correctement"
    }

# Afficher les informations du modèle
@app.get("/model-info")
def model_info():
    return {
        "best_model": metrics["best_model"],
        "metrics": metrics["metrics"],
        "all_models": metrics["all_models"],
        "importance": metrics["importance"],
        "feature_ranges": feature_ranges
    }

# Structure des données reçues pour une prédiction
class HouseFeatures(BaseModel):
    CRIM: float
    ZN: float
    INDUS: float
    CHAS: float
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: float
    TAX: float
    PTRATIO: float
    B: float
    LSTAT: float

# Effectuer une prédiction du prix d'une maison
@app.post("/predict")
def predict(house: HouseFeatures):

    # Transformer les données reçues en DataFrame
    data = pd.DataFrame(
        [[getattr(house, feature) for feature in FEATURES]],
        columns=FEATURES
    )

    # Effectuer la prédiction avec le modèle
    prediction = model.predict(data)[0]

    # Retourner le résultat
    return {
        "predicted_price": round(float(prediction), 2),
        "unit": "thousands of dollars"
    }
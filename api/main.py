import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent.parent
FEATURES = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

# Chargement unique au démarrage : le pipeline (imputation + scaler + modèle) et les fichiers annexes
model = joblib.load(BASE / "models" / "model.pkl")
metrics = json.loads((BASE / "models" / "metrics.json").read_text(encoding="utf-8"))
ranges = json.loads((BASE / "models" / "feature_ranges.json").read_text(encoding="utf-8"))


class HouseFeatures(BaseModel):
    CRIM: float = Field(..., ge=0, description="Criminalité par habitant")
    ZN: float = Field(..., ge=0, le=100, description="% de grands terrains résidentiels")
    INDUS: float = Field(..., ge=0, le=100, description="% de surfaces non commerciales")
    CHAS: int = Field(..., ge=0, le=1, description="1 si borde la rivière Charles")
    NOX: float = Field(..., gt=0, description="Concentration d'oxydes d'azote")
    RM: float = Field(..., gt=0, description="Nombre moyen de pièces")
    AGE: float = Field(..., ge=0, le=100, description="% de logements anciens")
    DIS: float = Field(..., gt=0, description="Distance aux centres d'emploi")
    RAD: int = Field(..., ge=1, description="Accès aux autoroutes")
    TAX: float = Field(..., gt=0, description="Taux d'impôt foncier")
    PTRATIO: float = Field(..., gt=0, description="Ratio élèves / enseignants")
    B: float = Field(..., ge=0, description="Indicateur démographique")
    LSTAT: float = Field(..., ge=0, le=100, description="% de population à faible statut")


app = FastAPI(
    title="Boston House Price API",
    version="1.0.0",
    description="Prédit le prix médian d'un logement à Boston (en milliers de $).",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {**metrics, "feature_ranges": ranges}


@app.post("/predict")
def predict(features: HouseFeatures):
    # Les 13 colonnes, dans l'ordre du contrat d'interface
    X = pd.DataFrame([features.model_dump()])[FEATURES]
    y = float(model.predict(X)[0])
    rmse = metrics["metrics"]["rmse"]
    return {
        "prediction_k_usd": round(y, 2),
        "prediction_usd": int(round(y * 1000)),
        "range_k_usd": [round(y - rmse, 2), round(y + rmse, 2)],
        "model": metrics["best_model"],
    }

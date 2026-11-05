# Boston Housing - Prédiction du prix des logements

![CI](https://github.com/gmennth/boston-housing-cloud/actions/workflows/ci.yml/badge.svg)

Projet Cloud Computing : prédire le prix médian d'un logement à Boston
(régression), avec joblib, FastAPI, Streamlit, Docker et déploiement cloud.

## Équipe et branches

| Membre | Rôle | Branche |
|---|---|---|
| Nathan | Chef de projet, documentation, CI | nathan-docs-ci |
| SATH | Données et EDA | sath-data-eda |
| Jérémie | Machine Learning, joblib | jeremie-ml-model |
| Gernis | API FastAPI + tests | gernis-api-fastapi |
| Wilfried | Interface Streamlit | wilfried-streamlit-ui |
| Ben | Docker et déploiement | ben-docker-deploy |

## Architecture

    data/boston.csv -> ml/train.py (Jérémie) -> models/model.pkl (joblib)
                                                     |
                                                     v
                         api/main.py FastAPI + Uvicorn (Gernis) :8000
                                                     ^
                                                     | POST /predict (JSON)
                         ui/streamlit_app.py Streamlit (Wilfried) :8501

    Docker : 2 conteneurs + docker-compose (Ben) -> Cloud (Render + Streamlit Cloud)

## Lancer en local

    pip install -r requirements.txt
    python ml/train.py
    uvicorn api.main:app --reload --port 8000
    streamlit run ui/streamlit_app.py

API : http://localhost:8000/docs - Interface : http://localhost:8501

## Lancer avec Docker

    docker compose up --build

# Guide de déploiement - Boston Housing Cloud

## 1. Présentation

Ce projet permet de prédire le prix d'un logement à partir de ses caractéristiques grâce au Machine Learning.

L'application repose sur deux services :
- API : FastAPI et Uvicorn.
- Interface utilisateur : Streamlit.

Docker Compose permet de lancer les deux services ensemble.

## 2. Prérequis

Avant de démarrer, installer :
- Docker Desktop
- Git
- Docker Compose

Docker Desktop doit être démarré.

## 3. Architecture Docker

Le projet utilise deux conteneurs :

| Service | Technologie | Port |
|---|---|---|
| api | FastAPI / Uvicorn | 8000 |
| ui | Streamlit | 8501 |

L'interface Streamlit communique avec l'API via l'adresse interne :

http://api:8000

## 4. Construction et démarrage

Depuis la racine du projet, exécuter :

```bash
docker compose up --build
```

## 5. Accès à l'application

Après le démarrage des conteneurs :

- Interface Streamlit : http://localhost:8501
- Documentation de l'API : http://localhost:8000/docs
- Vérification de l'API : http://localhost:8000/health

## 6. Arrêter les services

Pour arrêter les conteneurs :

```bash
docker compose down
```

## 7. Reconstruire les services

Après une modification du code :

```bash
docker compose up --build
```

## 8. Déploiement Cloud

Le déploiement final prévoit :

- API FastAPI sur Render.
- Interface Streamlit sur Streamlit Community Cloud.

Les liens publics seront ajoutés après le déploiement effectif.

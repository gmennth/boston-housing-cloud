# Notes de modélisation (partie Machine Learning)

Auteur : Jérémie, branche `jeremie-ml-model`.

## Objectif
Prédire le prix médian d'un quartier de Boston (variable `MEDV`, en milliers de dollars) à partir de 13 caractéristiques. C'est un problème de **régression**.

## Données
- `data/boston.csv` : 506 quartiers, 14 colonnes (13 variables + la cible).
- Contrôle qualité : 0 doublon ; 20 valeurs manquantes dans chacune de CRIM, ZN, INDUS, CHAS, AGE, LSTAT (112 lignes concernées).
- Le prix est plafonné à 50 pour 16 quartiers (limite du jeu de données).

## Méthode (`ml/train.py`)
1. Séparation 80 % apprentissage (404) / 20 % test (102), `random_state=42`.
2. Pipeline : imputation par la **médiane** (apprise sur l'apprentissage seul, donc sans fuite de données), standardisation, modèle.
3. Quatre modèles comparés : LinearRegression, Ridge, RandomForest (300 arbres), GradientBoosting.
4. Validation croisée à 5 plis sur l'apprentissage, puis mesure finale sur le test (R², RMSE, MAE).
5. Le meilleur modèle est choisi sur le R² en **validation croisée** ; le test sert à la mesure finale.
6. Importance des variables par permutation, sur le test.
7. Sauvegarde du **pipeline complet** avec joblib dans `models/model.pkl`.

## Résultats (jeu de test)
| Modèle | R² valid. croisée | R² test | RMSE | MAE |
|---|---|---|---|---|
| LinearRegression | 0,7135 | 0,6593 | 4,9983 | 3,1476 |
| Ridge | 0,7136 | 0,6593 | 4,9987 | 3,1446 |
| RandomForest | 0,8063 | 0,8751 | 3,0262 | 2,0782 |
| **GradientBoosting** | **0,8279** | **0,8999** | **2,7087** | **1,9224** |

Modèle retenu : **GradientBoosting** (erreur typique d'environ 2 700 dollars, R² d'environ 0,90). Les modèles linéaires plafonnent à 0,66 : la relation entre les caractéristiques et le prix n'est pas linéaire.

## Variables les plus importantes
RM (0,4276), LSTAT (0,3714), DIS (0,1247), NOX (0,0432), PTRATIO (0,0346). INDUS, CHAS, RAD et ZN sont proches de zéro. L'importance montre une association utile pour prédire, pas une cause.

## Fichiers produits
- `models/model.pkl` : pipeline complet (valeurs brutes en entrée).
- `models/metrics.json` : résultats, importance, version de scikit-learn.
- `models/feature_ranges.json` : minimum, maximum et médiane de chaque variable.
- `ml/figures/` : graphiques (prix réel contre prix prédit, résidus, importance, comparaison des modèles).

## Environnement
Python 3.13.14, scikit-learn 1.9.1, numpy 2.5.4, scipy 1.18.1, pandas 3.1.0, joblib 1.6.0.
Un fichier `.pkl` dépend de ces versions : elles doivent être identiques dans l'API et dans Docker.

## Limites
Petit jeu de données, prix plafonné à 50, données anciennes de Boston, variable B controversée sur le plan éthique. Usage pédagogique uniquement : ne pas utiliser pour des décisions réelles.

## Réglage des hyperparamètres
`ml/tune.py` teste 36 combinaisons de réglages du GradientBoosting par validation croisée, sans toucher au test. Le résultat est comparé aux réglages par défaut ; on ne change `GB_PARAMS` dans `train.py` que si le gain est réel.

## Lancer
```
python ml/train.py
python ml/tune.py
```

# Prévision des ventes - Projet Superstore

## Objectif
Prédire les ventes journalières à partir de données historiques (2014-2017)

## Modèle final
- **Algorithme** : Random Forest
- **MAE** : 1 491 €
- **RMSE** : 2 294 €

## Features importantes
1. lag_7 (29%) : ventes d'il y a 7 jours
2. mois (26%) : saisonnalité annuelle
3. lag_14 (18%) : ventes d'il y a 14 jours

## Structure du projet

projet_ventes/
├── data/ # Données brutes
├── notebooks/ # Notebook Jupyter
├── models/ # Modèle sauvegardé
├── scripts/ # Fonctions Python
├── requirements.txt # Dépendances
└── README.md # Documentation


## Installation
```bash
pip install -r requirements.txt

import joblib
model = joblib.load('models/modele_ventes.pkl')
prediction = model.predict([[mois, jour_semaine, weekend, lag_1, lag_7, lag_14]])



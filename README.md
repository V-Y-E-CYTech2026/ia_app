
# S&P 500 Prediction - XGBoost & Kedro

Ce projet est une solution d'apprentissage automatique (Machine Learning) conçue pour prédire les mouvements de l'indice S&P 500 en utilisant l'algorithme XGBoost. Il repose sur le framework Kedro pour garantir une structure de code modulaire, reproductible et scalable.

## Auteurs
* Youenn Bogaer
* Victor ANDRE
* Ewen DANO

---

## Installation

Le projet utilise **uv** pour la gestion des dépendances et de l'environnement virtuel, ce qui permet une installation rapide et fiable.

Pour installer l'ensemble des dépendances en une seule commande, exécutez :

```bash
uv sync
```

---

## Architecture du projet

Le workflow est divisé en trois pipelines principaux :

1. **data_processing** : Récupération des données brutes, nettoyage, gestion des valeurs manquantes et ingénierie des indicateurs techniques.
2. **Training** : Entraînement du modèle XGBoost sur les données historiques.
3. **Test** : Évaluation du modèle sur des données de test et calcul des métriques de performance.

---

## Configuration et Paramètres

Le comportement du projet est entièrement pilotable via le fichier `conf/base/parameters.yml`. Vous pouvez y modifier :

* **L'horizon de prédiction** : Définir le nombre de jours à l'avance pour la prédiction.
* **Le choix des features** : Sélectionner spécifiquement les colonnes et indicateurs que vous souhaitez injecter dans le modèle.
* **Le type de modèle** : Régression ou classification

---

## Utilisation

Toutes les commandes doivent être lancées via l'outil Kedro.

**Lancer l'intégralité du pipeline :**
```bash
uv run kedro run
```

**Lancer un pipeline spécifique :**
```bash
uv run kedro run --pipeline=preprocessing
```

**Visualiser le pipeline de données :**
```bash
uv run kedro viz
```

---

## MLflow

Le projet intègre **MLflow** pour le suivi des expériences. Cela permet d'enregistrer automatiquement les paramètres, les métriques (comme l'erreur quadratique ou la précision) et les artefacts du modèle à chaque exécution.

Pour lancer l'interface de visualisation MLflow :
```bash
uv run mlflow ui
```

---


## Bonus : Application Interactive

Le projet inclut un fichier `app.py`. Il s'agit d'une interface simplifiée permettant de piloter l'entraînement du modèle directement, offrant une alternative plus visuelle ou directe à la ligne de commande Kedro.
On peut driectement choisir l'horzion et le type de modèle (régression ou classification)

Pourlancer l'application :
```bash
uv run streamlit app.py
```

Dans /doc/ on peut retrouver notre présentation et notre rapport sur ce projet.

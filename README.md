# Description de l'Architecture du Projet

Ce projet est une représentation simplifiée d'une architecture typique pour un projet de machine learning. Il comprend plusieurs répertoires et fichiers qui organisent les données, les modèles, les métriques et les artefacts produits lors de l'entraînement et du déploiement des modèles.

## Introduction

Ce projet vise à développer un modèle de prédiction de tags pour les questions posées par les utilisateurs sur StackOverflow. StackOverflow est une plateforme de questions-réponses très populaire pour les développeurs, et son utilisation est répandue dans la communauté des développeurs de logiciels.

## Objectif

L'objectif principal de ce projet est de fournir une solution automatisée qui recommande les tags appropriés pour les questions posées par les utilisateurs de StackOverflow. En utilisant des techniques d'apprentissage automatique (Machine Learning), le modèle apprendra à partir des questions précédemment posées et de leurs tags associés pour prédire les tags pertinents pour de nouvelles questions.

## Structure du Projet

- **`data/`** : Ce répertoire contient les données utilisées pour l'entraînement des modèles, y compris les fichiers CSV contenant les données brutes et prétraitées.

- **`mlartifacts/`** : Ce répertoire stocke les artefacts générés lors de l'entraînement des modèles de machine learning. Chaque sous-répertoire correspond à un modèle spécifique.

- **`mlruns/`** : Ce répertoire est lié à MLflow, un outil de suivi et de gestion des expériences de machine learning. Il stocke les résultats, les métriques, les paramètres et les artefacts de chaque exécution de modèle.

- **`models/`** : Ce répertoire contient les modèles entraînés qui ont été sauvegardés après l'entraînement.

- **`commits/`** : Ce répertoire stocke les captures d'écran des commits Git effectués dans le cadre du développement du projet.

- **`lda/`** : Ce répertoire contient les artefacts spécifiques à la modélisation de sujets (Latent Dirichlet Allocation) utilisée dans le projet.

- **`__pycache__/`** : Ce répertoire contient les fichiers Python compilés automatiquement.

- **`README.md`** : Ce fichier, contenant une description détaillée de l'architecture du projet.

## Contenu des Sous-Répertoires

- Chaque répertoire sous `mlartifacts/0/` correspond à un modèle de machine learning spécifique, et contient les artefacts nécessaires à ce modèle, tels que les fichiers de modèle, les dépendances, etc.

- Les répertoires sous `mlruns/0/` sont utilisés par MLflow pour stocker les résultats et les métriques des expériences de machine learning.

## Utilisation

- Les données sont prétraitées et organisées dans le répertoire `data/` avant d'être utilisées pour l'entraînement des modèles.

- Les modèles de machine learning sont entraînés en utilisant les données disponibles et les artefacts sont sauvegardés dans le répertoire correspondant sous `mlartifacts/`.

- Les résultats de l'entraînement et les métriques sont suivis à l'aide de MLflow, et les résultats sont stockés dans le répertoire `mlruns/`.
# TREND_APP

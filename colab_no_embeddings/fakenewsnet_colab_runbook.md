# Colab (FakeNewsNet) — un seul notebook autonome par tâche (sans recalcul des embeddings)

Ce guide utilise les notebooks fusionnés du dossier `colab_no_embeddings/`.
Chaque notebook contient maintenant le code complet de la tâche (sans exécuter d'autres notebooks).

## 1) Préparer Colab

```bash
!git clone https://github.com/<VOTRE_USER>/<VOTRE_REPO>.git
%cd <VOTRE_REPO>
!pip install -r requirements.txt
```

## 2) Placer les données/embeddings dans `FakeNewsNet/data/`

Les données prétraitées et embeddings doivent déjà être disponibles.

## 3) Exécuter **un notebook par tâche**

- Détection de désinformation :
  - `colab_no_embeddings/fakenewsnet_disinformation_detection.ipynb`
- Prédiction de viralité :
  - `colab_no_embeddings/fakenewsnet_virality_prediction.ipynb`

Chaque notebook exécute directement l'entraînement/évaluation des modèles de la tâche correspondante, en travaillant sur les embeddings déjà présents.

## 4) Important

- Ne pas exécuter les scripts de `FakeNewsNet/data_preprocessing/` si les embeddings sont déjà là.
- Si besoin, laissez `WANDB_DISABLED=true` pour éviter le blocage lié à la connexion Weights & Biases.


## 5) Résumé des résultats (publication)

- Voir `colab_no_embeddings/task_results_summary.md` pour un résumé prêt à réutiliser dans un document de publication.


## 6) Fichiers de synthèse pour publication

Chaque notebook génère aussi dans `colab_no_embeddings/outputs/`:
- un CSV détaillé (folds CV)
- un CSV de synthèse (moyennes par modèle)
- un **Markdown de synthèse publication** (classement + top modèles)

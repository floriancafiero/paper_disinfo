# Colab (FakeNewsNet) — un seul notebook par tâche (sans recalcul des embeddings)

Ce guide utilise les notebooks fusionnés du dossier `colab_no_embeddings/`.

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

Chaque notebook lance automatiquement les notebooks FakeNewsNet de la tâche correspondante, en travaillant sur les embeddings déjà présents.

## 4) Important

- Ne pas exécuter les scripts de `FakeNewsNet/data_preprocessing/` si les embeddings sont déjà là.
- Si besoin, laissez `WANDB_DISABLED=true` pour éviter le blocage lié à la connexion Weights & Biases.


## 5) Résumé des résultats (publication)

- Voir `colab_no_embeddings/task_results_summary.md` pour un résumé prêt à réutiliser dans un document de publication.

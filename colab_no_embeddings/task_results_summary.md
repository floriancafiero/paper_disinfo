# Résumé des résultats par tâche (prêt pour publication)

> Source des chiffres: mémoire `report/memoire.tex` (résultats comparatifs RoBERTa/Mistral et conclusions globales).

## 1) Evons — Détection de la désinformation

- Les performances sont **quasi parfaites** sur cette tâche.
- Exemple représentatif (MLP): **F1 = 0.990** avec embeddings RoBERTa, **0.992** avec embeddings Mistral.
- Interprétation: tâche bien résolue sur ce dataset, mais vigilance sur le risque de *source confound* (modèle pouvant apprendre des indices de source).

## 2) Evons — Prédiction de viralité

- Les performances sont **nettement plus faibles** que pour la détection de désinformation.
- Exemple représentatif (gating model): **F1 = 0.323** (RoBERTa), **0.337** (Mistral).
- Interprétation: résultats encourageants mais insuffisants; la viralité dépend de facteurs externes au texte et reste difficile à anticiper.

## 3) FakeNewsNet — Détection de la désinformation

- Les performances sont **élevées et robustes** malgré un jeu de données plus petit et déséquilibré.
- Exemple représentatif (GRU): **F1 = 0.891** (RoBERTa), **0.906** (Mistral), avec amélioration également en ROC-AUC (0.961 → 0.972).
- Interprétation: la tâche est bien maîtrisée sur FakeNewsNet avec les architectures séquentielles testées.

## 4) FakeNewsNet — Prédiction de viralité

- Les performances sont **intermédiaires**: meilleures que sur Evons virality, mais en retrait par rapport à la détection de désinformation.
- Exemple représentatif (GRU): **F1 = 0.793** (RoBERTa), **0.773** (Mistral).
- Interprétation: la tâche reste difficile; la définition opérationnelle de la viralité (seuil médian des likes) constitue une limite méthodologique importante.

---

## Message de synthèse (publication)

Globalement, les expériences montrent que la **détection de la désinformation** est la tâche la plus performante sur les deux datasets, tandis que la **prédiction de viralité** demeure plus complexe et moins stable. Le changement de modèle d'embeddings (RoBERTa vs Mistral) modifie peu la hiérarchie générale des résultats: les écarts de F1 restent faibles et n'altèrent pas les conclusions principales.

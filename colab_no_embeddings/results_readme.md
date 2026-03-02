# Résultats et synthèses (Colab no embeddings)

Après exécution d'un notebook de `colab_no_embeddings/`, les artefacts sont écrits dans `colab_no_embeddings/outputs/`.

Pour chaque tâche, vous obtenez :

1. `*_merged_cv_results.csv` : résultats détaillés par fold.
2. `*_merged_summary.csv` : moyenne des métriques par modèle.
3. `*_publication_summary.md` : synthèse prête à intégrer dans un draft de publication.

Ces fichiers servent à accélérer la rédaction et la comparaison des variantes de modèles.

## Robustesse statistique

Les notebooks utilisent désormais par défaut `RepeatedStratifiedKFold` (10 x 3 = 30 splits) pour augmenter la stabilité des moyennes CV et la puissance des comparaisons pairwise.
Vous pouvez ajuster `CV_N_SPLITS` et `CV_N_REPEATS` dans chaque notebook selon vos contraintes de calcul.

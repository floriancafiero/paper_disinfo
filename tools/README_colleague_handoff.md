# Instructions pour le collègue — révision PLOS ONE

L'objectif est de **recalculer les résultats, sans transmettre les données
EVONS ni les tweets FakeNewsNet et sans recalculer les embeddings**.

## 1. Préparer l'environnement

Cloner **la branche de révision** (et non `main`) :

```bash
git clone --branch plos-one-major-revision --single-branch \
  https://github.com/floriancafiero/paper_disinfo.git
cd paper_disinfo

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install scipy nbclient nbformat ipykernel tabulate
```

Sous Windows PowerShell, remplacer `source .venv/bin/activate`
par `.venv\Scripts\Activate.ps1`.

**Matériel** : mémoire suffisante pour charger de grands tenseurs ;
GPU NVIDIA conseillé pour les entraînements, non nécessaire pour
l'audit. Les quatre quantiles exécutent plusieurs centaines de
fittings, donc prévoir éventuellement plusieurs heures.

## 2. Déposer les fichiers locaux (ne pas les envoyer à GitHub)

Les chemins exacts à respecter :

```text
evons/data/
    evons.csv
    title_embeddings.pt
    desc_embeddings.pt
    title_embeddings_mistral.pt
    desc_embeddings_mistral.pt

FakeNewsNet/data/
    ordered_fake_propagation_paths.jsonl
    ordered_real_propagation_paths.jsonl
    fake_news_net/
        ordered_fake_propagation_paths_emb.pt
        ordered_real_propagation_paths_emb.pt
        ordered_fake_propagation_paths_emb_mistral.pt
        ordered_real_propagation_paths_emb_mistral.pt
```

Les listes `ordered_*` doivent correspondre **dans le même ordre**
aux embeddings de la même catégorie ; une simple égalité de nombres
de lignes n'assure pas l'alignement. Si certains fichiers ont d'autres
noms, déplacer/copier localement plutôt que de modifier le dépôt.

Les ressources de référence dans les README :
- EVONS CSV : https://github.com/krstovski/evons
- EVONS embeddings :
  https://drive.google.com/drive/folders/1X27WjPEKzAcC5jXai8cuI8FEmWkjJb6l
- FakeNewsNet PolitiFact :
  https://drive.google.com/drive/folders/16UfhZw9lDhTjWKLL7kXCm9U4YU0uURV6
  (accès autorisé requis).

## 3. Contrôle préalable (sans entraînement)

```bash
python tools/colleague_revision_run.py --stage check
```

La console doit afficher `OK` pour **tous** les chemins, sans
`MISSING`. Ce contrôle vérifie les fichiers, mais pas l'intégrité
des embeddings ni leur alignement exact avec les JSONL/CSV.

## 4. Audits rapides — commencer par ici

```bash
python tools/colleague_revision_run.py --stage audit
```

Produit, dans `revision_results/`, l'inventaire EVONS/PolitiFact,
les quantiles et prévalences, les premières mesures d'engagement,
et pour les sous-ensembles fake/real des mesures « engagement futur »
contre « engagement total », sur des cascades comparables.
Ces scripts utilisent seulement CPU et librairie standard Python.

L'archive `revision_results/SHARE_AGGREGATES_ONLY.zip` peut être
transmise au coauteur pour reconstruire les tableaux.

## 5. Entraînement complet — GPU recommandé

Seulement après validation des audits, lancer :

```bash
python tools/colleague_revision_run.py --stage train
```

Ce mode recalcule les deux tâches de détection et la grille complète
q50, q75, q90, q95 des tâches de viralité. Il ne recalcule pas les
embeddings et n'appellera pas l'API Mistral.

Les scores sortent dans `colab_no_embeddings/outputs/`. Le lanceur
crée une archive partageable des **CSV agrégés**. Prévoir
plusieurs heures selon le GPU, la taille des données et l'activation
de XGBoost. Les calculs relancés repartent du début ; pas de reprise
automatique fine au milieu d'une expérience.

## 6. Ce qu'il faut nous renvoyer

**Uniquement :** `revision_results/SHARE_AGGREGATES_ONLY.zip`,
ainsi que les informations sur le GPU, la version de Python, les
versions des principales bibliothèques et les erreurs éventuelles.

Ne pas transmettre : les CSV d'articles bruts, les JSONL de tweets,
les fichiers `.pt`, ni les notebooks Jupyter exécutés (leurs
cellules pourraient contenir des contenus sensibles). **Ne rien
pousser vers GitHub**, y compris les résultats, sans inspection.

Pour une exécution interrompue, envoyer `revision_results/RUN_FAILED.txt`
*après* avoir vérifié qu'aucun chemin d'accès ou contenu sensible
n'y figure.

## Points à vérifier humainement

- Vérifier l'alignement exact des fichiers de embeddings et des
  séquences. Un filtrage différent de FakeNewsNet entre versions
  rendrait les résultats incohérents.
- Vérifier les classes (notamment si q95 laisse trop peu d'exemples
  positifs pour les splits). Si un entraînement échoue faute
  d'exemples, conserver le message : ne pas réduire silencieusement
  le quantile.
- Certains fichiers embeddings Mistral peuvent être des listes de
  vecteurs plutôt que des tensors : le collègue doit conserver le
  format attendu par les notebooks. Ne pas convertir arbitrairement
  sans documenter la transformation.
- La vérification du protocole 60/20/20 reste nécessaire au-delà
  des simples tests de syntaxe et d'audit.

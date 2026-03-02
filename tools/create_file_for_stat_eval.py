import csv, pathlib

files = [
    ("evons_disinformation_merged_cv_results.csv",   "evons",        "disinformation"),
    ("evons_virality_merged_cv_results.csv",          "evons",        "virality"),
    ("fakenewsnet_disinformation_merged_cv_results.csv", "fakenewsnet", "disinformation"),
    ("fakenewsnet_virality_merged_cv_results.csv",    "fakenewsnet",  "virality"),
]
metrics = ["accuracy", "f1", "precision", "recall", "auc"]
out_rows = []
for fname, dataset, task in files:
    path = pathlib.Path("colab_no_embeddings/outputs") / fname
    for row in csv.DictReader(path.open()):
        for m in metrics:
            out_rows.append({
                "dataset": dataset, "task": task, "metric": m,
                "fold": row["fold"], "model": row["model"], "value": row[m]
            })

with open("combined_cv.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dataset","task","metric","fold","model","value"])
    w.writeheader(); w.writerows(out_rows)
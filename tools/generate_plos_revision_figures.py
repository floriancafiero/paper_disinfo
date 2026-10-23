#!/usr/bin/env python3
"""Generate submission-oriented figures and aggregate tables from REVISED CSVs.

Does not read or export any raw article/tweet text or individual predictions.
Outputs go to report/revised_figures. Requires matplotlib only.
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUANTILES = (.5, .75, .9, .95)
WINDOWS = ("k1", "k3", "k5", "k10", "full")


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def future_figure(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, constrained_layout=True)
    for ax, (name, n) in zip(axs, (("real", 291), ("fake", 143))):
        rows = read_rows(ROOT / f"revision_results/future_{name}.csv")
        for q in QUANTILES:
            rs = sorted((r for r in rows if float(r["quantile"]) == q),
                        key=lambda r: int(r["k_prefix_tweets"]))
            assert len(rs) == 4
            ax.plot([int(r["k_prefix_tweets"]) for r in rs],
                    [float(r["auc_prefix_for_future_label"]) for r in rs],
                    marker="o", label=f"q{round(q * 100)}")
        ax.axhline(.5, color="grey", alpha=.5, linestyle=":", linewidth=1)
        ax.set(xlabel="First k recovered tweets", title=f"{name.capitalize()}-labelled stories (n={n})",
               xticks=[1, 3, 5, 10], ylim=(.45, 1.01))
        ax.grid(axis="y", alpha=.15)
        ax.legend(title="Future-label quantile", loc="upper left", fontsize=8)
    axs[0].set_ylabel("ROC-AUC: prefix favorites vs later-tweet favorites")
    fig.suptitle("Retrospective prospective-target audit on fixed cohorts", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def total_vs_future_figure(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, constrained_layout=True)
    for ax, name in zip(axs, ("real", "fake")):
        rows = read_rows(ROOT / f"revision_results/future_{name}.csv")
        for q in (.5, .95):
            rs = sorted((r for r in rows if float(r["quantile"]) == q),
                        key=lambda r: int(r["k_prefix_tweets"]))
            x = [int(r["k_prefix_tweets"]) for r in rs]
            ax.plot(x, [float(r["auc_prefix_for_future_label"]) for r in rs],
                    marker="o", label=f"q{round(q*100)} future-only")
            ax.plot(x, [float(r["auc_prefix_for_total_label_matched_cohort"]) for r in rs],
                    marker="x", linestyle="--", label=f"q{round(q*100)} total")
        ax.axhline(.5, color="grey", alpha=.5, linestyle=":", linewidth=1)
        ax.set(xlabel="First k recovered tweets", title=f"{name.capitalize()}-labelled",
               xticks=[1,3,5,10], ylim=(.45,1.01))
        ax.grid(axis="y", alpha=.15)
        ax.legend(fontsize=8)
    axs[0].set_ylabel("Rank ROC-AUC")
    fig.suptitle("Matched story cohort: future-only vs total engagement label", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def latex_escape(s):
    return str(s).replace("_", r"\_")


def summary_tables(output):
    base = ROOT / "colab_no_embeddings/outputs"
    lines = [
        "% Revised 60/20/20 experiment; average metrics over 10 OVERLAPPING stratified holdouts.",
        "% Best model is chosen post-hoc by mean calibrated F1 at each quantile/window.",
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\caption{EVONS virality: best classifier at each engagement quantile. AP is average precision.}",
        r"\label{tab:evons-revised}",
        r"\begin{tabular}{rlrrrr}", r"\toprule",
        r"Percentile & Model & F1 (cal.) & AP & ROC-AUC & Positive rate \\", r"\midrule",
    ]
    for q in (50,75,90,95):
        rows = read_rows(base / f"evons_virality_q{q}/evons_virality_merged_summary.csv")
        best = max(rows, key=lambda r: float(r["f1_tuned"]))
        lines.append(f'{q} & {latex_escape(best["model"])} & {float(best["f1_tuned"]):.3f} & '
                     f'{float(best["pr_auc"]):.3f} & {float(best["auc"]):.3f} & '
                     f'{float(best["pr_baseline"]):.3f} '+r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\caption{FakeNewsNet--PolitiFact: best calibrated F1 by textual observation window on a fixed 434-story cohort.}",
        r"\label{tab:fnn-revised}", r"\begin{tabular}{rrlrrrr}", r"\toprule",
        r"Percentile & Window & Best model & F1 (cal.) & AP & ROC-AUC & Positives \\", r"\midrule",
    ])
    for q in (50,75,90,95):
        rows = read_rows(base / f"fakenewsnet_virality_prefix_q{q}/fakenewsnet_virality_merged_summary.csv")
        for w in ("k1", "k3", "k5", "k10", "full"):
            best = max((r for r in rows if r["observation_window"] == w), key=lambda r: float(r["f1_tuned"]))
            lines.append(f'{q} & {w} & {latex_escape(best["model"])} & {float(best["f1_tuned"]):.3f} & '
                         f'{float(best["pr_auc"]):.3f} & {float(best["auc"]):.3f} & '
                         f'{best["n_positive_cohort"]} '+r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
        "% Caution: source-associated EVONS veracity labels, FNN fixed-length selection,",
        "% observational labels and small n at q95. Do not interpret model ranking as significance."])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=ROOT / "report/revised_figures")
    args = p.parse_args()
    future_figure(args.output_dir / "fig2_future_only.png")
    total_vs_future_figure(args.output_dir / "figS_total_vs_future.png")
    summary_tables(args.output_dir / "tables_revision.tex")
    print("Wrote two figures (PNG and PDF) and revised LaTeX tables:", args.output_dir)


if __name__ == "__main__":
    main()

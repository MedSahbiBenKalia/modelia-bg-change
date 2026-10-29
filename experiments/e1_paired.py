"""E1: compare two models image by image (per-image metrics written by e1_models.ipynb).

Reads results/e1/e1_metrics.csv and writes results/e1/e1_paired.md. For each group and metric:
the mean per-image difference (first model - second model), on how many images the first model is
better, and the p-value of a two-sided Wilcoxon signed-rank test. The test is paired (same images
for both models) and does not assume normal differences (a few failures skew them).

    python experiments/e1_paired.py
"""
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

RESULTS = Path(__file__).resolve().parents[1] / "results" / "e1"
PAIRS = [("BiRefNet", "InSPyReNet"), ("BiRefNet", "BiRefNet_lite")]
GROUPS = {"A_people": ["A_people"], "B_products": ["B_products"],
          "A+B": ["A_people", "B_products"], "C_structure": ["C_structure"]}
HIGHER_IS_BETTER = {"iou": True, "biou": True, "mae": False}


def compare(metrics, first, second, groups, metric):
    wide = metrics[metrics.group.isin(groups)].pivot(index="name", columns="model", values=metric)
    d = wide[first] - wide[second]
    wins = (d > 0).sum() if HIGHER_IS_BETTER[metric] else (d < 0).sum()
    p = wilcoxon(d).pvalue
    return f"{d.mean():+.4f} | {wins}/{len(d)} | {'<0.001' if p < 0.001 else f'{p:.3f}'}"


def main():
    metrics = pd.read_csv(RESULTS / "e1_metrics.csv")
    lines = ["# E1: paired comparison", "",
             "Same images for both models: mean per-image difference, number of images where the first model "
             "is better, two-sided Wilcoxon signed-rank p-value. Made by `experiments/e1_paired.py`.", ""]
    for first, second in PAIRS:
        lines += [f"## {first} vs {second}", "",
                  f"| group | metric | mean diff ({first} - {second}) | {first} better on | p |",
                  "|---|---|---:|---:|---:|"]
        lines += [f"| {g} | {metric} | {compare(metrics, first, second, groups, metric)} |"
                  for g, groups in GROUPS.items() for metric in HIGHER_IS_BETTER]
        lines.append("")
    (RESULTS / "e1_paired.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Curve plots from the evaluation CSVs (Fig. 4, and the ROAD equivalent).

Reads what ``eval_verification.py`` and ``eval_classification.py`` wrote and
draws the accuracy-over-removal-ratio curves.  Nothing is recomputed, so the
curves and the AUC numbers in the tables come from the same run.

    verification   Fig. 4: accuracy against the removal ratio P, one line per
                   XAI method, separately for deletion and insertion.  A
                   faithful map drops fastest under deletion and rises fastest
                   under insertion.
    classification the ROAD-Delete equivalent: mean logit drop against P.

Example:
    python scripts/plot_curves.py --task verification \
        --metrics results/CFP_metrics.csv --output results/figures
"""

import argparse
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # No display on a compute node.
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)

import _bootstrap  # noqa: E402

_bootstrap.ensure_repository_on_path()  # must precede the fame imports

from fame.metrics import PERCENTAGES  # noqa: E402
from fame.runlog import RunLogger  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["verification", "classification"])
    parser.add_argument("--metrics", required=True, help="CSV written by the eval script")
    parser.add_argument("--output", required=True)
    parser.add_argument("--clean-accuracy", type=float, default=None,
                        help="draw a reference line, e.g. 99.86 for CFP-FP with IResNet101")
    return parser.parse_args()


def verification_curves(table, args, log):
    """One figure per protocol, with deletion and insertion side by side."""
    for protocol, rows in table.groupby("protocol"):
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

        for axis, direction in zip(axes, ("Delete", "Insert")):
            subset = rows[rows["metric"] == direction]
            for _, row in subset.iterrows():
                accuracies = [100 * float(v) for v in str(row["accuracies"]).split(";")]
                percentages = list(PERCENTAGES)[: len(accuracies)]
                axis.plot(
                    percentages,
                    accuracies,
                    marker="o",
                    markersize=3,
                    label=f"{row['method']} ({row['auc']:.1f})",
                )

            if args.clean_accuracy is not None:
                axis.axhline(args.clean_accuracy, linestyle="--", linewidth=0.8, color="grey")
            axis.set_title(direction)
            axis.set_xlabel("removed pixels P (%)" if direction == "Delete" else "kept pixels P (%)")
            axis.grid(alpha=0.3)
            axis.legend(fontsize=8, title="method (AUC)", title_fontsize=8)

        axes[0].set_ylabel("verification accuracy (%)")
        figure.suptitle(f"{rows['dataset'].iloc[0]} {protocol}, {rows['model'].iloc[0]}")
        figure.tight_layout()

        path = os.path.join(args.output, f"curves_{rows['dataset'].iloc[0]}_{protocol}.pdf")
        figure.savefig(path)
        plt.close(figure)
        log.info(f"wrote {path}")


def classification_curves(table, args, log):
    """One figure per network: mean logit drop against the removal ratio."""
    road = table[table["metric"] == "ROAD-Delete"]
    if road.empty:
        raise SystemExit("no ROAD-Delete rows in the metrics CSV")

    for model_name, rows in road.groupby("model"):
        figure, axis = plt.subplots(figsize=(5.5, 4.2))
        for method, method_rows in rows.groupby("method"):
            ordered = method_rows.sort_values("parameter")
            axis.plot(
                ordered["parameter"], ordered["value"], marker="o", markersize=3, label=method
            )

        axis.set_xlabel("removed pixels P (%)")
        axis.set_ylabel("mean logit drop")
        axis.set_title(f"ROAD-Delete, {model_name}")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        figure.tight_layout()

        path = os.path.join(args.output, f"road_{model_name}.pdf")
        figure.savefig(path)
        plt.close(figure)
        log.info(f"wrote {path}")


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    log = RunLogger(args.output, f"curves_{args.task}", args)

    table = pd.read_csv(args.metrics)
    with log.stage(f"{args.task} curves"):
        if args.task == "verification":
            verification_curves(table, args, log)
        else:
            classification_curves(table, args, log)
    log.finish()


if __name__ == "__main__":
    main()

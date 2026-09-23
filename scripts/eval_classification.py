"""Evaluate classification attributions with IoU and ROAD-Delete (Tab. 1).

Merges ``eval_iou.py`` and ``eval_road_delete.py``.  Both metrics read the same
saved ``.npy`` maps and both loop over the same (method, model) grid, so they
share one pass over the data and write one long-format CSV.

Example:
    python scripts/eval_classification.py \
        --images-root /data/ILSVRC/Data/CLS-LOC/val \
        --synset-mapping /data/ILSVRC/LOC_synset_mapping.txt \
        --attributions results --output results/classification_metrics.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import _bootstrap

_bootstrap.ensure_repository_on_path()  # must precede the fame imports

from fame.data import ImageNetSubset
from fame.metrics import (
    intersection_over_union,
    logit_drop,
    noisy_linear_imputation,
    top_percent_mask,
)
from fame.models import CLASSIFIERS, build_classifier
from fame.runlog import RunLogger


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--synset-mapping", required=True)
    parser.add_argument("--protocol", default="protocols/imagenet.csv")
    parser.add_argument("--attributions", required=True, help="root holding <method>/<model>/*.npy")
    parser.add_argument("--output", required=True, help="destination CSV")
    parser.add_argument("--models", nargs="+", default=list(CLASSIFIERS))
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["GradCAM", "GradCAMElementWise", "HiResCAM", "FullGrad", "FAME"],
    )
    parser.add_argument("--iou-thresholds", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument(
        "--original-boxes",
        action="store_true",
        help="scale bounding box axes independently, as adjust_bbx does",
    )
    parser.add_argument(
        "--road-percentages",
        nargs="+",
        type=float,
        default=list(np.linspace(0, 100, 11)),
        help="removal ratios; the original sweeps 0 to 100 in eleven steps",
    )
    parser.add_argument(
        "--road-imputation",
        default="zero",
        choices=["zero", "linear"],
        help=(
            "'zero' blacks pixels out before normalization, which is what "
            "eval_road_delete.py does; 'linear' uses the neighbourhood "
            "imputation of Rong et al., which stops the classifier from reading "
            "the mask shape itself as a feature"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_attribution(root, method, model_name, name):
    path = os.path.join(root, method, model_name, f"{name}.npy")
    attribution = np.load(path)
    return np.squeeze(attribution)


def main():
    args = parse_args()
    log = RunLogger(args.output, "eval_classification", args, args.device)
    dataset = ImageNetSubset(
        args.protocol,
        args.images_root,
        args.synset_mapping,
        limit=args.limit,
        aspect_preserving_boxes=not args.original_boxes,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    rows = []
    for model_name in args.models:
        model = build_classifier(model_name, device=args.device)

        for method in args.methods:
            ious = {threshold: [] for threshold in args.iou_thresholds}
            drops = {percentage: [] for percentage in args.road_percentages}

            with log.stage(f"{method}/{model_name}"):
                for batch in tqdm(loader, desc=f"{method}/{model_name}"):
                    names = [
                        f"{image_id}_{int(target)}"
                        for image_id, target in zip(batch["image_id"], batch["target"])
                    ]
                    attributions = np.stack(
                        [load_attribution(args.attributions, method, model_name, n) for n in names]
                    )

                    for threshold in args.iou_thresholds:
                        for attribution, box in zip(attributions, batch["box"].numpy()):
                            ious[threshold].append(intersection_over_union(attribution, box, threshold))

                    images = batch["image"].to(args.device)
                    targets = batch["target"].to(args.device)
                    for percentage in args.road_percentages:
                        masks = np.stack(
                            [top_percent_mask(a, percentage) for a in attributions]
                        )
                        mask = torch.from_numpy(masks).to(args.device).float()[:, None]
                        if args.road_imputation == "linear":
                            perturbed = noisy_linear_imputation(images, mask)
                        else:
                            perturbed = images * mask
                        drops[percentage].extend(
                            logit_drop(model, images, perturbed, targets).cpu().tolist()
                        )

            for threshold, values in ious.items():
                rows.append(
                    {
                        "model": model_name,
                        "method": method,
                        "metric": "IoU",
                        "parameter": threshold,
                        "value": 100.0 * float(np.mean(values)),
                    }
                )
            for percentage, values in drops.items():
                rows.append(
                    {
                        "model": model_name,
                        "method": method,
                        "metric": "ROAD-Delete",
                        "parameter": percentage,
                        "value": float(np.mean(values)),
                    }
                )

        del model
        torch.cuda.empty_cache()

    table = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    table.to_csv(args.output, index=False, float_format="%.4f")
    log.info(f"wrote {args.output}")
    log.info(
        "\n"
        + table.pivot_table(index=["method"], columns=["metric", "parameter"], values="value")
        .to_string()
    )
    log.finish()


if __name__ == "__main__":
    main()

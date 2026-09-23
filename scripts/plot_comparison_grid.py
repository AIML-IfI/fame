"""Comparison figures (supplemental Sec. C, Fig. 9 to 12).

Reads attribution maps that the run scripts already wrote and lays them out as
the grids used in the paper:

    classification   Fig. 9.  Rows are networks, columns are XAI methods, one
                     sheet per image.
    verification     Fig. 10 to 12.  Rows are networks, columns are methods,
                     with e_+ and e_- shown side by side where a method
                     produces both.  Grad-CAM variants have no dissimilar map,
                     so those cells are left blank rather than filled with a
                     duplicate.

Nothing is recomputed here, which is the point: the figures and the numbers in
Tab. 1 and Tab. 2 come from the same files on disk.

Example:
    python scripts/plot_comparison_grid.py --task verification \
        --crops-root $CROPS --protocol-dir protocols/CFP --dataset CFP \
        --protocol 01FP --attributions results --pairs 0 1 2 \
        --output results/figures
"""

import argparse
import os

import numpy as np
from tqdm import tqdm

import _bootstrap

_bootstrap.ensure_repository_on_path()  # must precede the fame imports

from fame.data import ImageNetSubset, load_crop
from fame.runlog import RunLogger
from fame.visualization import bgr_to_rgb, contact_sheet, overlay

CLASSIFICATION_METHODS = ["GradCAM", "FullGrad", "HiResCAM", "FAME"]
VERIFICATION_METHODS = ["GradCAM", "GradCAMElementWise", "CorrRISE", "FGGB", "FAME"]
# Methods that produce a dissimilar map as well as a similar one.
TWO_SIDED = {"CorrRISE", "FGGB", "FAME"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["classification", "verification"])
    parser.add_argument("--attributions", required=True, help="root written by the run scripts")
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--alpha", type=float, default=0.5, help="heatmap weight in the overlay")

    # Classification
    parser.add_argument("--images-root", default=None)
    parser.add_argument("--synset-mapping", default=None)
    parser.add_argument("--protocol", default="protocols/imagenet.csv")
    parser.add_argument("--indices", nargs="+", type=int, default=[0], help="rows of the protocol")

    # Verification
    parser.add_argument("--crops-root", default=None)
    parser.add_argument("--protocol-dir", default=None)
    parser.add_argument("--dataset", default="CFP")
    parser.add_argument("--protocol-name", default="01FP")
    parser.add_argument("--pairs", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--side", default="p", choices=["g", "p"], help="explain the gallery or the probe"
    )
    return parser.parse_args()


def load_map(path):
    """Load a saved map, or return None when the method did not produce one."""
    if not os.path.exists(path):
        return None
    return np.squeeze(np.load(path))


def classification_figures(args, log):
    if not (args.images_root and args.synset_mapping):
        raise SystemExit("classification figures need --images-root and --synset-mapping")

    models = args.models or ["ResNet34", "VGG19", "ConvNeXt_Tiny"]
    methods = args.methods or CLASSIFICATION_METHODS
    dataset = ImageNetSubset(args.protocol, args.images_root, args.synset_mapping)

    for index in tqdm(args.indices, desc="images"):
        item = dataset[index]
        name = dataset.output_name(index)
        picture = item["image"].numpy().transpose(1, 2, 0)

        rows = []
        for model_name in models:
            panels = [np.uint8(255 * picture)]
            for method in methods:
                attribution = load_map(
                    os.path.join(args.attributions, method, model_name, f"{name}.npy")
                )
                panels.append(
                    None if attribution is None else overlay(picture, attribution, args.alpha)
                )
            rows.append(panels)

        figure = contact_sheet(
            rows,
            row_labels=models,
            column_labels=["input"] + methods,
            title=f"{item['image_id']} (class {int(item['target'])})",
        )
        os.makedirs(args.output, exist_ok=True)
        path = os.path.join(args.output, f"comparison_{name}.pdf")
        figure.savefig(path)
        figure.clf()
        log.info(f"wrote {path}")


def verification_figures(args, log):
    if not (args.crops_root and args.protocol_dir):
        raise SystemExit("verification figures need --crops-root and --protocol-dir")

    import pandas as pd

    models = args.models or ["Adaface_ir_18", "Adaface_ir_50", "Adaface_ir_101"]
    methods = args.methods or VERIFICATION_METHODS
    table = pd.read_csv(os.path.join(args.protocol_dir, f"{args.protocol_name}.csv"))

    columns = ["input"]
    for method in methods:
        columns.append(f"{method} e+")
        if method in TWO_SIDED:
            columns.append(f"{method} e-")

    for index in tqdm(args.pairs, desc="pairs"):
        row = table.loc[index]
        crop = load_crop(args.crops_root, row["G"] if args.side == "g" else row["P"])
        picture = bgr_to_rgb(crop).transpose(1, 2, 0)

        panels_by_model = []
        for model_name in models:
            directory = os.path.join(
                args.attributions, "{method}", model_name, args.dataset, args.protocol_name
            )
            panels = [np.uint8(255 * picture)]
            for method in methods:
                base = directory.format(method=method)
                for suffix in ("pos", "neg"):
                    if suffix == "neg" and method not in TWO_SIDED:
                        continue
                    attribution = load_map(
                        os.path.join(base, f"{index}_{args.side}_{suffix}.npy")
                    )
                    panels.append(
                        None if attribution is None else overlay(picture, attribution, args.alpha)
                    )
            panels_by_model.append(panels)

        figure = contact_sheet(
            panels_by_model,
            row_labels=models,
            column_labels=columns,
            title=f"{args.dataset} {args.protocol_name}, pair {index} "
            f"({'genuine' if int(row['T']) == 1 else 'impostor'})",
        )
        os.makedirs(args.output, exist_ok=True)
        path = os.path.join(
            args.output, f"comparison_{args.dataset}_{args.protocol_name}_{index}.pdf"
        )
        figure.savefig(path)
        figure.clf()
        log.info(f"wrote {path}")


def main():
    args = parse_args()
    log = RunLogger(args.output, f"figures_{args.task}", args)
    with log.stage(f"{args.task} figures"):
        if args.task == "classification":
            classification_figures(args, log)
        else:
            verification_figures(args, log)
    log.finish()


if __name__ == "__main__":
    main()

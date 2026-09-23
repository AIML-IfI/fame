"""Visualize which pixels feed each feature map location.

Covers Fig. 1 of the main paper and the complete sheets of Fig. 7 and 8 in the
supplemental material: one attribution per location of the final feature map,
which is the evidence used to argue that a[k] is not confined to the pixels
underneath it and that CAM's upsampling assumption therefore fails for deep
networks.  ``--mark-receptive-field`` draws the white boxes of Fig. 1 showing
where CAM assumes the information comes from, which is what makes the mismatch
legible.

Example:
    python scripts/run_fame_featuremap.py \
        --image sample.JPEG --model ResNet101 --output results/featuremap
"""

import argparse
import os

import numpy as np
import torch
import torchvision
from PIL import Image

import _bootstrap

_bootstrap.ensure_repository_on_path()  # must precede the fame imports

from fame import FameConfig, explain_feature_map
from fame.models import build_classifier, build_face_model, face_target_layer, target_layer
from fame.runlog import RunLogger
from fame.visualization import contact_sheet, mark_receptive_field, overlay


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--model",
        default="ResNet101",
        help="ResNet34, ResNet50 or ResNet101 for the 7x7 experiment",
    )
    parser.add_argument("--checkpoint", default=None, help="required for face models")
    parser.add_argument("--task", default="classification", choices=["classification", "face"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=49,
        help="locations optimized in parallel; 49 does a whole 7x7 map in one pass",
    )
    parser.add_argument("--rows", nargs="+", type=int, default=None, help="feature map rows to render")
    parser.add_argument("--grid", action="store_true", help="write a contact sheet of all locations")
    parser.add_argument(
        "--mark-receptive-field",
        action="store_true",
        help="outline the pixels CAM assumes each location sees (Fig. 1)",
    )
    return parser.parse_args()


def load_image(path: str, size: int, task: str) -> torch.Tensor:
    with Image.open(path) as handle:
        image = handle.convert("RGB")
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size if task == "face" else 232),
            torchvision.transforms.CenterCrop(size),
            torchvision.transforms.ToTensor(),
        ]
    )
    tensor = transform(image)
    if task == "face":
        # Face backbones expect BGR, matching the pair loader.
        tensor = tensor.flip(0)
    return tensor


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    log = RunLogger(args.output, "fame_featuremap", args, args.device)

    if args.task == "face":
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for face models")
        model = build_face_model(args.model, args.checkpoint, device=args.device)
        layer = face_target_layer(model)
        image = load_image(args.image, 112, "face")
    else:
        model = build_classifier(args.model, device=args.device)
        layer = target_layer(model, args.model)
        image = load_image(args.image, 224, "classification")

    image = image[None].to(args.device)

    # Determine the feature map size so that locations can be labelled by
    # their (row, col) index and, optionally, restricted to a few rows.
    from fame.models import FeatureMapExtractor

    with FeatureMapExtractor(model, layer) as extractor, torch.no_grad():
        _, _, height, width = extractor(image).shape
    log.info(f"feature map: {height}x{width} ({height * width} locations)")
    if (height, width) != (7, 7):
        log.warning(
            f"{args.model} gives a {height}x{width} feature map, not the 7x7 the "
            "experiment is built around; the sheet layout and the comparison "
            "against the ResNet results both assume 49 locations"
        )

    rows = args.rows if args.rows is not None else list(range(height))
    locations = [(row, col) for row in rows for col in range(width)]

    config = FameConfig(iterations=args.iterations)
    with log.stage(f"{args.model} feature map"):
        maps = explain_feature_map(
            model, layer, image, locations=locations, config=config, batch_size=args.batch_size
        )
    log.count(f"{args.model} feature map", len(locations))

    picture = image[0].cpu()
    if args.task == "face":
        picture = picture.flip(0)

    for (row, col), attribution in zip(locations, maps.cpu()):
        stem = os.path.join(args.output, f"{args.model}_r{row}_c{col}")
        np.save(stem + ".npy", attribution[0].numpy())

    if args.grid:
        # One row of the sheet per feature map row, so the layout mirrors the
        # feature map itself and a panel sits where its location does.
        panels = []
        for row_index, row in enumerate(rows):
            panel_row = []
            for column in range(width):
                attribution = maps[row_index * width + column].cpu()
                panel = overlay(picture, attribution)
                if args.mark_receptive_field:
                    panel = mark_receptive_field(panel, (row, column), (height, width))
                panel_row.append(panel)
            panels.append(panel_row)

        figure = contact_sheet(
            panels,
            row_labels=[f"row {row}" for row in rows],
            title=f"{args.model}: attribution per feature map location",
        )
        figure.savefig(os.path.join(args.output, f"{args.model}_featuremap.pdf"))

    log.finish()


if __name__ == "__main__":
    main()

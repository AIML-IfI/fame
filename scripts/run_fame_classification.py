"""Generate FAME attribution maps for ImageNet classification (Sec. 4.2).

Example:
    python scripts/run_fame_classification.py \
        --images-root /data/ILSVRC/Data/CLS-LOC/val \
        --synset-mapping /data/ILSVRC/LOC_synset_mapping.txt \
        --protocol protocols/imagenet.csv \
        --output results/FAME \
        --models ResNet50 ConvNeXt_Tiny
"""

import argparse
import os

import numpy as np
import torch
import torchvision
from tqdm import tqdm

import _bootstrap

_bootstrap.ensure_repository_on_path()  # must precede the fame imports

from fame import FameConfig, explain_classification
from fame.data import ImageNetSubset
from fame.models import CLASSIFIERS, IMAGENET_MEAN, IMAGENET_STD, build_classifier
from fame.runlog import RunLogger
from fame.visualization import overlay


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", required=True, help="ImageNet validation images")
    parser.add_argument("--synset-mapping", required=True, help="LOC_synset_mapping.txt")
    parser.add_argument("--protocol", default="protocols/imagenet.csv")
    parser.add_argument("--output", required=True, help="directory for the saved maps")
    parser.add_argument("--models", nargs="+", default=list(CLASSIFIERS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="early-stopping loss threshold; the original script used 1e-3",
    )
    parser.add_argument("--step-size", type=float, default=1.0 / 255.0)
    parser.add_argument("--blur-kernel", type=int, default=49)
    parser.add_argument("--blur-sigma", type=float, default=7.7)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N images")
    parser.add_argument(
        "--input-space",
        default="normalized",
        choices=["normalized", "pixel"],
        help=(
            "space in which FAME takes its 1/255 step. 'normalized' perturbs the "
            "mean/std normalized tensor and reproduces the reported results; "
            "'pixel' perturbs the [0, 1] image so the step is one grey level on "
            "every channel"
        ),
    )
    parser.add_argument("--save-overlay", action="store_true", help="also write PNG previews")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    log = RunLogger(args.output, "fame_classification", args, args.device)
    pixel_space = args.input_space == "pixel"
    config = FameConfig(
        step_size=args.step_size,
        iterations=args.iterations,
        epsilon=args.epsilon,
        blur_kernel=args.blur_kernel,
        blur_sigma=args.blur_sigma,
        # Clamping only makes sense when the tensor being perturbed is an image.
        clamp=(0.0, 1.0) if pixel_space else None,
    )
    normalize = torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    dataset = ImageNetSubset(
        args.protocol, args.images_root, args.synset_mapping, limit=args.limit
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    for model_name in args.models:
        # In normalized space the backbone is used bare and the batch is
        # normalized here; in pixel space the wrapper does it inside forward.
        model = build_classifier(model_name, device=args.device, normalized=pixel_space)
        destination = os.path.join(args.output, model_name)
        os.makedirs(destination, exist_ok=True)
        explained = 0

        with log.stage(model_name):
            for batch in tqdm(loader, desc=model_name):
                names = [
                    f"{image_id}_{int(target)}"
                    for image_id, target in zip(batch["image_id"], batch["target"])
                ]
                if not args.overwrite and all(
                    os.path.exists(os.path.join(destination, f"{name}.npy")) for name in names
                ):
                    continue

                images = batch["image"].to(args.device)
                targets = batch["target"].to(args.device)
                model_input = images if pixel_space else normalize(images)
                maps = explain_classification(model, model_input, targets, config).cpu()
                explained += len(names)

                for name, attribution, image in zip(names, maps, batch["image"]):
                    # Save the raw single-channel map; colouring happens at display
                    # time so that evaluation ranks pixels by attribution value.
                    np.save(os.path.join(destination, f"{name}.npy"), attribution[0].numpy())
                    if args.save_overlay:
                        import matplotlib.pyplot as plt

                        plt.imsave(
                            os.path.join(destination, f"{name}.png"), overlay(image, attribution)
                        )

        log.count(model_name, explained)
        del model
        torch.cuda.empty_cache()

    log.finish()


if __name__ == "__main__":
    main()

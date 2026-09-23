"""Parameter sensitivity of FAME (supplemental Sec. A, Fig. 6; Sec. 4.4, Fig. 5).

Sweeps the three knobs that shape a FAME map and lays the results out as a
contact sheet:

    iterations   Fig. 6(a), step size fixed at 1/255.  Maps are coarse after a
                 single step, sharpen quickly, and settle between roughly 75
                 and 200; past 300 the extra optimization mostly adds
                 background activation.
    step size    Fig. 6(b), 100 iterations.  Small eta gives smoother but
                 weaker maps, larger eta stronger and more contrasted ones,
                 with some background noise at the top end.
    blur         Fig. 5.  Small kernels keep high-frequency detail, large ones
                 wash the map out across the whole object.

Early stopping is disabled throughout, as Fig. 6 requires: with it on, the
shorter runs would not all take the number of steps they are labelled with.

Example:
    python scripts/run_parameter_sweep.py --image sample.JPEG --model ResNet50 \
        --sweep iterations --output results/sweeps
"""

import argparse
import json
import os

import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

import _bootstrap

_bootstrap.ensure_repository_on_path()  # must precede the fame imports

from fame import FameConfig, explain_classification, explain_verification
from fame.data import load_crop
from fame.models import IMAGENET_MEAN, IMAGENET_STD, build_classifier, build_face_model
from fame.runlog import RunLogger
from fame.visualization import bgr_to_rgb, contact_sheet, overlay

# Iteration counts of Fig. 6(a).  The runtime table has nine rows whose values
# grow linearly at about 19 s per iteration per 1000 images, which is
# consistent with this set; adjust with --values if your figure used another.
DEFAULT_ITERATIONS = [1, 25, 50, 75, 100, 200, 300, 400, 500]
DEFAULT_STEP_SIZES = [0.25 / 255, 0.5 / 255, 1.0 / 255, 2.0 / 255, 4.0 / 255, 8.0 / 255]
DEFAULT_BLURS = [5, 11, 21, 31, 49, 75]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="image to explain")
    parser.add_argument(
        "--reference",
        default=None,
        help="partner image; supplying it switches to the verification loss",
    )
    parser.add_argument("--model", default="ResNet50")
    parser.add_argument("--checkpoint", default=None, help="required for face models")
    parser.add_argument("--target-class", type=int, default=None, help="defaults to the prediction")
    parser.add_argument(
        "--sweep",
        nargs="+",
        default=["iterations", "step_size", "blur"],
        choices=["iterations", "step_size", "blur"],
    )
    parser.add_argument("--values", nargs="+", type=float, default=None, help="override the sweep")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=100, help="fixed value for other sweeps")
    parser.add_argument("--step-size", type=float, default=1.0 / 255.0)
    parser.add_argument("--blur-kernel", type=int, default=49)
    parser.add_argument("--blur-sigma", type=float, default=7.7)
    parser.add_argument(
        "--input-space", default="normalized", choices=["normalized", "pixel"]
    )
    return parser.parse_args()


def load_classification_image(path: str):
    with Image.open(path) as handle:
        image = handle.convert("RGB")
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(232),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
        ]
    )
    return transform(image)


def configs_for(sweep: str, values, args):
    """Build one FameConfig per swept value, holding the other knobs fixed."""
    base = dict(
        step_size=args.step_size,
        iterations=args.iterations,
        epsilon=None,  # Fig. 6 is produced without early stopping.
        blur_kernel=args.blur_kernel,
        blur_sigma=args.blur_sigma,
        clamp=(0.0, 1.0) if args.input_space == "pixel" else None,
    )

    for value in values:
        settings = dict(base)
        if sweep == "iterations":
            settings["iterations"] = int(value)
            label = f"n={int(value)}"
        elif sweep == "step_size":
            settings["step_size"] = float(value)
            label = f"eta={value * 255:.2f}/255"
        else:
            # Fig. 5 varies the kernel; sigma follows it so the smoothing scale
            # actually changes rather than being clipped by a fixed kernel.
            settings["blur_kernel"] = int(value)
            settings["blur_sigma"] = max(float(value) / 6.37, 0.5)
            label = f"b={int(value)}"
        yield label, FameConfig(**settings)


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    log = RunLogger(args.output, "parameter_sweep", args, args.device)
    face_mode = args.reference is not None

    if face_mode:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required when --reference is given")
        model = build_face_model(args.model, args.checkpoint, device=args.device)
        image = load_crop(os.path.dirname(args.image) or ".", os.path.basename(args.image))
        reference = load_crop(
            os.path.dirname(args.reference) or ".", os.path.basename(args.reference)
        )
        display = bgr_to_rgb(image).transpose(1, 2, 0)
        model_input = image[None].to(args.device)
        reference_input = reference[None].to(args.device)
    else:
        pixel_space = args.input_space == "pixel"
        model = build_classifier(args.model, device=args.device, normalized=pixel_space)
        image = load_classification_image(args.image)
        display = image.numpy().transpose(1, 2, 0)
        normalize = torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        raw = image[None].to(args.device)
        model_input = raw if pixel_space else normalize(raw)

        target = args.target_class
        if target is None:
            with torch.no_grad():
                target = int(model(model_input).argmax(dim=1).item())
            log.info(f"explaining predicted class {target}")
        targets = torch.tensor([target], device=args.device)

    defaults = {
        "iterations": DEFAULT_ITERATIONS,
        "step_size": DEFAULT_STEP_SIZES,
        "blur": DEFAULT_BLURS,
    }

    record = []
    rows, row_labels = [], []
    for sweep in args.sweep:
        values = args.values if args.values is not None else defaults[sweep]
        panels, labels = [], []

        with log.stage(sweep):
            for label, config in tqdm(list(configs_for(sweep, values, args)), desc=sweep):
                if face_mode:
                    maps = explain_verification(
                        model, model_input, reference_input, config, modes=("similar",)
                    )
                    attribution = maps["similar"][0].cpu()
                else:
                    attribution = explain_classification(model, model_input, targets, config)[0].cpu()

                panels.append(overlay(display, attribution))
                labels.append(label)
                np.save(
                    os.path.join(args.output, f"{args.model}_{sweep}_{label.replace('/', 'per')}.npy"),
                    attribution.numpy()[0],
                )
                record.append({"sweep": sweep, "label": label})

        rows.append(panels)
        row_labels.append(sweep)
        figure = contact_sheet([panels], column_labels=labels, title=f"FAME: {sweep} sweep")
        figure.savefig(os.path.join(args.output, f"{args.model}_{sweep}.pdf"))
        figure.clf()

    if len(rows) > 1:
        figure = contact_sheet(rows, row_labels=row_labels, title=f"FAME parameters ({args.model})")
        figure.savefig(os.path.join(args.output, f"{args.model}_sweeps.pdf"))

    with open(os.path.join(args.output, f"{args.model}_sweeps.json"), "w") as handle:
        json.dump(record, handle, indent=2)

    log.finish()


if __name__ == "__main__":
    main()

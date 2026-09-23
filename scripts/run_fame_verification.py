"""Generate attribution maps for face verification (Sec. 4.3).

Handles FAME and every baseline through one loop, so the maps that later feed
the deletion/insertion evaluation are produced under identical conditions.
Both sides of each pair are explained: the gallery is explained against a
frozen probe embedding and vice versa.

Saved ``.npy`` files hold the finished grayscale attribution: absolute
perturbation, converted to grey, blurred and normalized to [0, 1].  The
evaluation therefore loads and ranks them directly, with no post-processing to
repeat and nothing to keep in sync between the two scripts.  ``--save-overlay``
additionally writes the jet-coloured overlay as a PNG for figures; colour never
enters the arrays that the metrics read.

Example:
    python scripts/run_fame_verification.py \
        --images-root /data/ARface --protocol-dir protocols/ARface \
        --checkpoint /models/adaface_ir101_webface12m.ckpt \
        --model Adaface_ir_101 --dataset ARface --method FAME \
        --output results/FAME
"""

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

import _bootstrap

_bootstrap.ensure_repository_on_path()  # must precede the fame imports

from fame import FameConfig, explain_verification
from fame.attribution import smooth_and_normalize
from fame.baselines import corrise, fggb, verification_cam
from fame.data import PROTOCOLS, VerificationPairs
from fame.metrics import eer_threshold
from fame.models import build_face_model, face_target_layer
from fame.runlog import RunLogger
from fame.visualization import bgr_to_rgb, overlay

METHODS = ["FAME", "GradCAM", "GradCAMElementWise", "CorrRISE", "FGGB"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", required=True, help="directory of aligned HDF5 crops")
    parser.add_argument("--protocol-dir", required=True, help="directory of protocol CSVs")
    parser.add_argument("--checkpoint", required=True, help="AdaFace checkpoint")
    parser.add_argument("--model", default="Adaface_ir_101")
    parser.add_argument("--dataset", default="ARface", choices=list(PROTOCOLS))
    parser.add_argument("--protocols", nargs="+", default=None)
    parser.add_argument("--method", default="FAME", choices=METHODS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="early-stopping loss threshold, matching the original setting",
    )
    parser.add_argument("--step-size", type=float, default=1.0 / 255.0)
    parser.add_argument("--blur-kernel", type=int, default=49)
    parser.add_argument("--blur-sigma", type=float, default=7.7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--save-overlay", action="store_true", help="also write jet-coloured PNG previews"
    )
    parser.add_argument(
        "--fggb-threshold",
        type=float,
        default=None,
        help=(
            "theta for FGGB; estimated from the protocol's EER when omitted. "
            "The original reads it from scores_eer.csv, where it is stored "
            "rounded to four decimals"
        ),
    )
    return parser.parse_args()


@torch.no_grad()
def clean_scores(model, dataset, device):
    """Similarity of every unperturbed pair, used to locate the EER threshold."""
    scores = []
    for index in tqdm(range(len(dataset)), desc="clean scores", leave=False):
        item = dataset[index]
        gallery = model(item["gallery"][None].to(device))
        probe = model(item["probe"][None].to(device))
        gallery = gallery / gallery.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        probe = probe / probe.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        scores.append(float((gallery * probe).sum()))
    return scores


# Smoothing the original applied to each baseline before saving.  FGGB was
# blurred with kernel 25 and sigma 5; CorrRISE and the CAMs were not.
BASELINE_BLUR = {"FGGB": (25, 5.0), "CorrRISE": (None, 0.0)}


def explain_pair(method, model, target_layers, image, reference, config, threshold):
    """Return a dict with ``similar`` and optionally ``dissimilar`` maps."""
    if method == "FAME":
        maps = explain_verification(model, image, reference, config)
        return {key: value[0] for key, value in maps.items()}

    # Baselines get the same treatment they had in the original code, then are
    # normalized to [0, 1] so every method on disk has one value range.
    if method == "FGGB":
        maps = fggb(model, image, reference, threshold=threshold)
    elif method == "CorrRISE":
        maps = corrise(model, image, reference)
    else:
        # CAM baselines produce a single map with no similar/dissimilar split.
        attribution = verification_cam(model, target_layers, image, reference, method=method)
        maps = {"similar": torch.from_numpy(attribution[0])[None]}

    kernel, sigma = BASELINE_BLUR.get(method, (None, 0.0))
    return {
        key: smooth_and_normalize(value, blur_kernel=kernel, blur_sigma=sigma)
        for key, value in maps.items()
    }


def main():
    args = parse_args()
    config = FameConfig(
        step_size=args.step_size,
        iterations=args.iterations,
        epsilon=args.epsilon,
        blur_kernel=args.blur_kernel,
        blur_sigma=args.blur_sigma,
    )
    log = RunLogger(args.output, f"{args.method}_verification", args, args.device)
    model = build_face_model(args.model, args.checkpoint, device=args.device)
    target_layers = [face_target_layer(model)]
    protocols = args.protocols or PROTOCOLS[args.dataset]

    for protocol in protocols:
        dataset = VerificationPairs(
            os.path.join(args.protocol_dir, f"{protocol}.csv"),
            args.images_root,
            limit=args.limit,
        )
        destination = os.path.join(args.output, args.model, args.dataset, protocol)
        os.makedirs(destination, exist_ok=True)

        threshold = args.fggb_threshold
        if args.method == "FGGB" and threshold is None:
            threshold = eer_threshold(clean_scores(model, dataset, args.device), dataset.labels)
            log.info(f"{protocol}: estimated FGGB threshold {threshold:.4f}")

        with log.stage(f"{args.method}/{protocol}"):
            for index in tqdm(range(len(dataset)), desc=f"{args.method}/{protocol}"):
                item = dataset[index]
                gallery = item["gallery"][None].to(args.device)
                probe = item["probe"][None].to(args.device)

                for side, image, reference in (("g", gallery, probe), ("p", probe, gallery)):
                    maps = explain_pair(
                        args.method, model, target_layers, image, reference, config, threshold
                    )
                    for mode, attribution in maps.items():
                        suffix = "pos" if mode == "similar" else "neg"
                        stem = os.path.join(destination, f"{index}_{side}_{suffix}")

                        # Grayscale, already blurred and normalized to [0, 1].
                        grayscale = attribution.detach().cpu().numpy()[0]
                        np.save(stem + ".npy", grayscale)

                        if args.save_overlay:
                            import matplotlib.pyplot as plt

                            crop = item["gallery"] if side == "g" else item["probe"]
                            plt.imsave(stem + ".png", overlay(bgr_to_rgb(crop), grayscale))
        log.count(f"{args.method}/{protocol}", len(dataset))

    log.finish()


if __name__ == "__main__":
    main()

"""Deletion and insertion evaluation for face verification (Tab. 2).

This single script replaces ``eval_delete_{fame,cam,fggb,corr}.py`` and
``eval_insert_{fame,cams,fggb,corr}.py`` plus ``eval_auc.py``.  Those eight
files were copies of one another whose only real differences were the filename
of the saved map, whether the map needed extra post-processing on load, and one
comparison operator in the masking function.  Here the direction is a flag and
the maps are already stored as plain single-channel arrays, so nothing is
method-specific.

The old FAME variants re-applied a 49/7.7 Gaussian blur to maps that had
already been blurred and jet-coloured when they were saved, meaning pixels were
ranked by a colormap rather than by attribution.  Producing maps with
``run_fame_verification.py`` and reading them here avoids that entirely.

Example:
    python scripts/eval_verification.py \
        --images-root /data/ARface --protocol-dir protocols/ARface \
        --checkpoint /models/adaface_ir101_webface12m.ckpt \
        --model Adaface_ir_101 --dataset ARface \
        --attributions results --methods FAME GradCAM CorrRISE FGGB \
        --output results/ARface_metrics.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import _bootstrap

_bootstrap.ensure_repository_on_path()  # must precede the fame imports

from fame.data import PROTOCOLS, VerificationPairs
from fame.metrics import PERCENTAGES, curve_from_scores, eer_threshold, top_percent_mask
from fame.models import build_face_model
from fame.runlog import RunLogger


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--protocol-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="Adaface_ir_101")
    parser.add_argument("--dataset", default="ARface", choices=list(PROTOCOLS))
    parser.add_argument("--protocols", nargs="+", default=None)
    parser.add_argument("--attributions", required=True, help="root written by the run scripts")
    parser.add_argument("--methods", nargs="+", default=["FAME"])
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--perturb",
        default="probe",
        choices=["probe", "both"],
        help="perturb the probe only (paper protocol) or both images",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "decision threshold; recomputed from the clean scores when omitted. "
            "scores_eer.csv stores it rounded to four decimals, so pass the "
            "stored value to match previously reported accuracies exactly"
        ),
    )
    return parser.parse_args()


@torch.no_grad()
def similarity(model, gallery, probe):
    """Cosine similarity between two batches of images."""
    embedding_g = model(gallery)
    embedding_p = model(probe)
    embedding_g = embedding_g / embedding_g.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
    embedding_p = embedding_p / embedding_p.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
    return (embedding_g * embedding_p).sum(dim=1)


def apply_mask(image: torch.Tensor, attribution: np.ndarray, percentage: float, keep_top: bool):
    """Zero out (or keep only) the top-ranked pixels of one image."""
    mask = top_percent_mask(attribution, percentage, keep_top=keep_top)
    return image * torch.from_numpy(mask).to(image.dtype).to(image.device)


def load_map(root, method, model_name, dataset_name, protocol, index, side):
    path = os.path.join(
        root, method, model_name, dataset_name, protocol, f"{index}_{side}_pos.npy"
    )
    return np.squeeze(np.load(path))


def main():
    args = parse_args()
    log = RunLogger(args.output, "eval_verification", args, args.device)
    model = build_face_model(args.model, args.checkpoint, device=args.device)
    protocols = args.protocols or PROTOCOLS[args.dataset]

    rows = []
    for protocol in protocols:
        dataset = VerificationPairs(
            os.path.join(args.protocol_dir, f"{protocol}.csv"),
            args.images_root,
            limit=args.limit,
        )
        labels = dataset.labels

        # The operating point comes from the clean scores, so every method is
        # judged against the same decision threshold.
        clean = []
        for index in tqdm(range(len(dataset)), desc=f"{protocol} clean", leave=False):
            item = dataset[index]
            clean.append(
                float(
                    similarity(
                        model,
                        item["gallery"][None].to(args.device),
                        item["probe"][None].to(args.device),
                    )
                )
            )
        threshold = args.threshold if args.threshold is not None else eer_threshold(clean, labels)
        source = "given" if args.threshold is not None else "computed"
        log.info(
            f"{protocol}: {source} threshold {threshold:.4f}, clean accuracy "
            f"{float(((np.array(clean) >= threshold).astype(int) == labels).mean()):.4f}"
        )

        for method in args.methods:
            for direction, keep_top in (("Delete", False), ("Insert", True)):
                with log.stage(f"{method}/{protocol}/{direction}"):
                    scores_by_percentage = {}
                    for percentage in tqdm(
                        PERCENTAGES, desc=f"{method}/{protocol}/{direction}", leave=False
                    ):
                        scores = []
                        for index in range(len(dataset)):
                            item = dataset[index]
                            gallery = item["gallery"][None].to(args.device)
                            probe = item["probe"][None].to(args.device)

                            probe = apply_mask(
                                probe,
                                load_map(
                                    args.attributions, method, args.model, args.dataset,
                                    protocol, index, "p",
                                ),
                                percentage,
                                keep_top,
                            )
                            if args.perturb == "both":
                                gallery = apply_mask(
                                    gallery,
                                    load_map(
                                        args.attributions, method, args.model, args.dataset,
                                        protocol, index, "g",
                                    ),
                                    percentage,
                                    keep_top,
                                )
                            scores.append(float(similarity(model, gallery, probe)))
                        scores_by_percentage[float(percentage)] = scores

                curve = curve_from_scores(scores_by_percentage, labels, threshold)
                rows.append(
                    {
                        "model": args.model,
                        "dataset": args.dataset,
                        "protocol": protocol,
                        "method": method,
                        "metric": direction,
                        "auc": 100.0 * curve["auc"],
                        "accuracies": ";".join(f"{a:.4f}" for a in curve["accuracies"]),
                    }
                )
                log.info(f"  {method} {direction}: AUC {100 * curve['auc']:.2f}")

    table = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    table.to_csv(args.output, index=False, float_format="%.4f")
    log.info(f"wrote {args.output}")
    log.info(
        "\n" + table.pivot_table(index=["protocol", "method"], columns="metric", values="auc").to_string()
    )
    log.finish()


if __name__ == "__main__":
    main()

"""Verification scores and the EER operating point for every protocol.

Port of ``predict.py`` plus the EER table it feeds.  Two outputs, matching the
originals:

    <output>/<model>/<dataset>_<protocol>.csv   one row per pair, with the
                                                columns Sim-Score and targets
    <output>/scores_eer.csv                     one row per (model, dataset,
                                                protocol) with thr and eer

Everything downstream needs the threshold: it decides the accuracy at every
removal ratio in Tab. 2, and it is the theta that FGGB subtracts.  Running this
first and passing the result to the other scripts keeps one operating point per
protocol across all methods, which is what makes their AUCs comparable.

Note that the original stores the threshold formatted to four decimals, so the
value used downstream is rounded; ``--full-precision`` keeps all digits
instead.

Example:
    python scripts/compute_verification_scores.py \
        --images-root $CROPS --protocol-dir $PROTOCOLS \
        --checkpoints Adaface_ir_101=/models/adaface_ir101_webface12m.ckpt \
        --datasets ARface SCface CFP --output results/scores
"""

import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm

import _bootstrap

_bootstrap.ensure_repository_on_path()  # must precede the fame imports

from fame.data import PROTOCOLS, VerificationPairs
from fame.metrics import accuracy_at, equal_error_rate
from fame.models import FACE_MODELS, build_face_model
from fame.runlog import RunLogger


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", required=True, help="root of the aligned HDF5 crops")
    parser.add_argument(
        "--protocol-dir",
        required=True,
        help="directory holding <dataset>/<protocol>.csv",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        metavar="NAME=PATH",
        help=f"one or more of {list(FACE_MODELS)} mapped to a checkpoint",
    )
    parser.add_argument("--datasets", nargs="+", default=list(PROTOCOLS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--full-precision",
        action="store_true",
        help="store the threshold with all digits instead of rounding to four decimals",
    )
    return parser.parse_args()


@torch.no_grad()
def pair_scores(model, dataset, device):
    """Cosine similarity of every pair, gallery against probe."""
    scores = []
    for index in tqdm(range(len(dataset)), desc="scoring", leave=False):
        item = dataset[index]
        gallery = model(item["gallery"][None].to(device))
        probe = model(item["probe"][None].to(device))
        gallery = gallery / gallery.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        probe = probe / probe.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        scores.append(float((gallery * probe).sum()))
    return scores


def main():
    args = parse_args()
    log = RunLogger(args.output, "verification_scores", args, args.device)

    checkpoints = {}
    for entry in args.checkpoints:
        if "=" not in entry:
            raise SystemExit(f"expected NAME=PATH, got {entry!r}")
        name, path = entry.split("=", 1)
        checkpoints[name] = path

    summary = []
    for model_name, checkpoint in checkpoints.items():
        model = build_face_model(model_name, checkpoint, device=args.device)
        destination = os.path.join(args.output, model_name)
        os.makedirs(destination, exist_ok=True)

        for dataset_name in args.datasets:
            for protocol in PROTOCOLS[dataset_name]:
                stage = f"{model_name}/{dataset_name}/{protocol}"
                with log.stage(stage):
                    dataset = VerificationPairs(
                        os.path.join(args.protocol_dir, dataset_name, f"{protocol}.csv"),
                        args.images_root,
                        limit=args.limit,
                    )
                    scores = pair_scores(model, dataset, args.device)
                    labels = dataset.labels

                    table = pd.DataFrame({"Sim-Score": scores, "targets": labels})
                    table.to_csv(
                        os.path.join(destination, f"{dataset_name}_{protocol}.csv"), index=False
                    )

                    threshold, eer = equal_error_rate(scores, labels)
                    stored = threshold if args.full_precision else float(f"{threshold:.4f}")
                    accuracy = accuracy_at(scores, labels, stored)

                log.count(stage, len(dataset))
                log.info(
                    f"{stage}: thr {stored:.4f}, eer {100 * eer:.2f}%, "
                    f"clean accuracy {100 * accuracy:.2f}%"
                )
                summary.append(
                    {
                        "model_name": model_name,
                        "data_name": dataset_name,
                        "protocol": protocol,
                        "thr": f"{stored:.4f}" if not args.full_precision else stored,
                        "eer": f"{eer:.4f}",
                        "accuracy": f"{accuracy:.4f}",
                        "pairs": len(dataset),
                    }
                )

        del model
        torch.cuda.empty_cache()

    frame = pd.DataFrame(summary)
    path = os.path.join(args.output, "scores_eer.csv")
    frame.to_csv(path, index=False)
    log.info(f"wrote {path}")
    log.info("\n" + frame.to_string(index=False))
    log.finish()


if __name__ == "__main__":
    main()

"""Runtime evaluation (supplemental Sec. B, Tab. 3).

Two tables:

    (a) FAME on 1000 ImageNet images with ResNet50, as a function of the number
        of iterations.  Cost grows linearly, so the iteration count is the only
        real runtime knob.
    (b) All XAI methods on the 700 pairs of CFP-FP with IResNet101.  Grad-CAM
        variants are fastest because they stop at the activation map; CorrRISE
        and FGGB are slowest because they need hundreds of masked forward
        passes or one backward pass per embedding dimension; FAME sits in
        between.

``--mode sequential`` is the default and is what the paper reports: the
reference implementations of CorrRISE and FGGB process one image at a time, so
batching FAME alone would not be a like-for-like comparison.  ``--mode batched``
measures the parallel implementation the supplemental mentions, which is the
number that matters if you actually want throughput.

Example:
    python scripts/run_runtime_benchmark.py --table a \
        --images-root $VAL --synset-mapping $SYNSETS --limit 1000 \
        --output results/runtime_a.csv
"""

import argparse
import os

import pandas as pd
import torch
import torchvision
from tqdm import tqdm

import _bootstrap

_bootstrap.ensure_repository_on_path()  # must precede the fame imports

from fame import FameConfig, explain_classification, explain_verification
from fame.baselines import corrise, fggb, verification_cam
from fame.benchmark import measure
from fame.runlog import RunLogger
from fame.data import ImageNetSubset, VerificationPairs
from fame.metrics import eer_threshold
from fame.models import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_classifier,
    build_face_model,
    face_target_layer,
)

# Iteration counts of Tab. 3(a); the table has nine rows.
DEFAULT_ITERATIONS = [1, 25, 50, 75, 100, 200, 300, 400, 500]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True, choices=["a", "b"])
    parser.add_argument("--output", required=True, help="destination CSV")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--mode",
        default="sequential",
        choices=["sequential", "batched"],
        help="sequential matches the reported numbers; batched measures throughput",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="only used in batched mode")

    # Table (a)
    parser.add_argument("--images-root", default=None)
    parser.add_argument("--synset-mapping", default=None)
    parser.add_argument(
        "--protocol",
        default="protocols/imagenet_sub.csv",
        help="Tab. 3(a) reports 1000 images, which is what imagenet_sub.csv holds",
    )
    parser.add_argument("--model", default=None, help="defaults to ResNet50 / Adaface_ir_101")
    parser.add_argument("--iterations", nargs="+", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--limit", type=int, default=1000)

    # Table (b)
    parser.add_argument("--crops-root", default=None)
    parser.add_argument("--protocol-dir", default=None)
    parser.add_argument("--face-protocol", default="01FP", help="CFP-FP in the paper")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["GradCAM", "GradCAMElementWise", "CorrRISE", "FGGB", "FAME"],
    )
    parser.add_argument("--fame-iterations", type=int, default=500)
    return parser.parse_args()


def table_a(args, log):
    """FAME runtime against iteration count on image classification."""
    if not (args.images_root and args.synset_mapping):
        raise SystemExit("table (a) needs --images-root and --synset-mapping")

    model_name = args.model or "ResNet50"
    model = build_classifier(model_name, device=args.device, normalized=False)
    normalize = torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    dataset = ImageNetSubset(
        args.protocol, args.images_root, args.synset_mapping, limit=args.limit
    )

    rows = []
    for iterations in args.iterations:
        config = FameConfig(iterations=iterations, epsilon=None)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size if args.mode == "batched" else 1,
            shuffle=False,
            num_workers=4,
        )

        with measure(args.device) as timing:
            for batch in tqdm(loader, desc=f"{iterations} iterations", leave=False):
                images = normalize(batch["image"].to(args.device))
                explain_classification(model, images, batch["target"].to(args.device), config)

        rows.append(
            {
                "table": "a",
                "model": model_name,
                "mode": args.mode,
                "iterations": iterations,
                "images": len(dataset),
                "seconds": round(timing["seconds"], 2),
                "seconds_per_image": round(timing["seconds"] / len(dataset), 4),
                "peak_memory_gb": round(timing.get("peak_memory_gb", 0.0), 3),
            }
        )
        log.info(f"  {iterations:4d} iterations: {timing['seconds']:.2f} s")

    return rows


def table_b(args, log):
    """Runtime of every XAI method on one verification protocol."""
    if not (args.crops_root and args.protocol_dir and args.checkpoint):
        raise SystemExit("table (b) needs --crops-root, --protocol-dir and --checkpoint")

    model_name = args.model or "Adaface_ir_101"
    model = build_face_model(model_name, args.checkpoint, device=args.device)
    layers = [face_target_layer(model)]
    dataset = VerificationPairs(
        os.path.join(args.protocol_dir, f"{args.face_protocol}.csv"),
        args.crops_root,
        limit=args.limit,
    )
    config = FameConfig(iterations=args.fame_iterations, epsilon=None)

    threshold = None
    if "FGGB" in args.methods:
        # FGGB needs theta before it can run at all, and estimating it is part
        # of its cost only once, so it is measured outside the timed loop.
        scores = []
        with torch.no_grad():
            for index in range(len(dataset)):
                item = dataset[index]
                gallery = model(item["gallery"][None].to(args.device))
                probe = model(item["probe"][None].to(args.device))
                gallery = gallery / gallery.norm(dim=1, keepdim=True).clamp_min(1e-8)
                probe = probe / probe.norm(dim=1, keepdim=True).clamp_min(1e-8)
                scores.append(float((gallery * probe).sum()))
        threshold = eer_threshold(scores, dataset.labels)

    rows = []
    for method in args.methods:
        with measure(args.device) as timing:
            for index in tqdm(range(len(dataset)), desc=method, leave=False):
                item = dataset[index]
                gallery = item["gallery"][None].to(args.device)
                probe = item["probe"][None].to(args.device)

                # Both sides of the pair are explained, as in the run script.
                for image, reference in ((gallery, probe), (probe, gallery)):
                    if method == "FAME":
                        explain_verification(model, image, reference, config)
                    elif method == "FGGB":
                        fggb(model, image, reference, threshold=threshold)
                    elif method == "CorrRISE":
                        corrise(model, image, reference)
                    else:
                        verification_cam(model, layers, image, reference, method=method)

        rows.append(
            {
                "table": "b",
                "model": model_name,
                "protocol": args.face_protocol,
                "mode": args.mode,
                "method": method,
                "pairs": len(dataset),
                "seconds": round(timing["seconds"], 2),
                "seconds_per_pair": round(timing["seconds"] / len(dataset), 4),
                "peak_memory_gb": round(timing.get("peak_memory_gb", 0.0), 3),
            }
        )
        log.info(f"  {method:20s}: {timing['seconds']:.2f} s")

    return rows


def main():
    args = parse_args()
    if args.table == "b" and args.limit == 1000:
        # CFP-FP holds 700 pairs; do not silently cap at the table (a) default.
        args.limit = None

    log = RunLogger(args.output, f"runtime_table_{args.table}", args, args.device)

    rows = table_a(args, log) if args.table == "a" else table_b(args, log)

    table = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    table.to_csv(args.output, index=False)

    # The manifest already records the environment, so the table needs no
    # separate sidecar file.
    log.info(f"wrote {args.output}")
    log.info("\n" + table.to_string(index=False))
    log.finish()


if __name__ == "__main__":
    main()

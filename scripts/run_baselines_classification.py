"""Generate CAM baseline attributions for ImageNet (Tab. 1).

Replaces ``code_cam.py``.  Two behavioural fixes worth noting:

* The target layer is resolved per architecture instead of being hard-coded to
  a ConvNeXt path, so all five backbones can be run in one go.
* The multiprocessing pool that shipped a CUDA model to every worker is gone.
  CAM already batches internally and the pool mainly duplicated GPU memory.
"""

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

import _bootstrap

_bootstrap.ensure_repository_on_path()  # must precede the fame imports

from fame.baselines import CAM_METHODS, classification_cam
from fame.data import ImageNetSubset
from fame.models import CLASSIFIERS, build_classifier, target_layer
from fame.runlog import RunLogger


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--synset-mapping", required=True)
    parser.add_argument("--protocol", default="protocols/imagenet.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", nargs="+", default=list(CLASSIFIERS))
    parser.add_argument("--methods", nargs="+", default=list(CAM_METHODS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    log = RunLogger(args.output, "cam_classification", args, args.device)
    dataset = ImageNetSubset(
        args.protocol, args.images_root, args.synset_mapping, limit=args.limit
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    for model_name in args.models:
        # CAM implementations apply the model directly, and the wrapper that
        # folds normalization into the forward pass is transparent to them.
        model = build_classifier(model_name, device=args.device)
        layers = [target_layer(model, model_name)]

        for method in args.methods:
            destination = os.path.join(args.output, method, model_name)
            os.makedirs(destination, exist_ok=True)

            with log.stage(f"{method}/{model_name}"):
                for batch in tqdm(loader, desc=f"{method}/{model_name}"):
                    names = [
                        f"{image_id}_{int(target)}"
                        for image_id, target in zip(batch["image_id"], batch["target"])
                    ]
                    if not args.overwrite and all(
                        os.path.exists(os.path.join(destination, f"{name}.npy")) for name in names
                    ):
                        continue

                    maps = classification_cam(
                        model,
                        layers,
                        batch["image"].to(args.device),
                        batch["target"].tolist(),
                        method=method,
                    )
                    for name, attribution in zip(names, maps):
                        np.save(os.path.join(destination, f"{name}.npy"), attribution)

        del model
        torch.cuda.empty_cache()

    log.finish()


if __name__ == "__main__":
    main()

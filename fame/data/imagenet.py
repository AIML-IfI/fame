"""ImageNet validation data for the image classification experiments.

The protocol CSV has one row per image with columns ``ImageId`` and
``PredictionString``.  The latter packs the ground truth as repeated groups of
five fields, ``synset x_min y_min x_max y_max``, so a stride of 5 recovers the
synsets and the first group gives the bounding box used for IoU.
"""

import os
from typing import List, Optional, Tuple

import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset

RESIZE_SIZE = 232
CROP_SIZE = 224


def load_synset_mapping(path: str) -> dict:
    """Map WordNet synset ids to the class indices used by torchvision."""
    with open(path, "r") as handle:
        synsets = [line.strip().split(" ")[0] for line in handle if line.strip()]
    return {synset: index for index, synset in enumerate(synsets)}


def parse_prediction_string(prediction_string: str) -> Tuple[List[str], List[int]]:
    """Split a PredictionString into its synsets and its first bounding box."""
    fields = prediction_string.split()
    synsets = fields[::5]
    box = [int(value) for value in fields[1:5]]
    return synsets, box


def adjust_box(
    box: List[int], width: int, height: int, aspect_preserving: bool = True
) -> List[int]:
    """Map a box from original image coordinates into the resized centre crop.

    ``transforms.Resize(232)`` scales by the shorter side and keeps the aspect
    ratio, which is what ``aspect_preserving=True`` reproduces.

    ``adjust_bbx`` in the original ``eval_iou.py`` instead scales the two axes
    independently, as if the image were resized to a square 232x232, and then
    subtracts a fixed offset of (232 - 224) / 2 on both axes.  That places the
    box correctly only for images that are already square; for a 500x375 image
    it stretches the box along the shorter axis.  ``aspect_preserving=False``
    selects it, for reproducing previously reported IoU numbers.
    """
    x_min, y_min, x_max, y_max = box

    if aspect_preserving:
        scale = RESIZE_SIZE / min(width, height)
        scale_x = scale_y = scale
        offset_x = (width * scale - CROP_SIZE) / 2
        offset_y = (height * scale - CROP_SIZE) / 2
    else:
        scale_x = RESIZE_SIZE / width
        scale_y = RESIZE_SIZE / height
        offset_x = offset_y = (RESIZE_SIZE - CROP_SIZE) / 2

    x_min = max(0.0, x_min * scale_x - offset_x)
    y_min = max(0.0, y_min * scale_y - offset_y)
    x_max = min(float(CROP_SIZE), x_max * scale_x - offset_x)
    y_max = min(float(CROP_SIZE), y_max * scale_y - offset_y)
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


class ImageNetSubset(Dataset):
    """Images listed in the protocol CSV, resized and cropped to 224x224.

    Images are returned in [0, 1]; mean/std normalization belongs to the model
    (see ``fame.models.wrappers.NormalizedModel``).
    """

    def __init__(
        self,
        csv_path: str,
        images_root: str,
        synset_mapping_path: str,
        limit: Optional[int] = None,
        aspect_preserving_boxes: bool = True,
    ):
        self.table = pd.read_csv(csv_path)
        if limit is not None:
            self.table = self.table.iloc[:limit].reset_index(drop=True)
        self.images_root = images_root
        self.aspect_preserving_boxes = aspect_preserving_boxes
        self.synset_to_index = load_synset_mapping(synset_mapping_path)
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(RESIZE_SIZE),
                torchvision.transforms.CenterCrop(CROP_SIZE),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict:
        row = self.table.loc[index]
        image_id = row["ImageId"]
        synsets, box = parse_prediction_string(row["PredictionString"])

        with Image.open(os.path.join(self.images_root, image_id + ".JPEG")) as handle:
            image = handle.convert("RGB")
            width, height = image.size
            tensor = self.transform(image)

        return {
            "image": tensor,
            "target": torch.tensor(self.synset_to_index[synsets[0]], dtype=torch.long),
            "image_id": image_id,
            "box": torch.tensor(
                adjust_box(box, width, height, self.aspect_preserving_boxes), dtype=torch.long
            ),
        }

    def output_name(self, index: int) -> str:
        """File stem used for saved attributions, matching the original layout."""
        row = self.table.loc[index]
        synsets, _ = parse_prediction_string(row["PredictionString"])
        return f"{row['ImageId']}_{self.synset_to_index[synsets[0]]}"

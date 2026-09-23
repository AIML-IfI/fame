"""Verification protocols for AR Face, SCface and CFP.

Port of the ``Pairs`` dataset from the original ``utils.py``:

* A protocol CSV holds one row per pair in columns ``G`` (gallery), ``P``
  (probe) and ``T`` (1 for a genuine pair, 0 for an impostor pair).
* Aligned crops live in per-image HDF5 files whose names come from the CSV path
  with ``/`` replaced by ``_`` and the extension replaced by ``h5``; the crop
  itself sits in the ``data`` dataset.
* Crops are read with ``ToTensor``, so uint8 images arrive scaled to [0, 1].

The channel swap is *not* part of the original ``Pairs`` class -- it was applied
separately by the AdaFace-specific code paths (``cv2.cvtColor(img, RGB2BGR)`` in
``process_pair_*_adaface`` and in the deletion evaluation).  It is a constructor
flag here so that both paths stay reachable, and it defaults to on because every
model in the paper is an AdaFace model.
"""

import os
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Protocols reported in Tab. 2 of the paper.
PROTOCOLS = {
    "ARface": ["frontal", "glass", "scarf"],
    "SCface": ["close", "medium", "far"],
    "CFP": ["01FF", "01FP"],
}


def h5_filename(image_path: str) -> str:
    """Translate a protocol path into the flattened HDF5 filename."""
    return image_path.rsplit(".", 1)[0].replace("/", "_") + ".h5"


def load_crop(images_root: str, image_path: str, to_bgr: bool = True) -> torch.Tensor:
    """Read one aligned crop as a float tensor of shape (3, H, W) in [0, 1]."""
    with h5py.File(os.path.join(images_root, h5_filename(image_path)), "r") as handle:
        crop = np.asarray(handle["data"][:])

    if crop.ndim != 3 or crop.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 crop, got shape {crop.shape}")
    if to_bgr:
        crop = crop[:, :, ::-1]

    tensor = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1)))
    if tensor.dtype == torch.uint8:
        return tensor.float() / 255.0
    return tensor.float()


class VerificationPairs(Dataset):
    """Gallery/probe pairs for one dataset and protocol."""

    def __init__(
        self,
        protocol_csv: str,
        images_root: str,
        to_bgr: bool = True,
        genuine_only: bool = False,
        limit: Optional[int] = None,
    ):
        table = pd.read_csv(protocol_csv)
        missing = {"G", "P", "T"} - set(table.columns)
        if missing:
            raise ValueError(
                f"{protocol_csv} is missing the column(s) {sorted(missing)}; "
                "protocols are expected to have G (gallery), P (probe) and T (label)"
            )
        if genuine_only:
            # Figures in the paper visualize genuine pairs; the quantitative
            # deletion/insertion evaluation needs both classes to score
            # verification accuracy against the decision threshold.
            table = table[table["T"].astype(int) == 1]
        if limit is not None:
            table = table.iloc[:limit]
        self.table = table.reset_index(drop=True)
        self.images_root = images_root
        self.to_bgr = to_bgr

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict:
        row = self.table.loc[index]
        return {
            "gallery": load_crop(self.images_root, row["G"], self.to_bgr),
            "probe": load_crop(self.images_root, row["P"], self.to_bgr),
            "label": torch.tensor(int(row["T"]), dtype=torch.long),
            "index": index,
        }

    @property
    def labels(self) -> np.ndarray:
        return self.table["T"].astype(int).to_numpy()

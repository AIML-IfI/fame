"""Attribution quality metrics for image classification (Tab. 1)."""

from typing import Sequence

import numpy as np
import torch


def box_mask(box: Sequence[int], size: int = 224) -> np.ndarray:
    """Boolean mask of the ground truth bounding box."""
    mask = np.zeros((size, size), dtype=bool)
    x_min, y_min, x_max, y_max = (int(value) for value in box)
    mask[y_min:y_max, x_min:x_max] = True
    return mask


def intersection_over_union(attribution: np.ndarray, box: Sequence[int], threshold: float) -> float:
    """IoU between the thresholded attribution and the object box.

    Lower thresholds accept broader saliency, higher ones keep only the peaks;
    the paper reports 0.3, 0.5 and 0.7.
    """
    salient = np.squeeze(attribution) >= threshold
    target = box_mask(box, size=salient.shape[-1])
    intersection = np.logical_and(salient, target).sum()
    union = np.logical_or(salient, target).sum()
    # The epsilon matches compute_iou and keeps an empty union from dividing by
    # zero, which happens when the map is entirely below the threshold and the
    # box is degenerate.
    return float(intersection / (union + 1e-8))


def top_percent_mask(attribution: np.ndarray, percent: float, keep_top: bool = False) -> np.ndarray:
    """Mask selecting pixels by attribution rank.

    With ``keep_top=False`` the most important ``percent`` of pixels are set to
    zero (deletion); with ``keep_top=True`` only they are kept (insertion).
    These are ``top_p_mask`` and ``top_p_mask_insert``, and the comparison is
    deliberately not special-cased at the ends: at P = 100 the threshold is the
    minimum, so deletion still keeps whatever pixels sit exactly at it.
    """
    values = np.squeeze(attribution)
    threshold = np.percentile(values, 100 - percent)
    mask = values > threshold if keep_top else values <= threshold
    return mask.astype(np.uint8)


@torch.no_grad()
def logit_drop(
    model: torch.nn.Module,
    images: torch.Tensor,
    perturbed: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Per-sample drop in the ground truth logit after perturbation.

    This is the quantity summarised as ROAD-Delete: a faithful attribution
    should cause a large drop once its top-ranked pixels are removed.  Averaging
    is left to the caller so that partial batches are handled correctly -- the
    original script divided by a hard-coded 5000 regardless of dataset size.
    """
    index = torch.arange(images.shape[0], device=images.device)
    original = model(images)[index, targets]
    modified = model(perturbed)[index, targets]
    return original - modified


def noisy_linear_imputation(
    image: torch.Tensor,
    mask: torch.Tensor,
    iterations: int = 50,
    noise: float = 0.01,
) -> torch.Tensor:
    """Fill masked pixels by diffusing their neighbours (ROAD imputation).

    Rong et al. replace removed pixels with the solution of a linear system
    over the 8-neighbourhood rather than with a constant, which prevents the
    classifier from reading the mask shape itself as a feature.  The system is
    solved here with Jacobi iterations, which converges to the same fixed point
    without a sparse solver dependency.

    Args:
        image: (B, C, H, W) in [0, 1].
        mask: (B, 1, H, W), 1 for pixels to keep and 0 for pixels to impute.
        iterations: number of relaxation sweeps.
        noise: standard deviation of the noise added to the imputed values.
    """
    keep = mask.to(image.dtype)
    filled = image * keep

    # Weights of the 8-neighbourhood: diagonals count less, as in the original
    # formulation where they are scaled by 1/sqrt(2).
    diagonal = 1.0 / np.sqrt(2.0)
    kernel = torch.tensor(
        [[diagonal, 1.0, diagonal], [1.0, 0.0, 1.0], [diagonal, 1.0, diagonal]],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3)

    channels = image.shape[1]
    weight = kernel.expand(channels, 1, 3, 3).contiguous()
    single = kernel

    for _ in range(iterations):
        neighbour_sum = torch.nn.functional.conv2d(filled, weight, padding=1, groups=channels)
        neighbour_weight = torch.nn.functional.conv2d(
            torch.ones_like(keep), single, padding=1
        )
        average = neighbour_sum / neighbour_weight.clamp_min(1e-8)
        # Known pixels stay fixed; unknown ones take the neighbourhood average.
        filled = keep * image + (1.0 - keep) * average

    if noise > 0:
        perturbation = torch.randn_like(filled) * noise
        filled = filled + (1.0 - keep) * perturbation

    return filled.clamp(0.0, 1.0)

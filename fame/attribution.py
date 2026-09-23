"""Turning a FAME perturbation into a FAME attribution map.

This implements Eq. (4) of the paper and the post-processing that follows it:

    grayscale  ->  delta_x = |gray(x_adv) - gray(x)|  ->  Gaussian blur  ->  normalize

Note the order: ``get_difference`` converts both images to grey and *then*
subtracts.  Subtracting first and greying the result is not the same thing,
because the absolute value sits between the two steps -- one gives
``|sum_c w_c (a_c - b_c)|`` and the other ``sum_c w_c |a_c - b_c|``, which
differ whenever the colour channels move in opposite directions.

The original code base carried three slightly different copies of this
procedure (kernel 49 / sigma 7.7 for image classification, kernel 25 / sigma 5
for face recognition, plus a third copy inside the evaluation scripts that was
applied a *second* time to already blurred maps).  Everything now goes through
``fame_attribution`` so that a single set of parameters is used both when the
maps are produced and when they are evaluated.
"""

from typing import Optional

import torch
import torchvision

# Blur settings reported in Sec. 4 of the paper.  Sec. 4.4 discusses the effect
# of other choices; smaller kernels give finer, noisier maps.
DEFAULT_BLUR_KERNEL = 49
DEFAULT_BLUR_SIGMA = 7.7

# Luma weights.  These are torchvision's, which the original code reached
# through transforms.Grayscale; note the leading coefficient is 0.2989 rather
# than the 0.299 of ITU-R BT.601, and the FGGB evaluation used the latter.
_LUMA = torch.tensor([0.2989, 0.587, 0.114])


def to_grayscale(x: torch.Tensor) -> torch.Tensor:
    """Collapse a (B, C, H, W) batch to a single channel (B, 1, H, W)."""
    if x.shape[1] == 1:
        return x
    weights = _LUMA.to(dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x * weights).sum(dim=1, keepdim=True)


def normalize(x: torch.Tensor, mode: str = "minmax", eps: float = 1e-8) -> torch.Tensor:
    """Normalize each map in the batch to [0, 1].

    ``minmax`` rescales between the per-map minimum and maximum; ``max`` only
    divides by the maximum, as described in the paper.  The two differ whenever
    the smallest attribution is clearly above zero, which happens for heavily
    blurred maps.
    """
    maximum = x.amax(dim=(-2, -1), keepdim=True)
    if mode == "max":
        return x / (maximum + eps)
    if mode == "minmax":
        minimum = x.amin(dim=(-2, -1), keepdim=True)
        return (x - minimum) / (maximum - minimum + eps)
    raise ValueError(f"unknown normalization mode: {mode}")


def smooth_and_normalize(
    attribution: torch.Tensor,
    blur_kernel: Optional[int] = DEFAULT_BLUR_KERNEL,
    blur_sigma: float = DEFAULT_BLUR_SIGMA,
    normalization: str = "minmax",
) -> torch.Tensor:
    """Blur and normalize a map that is already single channel.

    Shared by FAME and by the baselines that the original code also smoothed:
    ``code_fggb.py`` blurs its maps with kernel 25 and sigma 5 before saving,
    while ``Corrise.py`` saves raw correlations.  Applying the right one at
    generation time keeps the evaluation free of method-specific handling.
    """
    if blur_kernel is not None:
        kernel = int(blur_kernel)
        if kernel % 2 == 0:
            kernel += 1
        attribution = torchvision.transforms.functional.gaussian_blur(
            attribution, kernel_size=[kernel, kernel], sigma=[blur_sigma, blur_sigma]
        )
    return normalize(attribution, mode=normalization)


def fame_attribution(
    image: torch.Tensor,
    image_adv: torch.Tensor,
    blur_kernel: Optional[int] = DEFAULT_BLUR_KERNEL,
    blur_sigma: float = DEFAULT_BLUR_SIGMA,
    normalization: str = "minmax",
) -> torch.Tensor:
    """Compute the FAME attribution map from an original/perturbed image pair.

    Args:
        image: original input, shape (B, C, H, W).
        image_adv: optimizer output for the same batch, same shape.
        blur_kernel: Gaussian kernel size in pixels, or ``None`` to skip the
            smoothing step and keep the raw perturbation field.
        blur_sigma: standard deviation of the Gaussian.
        normalization: see :func:`normalize`.

    Returns:
        Attribution maps of shape (B, 1, H, W) with values in [0, 1].
    """
    difference = (to_grayscale(image_adv.detach()) - to_grayscale(image.detach())).abs()
    # get_difference divides by the per-image maximum before lots_vis
    # normalizes again.  That looks redundant, and it almost is -- blur is
    # linear and min-max rescaling is scale invariant -- except that the
    # normalizer adds 1e-8 to its denominator, which is not.  Keeping the
    # division reproduces the original exactly; dropping it shifts every map by
    # around 3e-5.
    difference = difference / difference.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return smooth_and_normalize(difference, blur_kernel, blur_sigma, normalization)

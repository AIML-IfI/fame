"""FAME entry points for the three settings of Sec. 3.4.

All three are the same procedure -- run the optimizer, take the absolute perturbation,
blur, normalize -- and differ only in the loss that is descended:

    feature map      L_a    = |a[k]|          drive one activation to zero
    classification   L_cls  = z_o             suppress the class logit
    verification     L_+    = s               remove the evidence for a match
                     L_-    = 1 - s           remove the evidence against it
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from .attribution import DEFAULT_BLUR_KERNEL, DEFAULT_BLUR_SIGMA, fame_attribution
from .losses import classification_loss, feature_map_loss, verification_loss
from .optimizer import fame
from .models.wrappers import FeatureMapExtractor


@dataclass
class FameConfig:
    """Hyper-parameters shared by every FAME variant.

    The defaults are the settings reported in Sec. 4: eta = 1/255 and 500
    iterations, with a Gaussian blur of sigma 7.7.  The supplemental material
    shows that 75-200 iterations already give stable maps, so ``iterations`` is
    the first knob to turn when runtime matters.

    ``clamp`` is off by default so that the updates stay unconstrained, as in
    the original implementation.  Set it to ``(0.0, 1.0)`` only when the images
    handed to FAME are raw pixels rather than normalized tensors.
    """

    step_size: float = 1.0 / 255.0
    iterations: int = 500
    epsilon: Optional[float] = None
    blur_kernel: Optional[int] = DEFAULT_BLUR_KERNEL
    blur_sigma: float = DEFAULT_BLUR_SIGMA
    normalization: str = "minmax"
    clamp: Optional[tuple] = None


def _run(image: torch.Tensor, loss_fn, config: FameConfig) -> torch.Tensor:
    image_adv = fame(
        image,
        loss_fn,
        step_size=config.step_size,
        iterations=config.iterations,
        epsilon=config.epsilon,
        clamp=config.clamp,
    )
    return fame_attribution(
        image,
        image_adv,
        blur_kernel=config.blur_kernel,
        blur_sigma=config.blur_sigma,
        normalization=config.normalization,
    )


def explain_classification(
    model: torch.nn.Module,
    image: torch.Tensor,
    class_indices: torch.Tensor,
    config: Optional[FameConfig] = None,
) -> torch.Tensor:
    """Attribution for the logit of a chosen class.

    Args:
        model: classifier taking [0, 1] images of shape (B, 3, H, W).
        image: input batch.
        class_indices: one class index per batch element.
        config: hyper-parameters; defaults to :class:`FameConfig`.

    Returns:
        Attribution maps of shape (B, 1, H, W) in [0, 1].
    """
    config = config or FameConfig()
    return _run(image, classification_loss(model, class_indices), config)


def explain_verification(
    model: torch.nn.Module,
    image: torch.Tensor,
    reference_image: torch.Tensor,
    config: Optional[FameConfig] = None,
    modes: Sequence[str] = ("similar", "dissimilar"),
) -> dict:
    """Attribution for one side of a verification pair.

    The reference embedding is computed once and frozen, so the returned maps
    explain ``image`` only.  Call twice with the roles swapped to explain both
    the gallery and the probe.

    Args:
        model: embedding network taking [0, 1] images.
        image: the image being explained, shape (B, 3, H, W).
        reference_image: the fixed image of the pair, broadcastable to ``image``.
        modes: which of ``"similar"`` (e_+) and ``"dissimilar"`` (e_-) to compute.

    Returns:
        Dict mapping each requested mode to maps of shape (B, 1, H, W).
    """
    config = config or FameConfig()
    with torch.no_grad():
        reference_embedding = model(reference_image).detach()

    return {
        mode: _run(image, verification_loss(model, reference_embedding, mode=mode), config)
        for mode in modes
    }


def explain_feature_map(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    image: torch.Tensor,
    locations: Optional[Sequence[Tuple[int, int]]] = None,
    config: Optional[FameConfig] = None,
    batch_size: int = 16,
) -> torch.Tensor:
    """Attribution for every location of the feature map (Sec. 4.1).

    Used to test the CAM assumption that a[k] only sees the pixels underneath
    it.  A single image is replicated across the batch so that several
    locations are optimized in parallel; batch elements are independent because
    the optimizer normalizes the gradient per sample.

    Args:
        model: network taking [0, 1] images.
        target_layer: module whose output is the feature map ``a``.
        image: a single image of shape (1, 3, H, W) or (3, H, W).
        locations: feature map positions to explain.  Defaults to all of them.
        batch_size: how many locations to optimize at once.

    Returns:
        Maps of shape (len(locations), 1, H, W), ordered like ``locations``.
    """
    config = config or FameConfig()
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.shape[0] != 1:
        raise ValueError("explain_feature_map handles one image at a time")

    with FeatureMapExtractor(model, target_layer) as extractor:
        if locations is None:
            with torch.no_grad():
                _, _, height, width = extractor(image).shape
            locations = [(row, col) for row in range(height) for col in range(width)]

        maps = []
        for start in range(0, len(locations), batch_size):
            chunk = list(locations[start : start + batch_size])
            batch = image.expand(len(chunk), -1, -1, -1).contiguous()
            maps.append(_run(batch, feature_map_loss(extractor, chunk), config))

    return torch.cat(maps, dim=0)

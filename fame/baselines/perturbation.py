"""Perturbation-based baseline: CorrRISE (Lu et al., WACV 2024).

The module is named for the family rather than the method, so that it does not
shadow the exported ``corrise`` function on the package.

Random black patches are pasted over the image, the similarity to the fixed
partner is recomputed for each mask, and the correlation between a pixel's
occlusion pattern and the resulting scores gives its attribution.  Positive
correlation means keeping the pixel raises similarity, so it supports the
match.

``Corrise.py`` rank-transforms the scores before correlating, which makes this
Spearman rather than Pearson and stops a few extreme scores from dominating;
``rank_based=False`` gives the plain Pearson version.  Ranking the masks as the
original also does is a no-op, since they are binary and ranking maps 0 and 1
onto two constants, which correlation is invariant to.

Parameters follow ``Corrise.py``: 500 masks, each with 3 patches of 30x30
pixels.  That file also builds the mask set once at import time and reuses it
for every pair, so ``masks`` can be passed in here to get the same behaviour --
sharing one set removes mask sampling as a source of variance between images.
"""

from typing import Dict, Optional

import torch


def random_masks(
    count: int,
    height: int,
    width: int,
    patches: int = 3,
    patch_size: int = 30,
    generator: Optional[torch.Generator] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate (count, 1, H, W) masks that are 0 inside patches and 1 elsewhere.

    Patch origins are drawn from ``[0, H - patch_size]`` so that every patch
    fits entirely inside the image, matching ``generate_mask``.  Drawing from
    the full range and letting patches run off the edge would occlude border
    pixels less often and bias their correlation.
    """
    masks = torch.ones(count, 1, height, width, device=device)
    for mask in masks:
        rows = torch.randint(0, height - patch_size + 1, (patches,), generator=generator, device=device)
        cols = torch.randint(0, width - patch_size + 1, (patches,), generator=generator, device=device)
        for row, col in zip(rows.tolist(), cols.tolist()):
            mask[0, row : row + patch_size, col : col + patch_size] = 0.0
    return masks


@torch.no_grad()
def corrise(
    model: torch.nn.Module,
    image: torch.Tensor,
    reference_image: torch.Tensor,
    num_masks: int = 500,
    patches: int = 3,
    patch_size: int = 30,
    batch_size: int = 250,
    seed: Optional[int] = None,
    rank_based: bool = True,
    masks: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """CorrRISE attribution for one side of a verification pair.

    Args:
        masks: optional (N, 1, H, W) mask set to reuse across images, as the
            original does.  When given, ``num_masks``, ``patches``,
            ``patch_size`` and ``seed`` are ignored.

    Returns:
        Dict with ``similar`` (e_+) and ``dissimilar`` (e_-), each (1, H, W).
    """
    device = image.device
    _, _, height, width = image.shape

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    reference = model(reference_image)
    reference = reference / reference.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)

    if masks is None:
        masks = random_masks(
            num_masks, height, width, patches, patch_size, generator=generator, device=device
        )
    else:
        masks = masks.to(device)
        num_masks = masks.shape[0]

    scores = []
    for start in range(0, num_masks, batch_size):
        chunk = masks[start : start + batch_size]
        occluded = image * chunk
        embedding = model(occluded)
        embedding = embedding / embedding.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        scores.append((embedding * reference).sum(dim=1))
    scores = torch.cat(scores)

    if rank_based:
        # Average ranks, so that tied scores are handled like scipy's rankdata.
        order = scores.argsort()
        ranks = torch.empty_like(scores)
        ranks[order] = torch.arange(1, scores.numel() + 1, dtype=scores.dtype, device=device)
        scores = ranks

    # Correlate the per-pixel keep indicator with the similarity, for all pixels
    # at once.  A pixel that is kept when the score is high correlates
    # positively and therefore supports the match.
    keep = masks[:, 0]
    keep_centered = keep - keep.mean(dim=0, keepdim=True)
    scores_centered = (scores - scores.mean()).view(-1, 1, 1)

    covariance = (keep_centered * scores_centered).mean(dim=0)
    denominator = keep_centered.pow(2).mean(dim=0).sqrt() * scores_centered.pow(2).mean().sqrt()
    correlation = covariance / denominator.clamp_min(1e-8)
    # Pixels never occluded have zero variance and produce NaN, as in the
    # original, which replaces them with zero.
    correlation = torch.nan_to_num(correlation, nan=0.0)

    return {
        "similar": correlation.clamp(min=0.0).unsqueeze(0),
        "dissimilar": (-correlation).clamp(min=0.0).unsqueeze(0),
    }

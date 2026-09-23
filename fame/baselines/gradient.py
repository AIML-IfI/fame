"""Gradient-based baseline: Feature-Guided Gradient Backpropagation (Lu et al., FG 2024).

The module is named for the family rather than the method, so that it does not
shadow the exported ``fggb`` function on the package.

Each embedding dimension is backpropagated to the input separately, the
resulting maps are normalized individually and then combined with the weights
v_i - theta, where v = phi_g * phi_p.  Thresholding the weighted sum at zero
splits it into e_+ and e_-.

The threshold is distributed over the embedding, ``v_i - theta / D``, which is
what ``code_fggb.py`` does and the only scaling that makes sense: the v_i sum
to the cosine similarity, so each is on the order of ``cos / D`` (about 0.0015
for a 512-D embedding).  Subtracting a whole EER threshold of ~0.3 from every
v_i would drive all the weights negative, leaving e_+ empty and e_- covering
the entire image.  ``per_dimension=True`` selects that degenerate variant and
exists only so the difference can be demonstrated.

Relative to the original, ``torch.autograd.grad`` replaces the manual
``zero_grad`` / ``backward`` bookkeeping, and gradients are taken for one image
per call rather than both at once; the caller loops over the two sides, which
costs the same number of backward passes.
"""

from typing import Dict

import torch


def _embedding_gradients(model: torch.nn.Module, image: torch.Tensor, eps: float = 1e-8):
    """Gradient of every embedding dimension with respect to the input.

    The embedding is L2-normalized *before* differentiating, so the gradients
    run through the normalization exactly as in ``code_fggb.py``, where
    ``_normalize_embedding`` is applied and then ``f_a[0, k].backward()`` is
    called.  Differentiating the raw embedding instead gives a different map,
    because the normalization couples the dimensions.

    Returns a tensor of shape (D, H, W): absolute gradients averaged over the
    colour channels and L2-normalized per dimension, as FGGB prescribes.
    """
    image = image.clone().detach().requires_grad_(True)
    raw = model(image)
    embedding = (raw / (raw.norm(p=2, dim=1, keepdim=True) + eps))[0]
    dimensions = embedding.shape[0]

    maps = []
    for index in range(dimensions):
        (gradient,) = torch.autograd.grad(
            embedding[index], image, retain_graph=index < dimensions - 1
        )
        gradient = gradient.detach().abs().mean(dim=1)[0]
        maps.append(gradient / (gradient.norm(p=2) + 1e-8))

    return torch.stack(maps, dim=0)


def fggb(
    model: torch.nn.Module,
    image: torch.Tensor,
    reference_image: torch.Tensor,
    threshold: float = 0.0,
    per_dimension: bool = False,
) -> Dict[str, torch.Tensor]:
    """FGGB attribution for one side of a verification pair.

    Args:
        model: embedding network taking [0, 1] images.
        image: the image being explained, shape (1, 3, H, W).
        reference_image: the other image of the pair.
        threshold: theta, normally the EER threshold of the model on the
            protocol being explained.  ``code_fggb.py`` reads it per
            (model, dataset, protocol) from a precomputed scores table.
        per_dimension: subtract theta from every v_i instead of distributing it
            over the embedding.  Degenerate for realistic thresholds; see the
            module docstring.

    Returns:
        Dict with ``similar`` (e_+) and ``dissimilar`` (e_-), each (1, H, W).
    """
    with torch.no_grad():
        reference = model(reference_image)
        reference = reference / (reference.norm(p=2, dim=1, keepdim=True) + 1e-8)
        current = model(image)
        current = current / (current.norm(p=2, dim=1, keepdim=True) + 1e-8)

    weights = (reference * current)[0]
    # v_i - theta / D: theta is compared against the cosine similarity, which is
    # the sum of the v_i, so it has to be spread across the dimensions.
    weights = weights - (threshold if per_dimension else threshold / weights.shape[0])

    gradients = _embedding_gradients(model, image)
    weighted = (gradients * weights[:, None, None]).sum(dim=0, keepdim=True)

    return {
        "similar": weighted.clamp(min=0.0),
        "dissimilar": (-weighted).clamp(min=0.0),
    }

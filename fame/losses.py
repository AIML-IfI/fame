"""Loss functions that specialise FAME to a task.

Each factory returns a callable ``loss_fn(x) -> (B,)`` suitable for
:func:`fame.fame.fame`.  The optimizer always *descends* the returned loss, so the sign
of each loss already encodes the question being asked.
"""

from typing import Sequence

import torch

from .models.wrappers import FeatureMapExtractor


def feature_map_loss(extractor: FeatureMapExtractor, locations: Sequence[tuple]):
    """L_a from Eq. (5): drive one feature map location towards zero.

    Answers "which input pixels carry information that reaches a[k]?".  Each
    batch element is optimized for its own location, so a whole feature map can
    be visualized in one batched pass by repeating the image B times and
    passing B different locations.

    The channel reduction is a sum of absolute values, which is the L1 norm
    that Eq. (5) writes and what the original (now lost) script used.  It is
    not interchangeable with the alternatives: a plain sum cancels positive
    against negative activations, and an L2 norm weights large channels more
    heavily.  The three agree only when every channel is non-negative, as after
    a ReLU, and diverge for layers that can output negative values -- which
    includes the ConvNeXt blocks used as a target layer here.

    Args:
        extractor: wrapper exposing the target layer's activation.
        locations: one ``(row, col)`` pair per batch element.
    """
    rows = torch.tensor([loc[0] for loc in locations], dtype=torch.long)
    cols = torch.tensor([loc[1] for loc in locations], dtype=torch.long)

    def loss_fn(x: torch.Tensor) -> torch.Tensor:
        activation = extractor(x)  # (B, C, H_a, W_a)
        index = torch.arange(activation.shape[0], device=activation.device)
        # Select a[k] per sample, penalizing that location only and leaving the
        # rest of the feature map free to change.
        selected = activation[index, :, rows.to(activation.device), cols.to(activation.device)]
        return selected.abs().sum(dim=1)

    return loss_fn


def classification_loss(
    model: torch.nn.Module, class_indices: torch.Tensor, absolute: bool = False
):
    """L_cls: the logit of the class of interest.

    Descending it asks "which pixels have to change to stop the network from
    predicting this class?".  Sec. 3.4 defines the loss as the raw logit z_o,
    which is the default here.

    ``code_fame.py`` instead minimizes ``L1Loss(z_o, 0)``, i.e. |z_o|, which is
    ``absolute=True``.  The two are identical whenever the logit is positive --
    the usual case for the ground truth class of a correctly classified image
    -- but they move in opposite directions once it turns negative, which
    happens for the misclassified images in the protocol.
    """

    def loss_fn(x: torch.Tensor) -> torch.Tensor:
        logits = model(x)
        index = torch.arange(logits.shape[0], device=logits.device)
        selected = logits[index, class_indices.to(logits.device)]
        return selected.abs() if absolute else selected

    return loss_fn


def cosine_similarity(
    embedding_a: torch.Tensor,
    embedding_b: torch.Tensor,
    detach_norms: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cosine similarity of Eq. (2), computed row-wise over a batch.

    Gradients flow through the denominators by default, matching
    ``Cosine_Loss`` and ``compute_norm`` in the original ``utils.py``, where
    the norms are ordinary differentiable operations.  ``detach_norms=True``
    freezes them instead, so the perturbation is shaped only by the direction
    of the embedding and not by its length; that is the variant the CAM target
    ``CosineDistanceTarget`` uses, since it precomputes the norm product from
    the clean images.
    """
    norm_a = embedding_a.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    norm_b = embedding_b.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    if detach_norms:
        norm_a = norm_a.detach()
        norm_b = norm_b.detach()
    return ((embedding_a / norm_a) * (embedding_b / norm_b)).sum(dim=-1)


def verification_loss(
    model: torch.nn.Module,
    reference_embedding: torch.Tensor,
    mode: str = "similar",
    detach_norms: bool = False,
):
    """L_+ and L_- from Eq. (6).

    ``similar`` minimizes s, so the perturbation removes the evidence that
    makes the pair look alike and e_+ marks the regions supporting the match.
    ``dissimilar`` minimizes 1 - s, i.e. maximizes similarity, and e_- marks
    the regions that currently push the pair apart.  Unlike FGGB, neither needs
    a threshold to separate the two.  These are ``Cosine_Loss`` and
    ``myCosine_Loss`` in the original ``utils.py``.

    Args:
        model: embedding network applied to the image being explained.
        reference_embedding: embedding of the fixed image of the pair, shape
            (1, D) or (B, D).  It must already be detached -- keeping the other
            side of the pair frozen is what makes the attribution one-sided.
        mode: ``"similar"`` for e_+ or ``"dissimilar"`` for e_-.
    """
    if mode not in ("similar", "dissimilar"):
        raise ValueError(f"unknown mode: {mode}")
    reference = reference_embedding.detach()

    def loss_fn(x: torch.Tensor) -> torch.Tensor:
        embedding = model(x)
        reference_batch = reference.expand_as(embedding)
        similarity = cosine_similarity(reference_batch, embedding, detach_norms)
        return similarity if mode == "similar" else 1.0 - similarity

    return loss_fn

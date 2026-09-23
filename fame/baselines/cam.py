"""CAM baselines, wrapping ``pytorch_grad_cam``.

The original repository had eighteen ``cam_<dataset>_<target>[_partial][_adaface].py``
files that differed only in which similarity target was built and whether the
input needed a channel swap.  Both axes are parameters here, so one function
covers every combination.
"""

from typing import Sequence

import numpy as np
import torch

# Names follow the paper: Grad-CAM-EW is pytorch_grad_cam's GradCAMElementWise.
CAM_METHODS = {
    "GradCAM": "GradCAM",
    "GradCAMElementWise": "GradCAMElementWise",
    "HiResCAM": "HiResCAM",
    "FullGrad": "FullGrad",
}


def _resolve(method: str):
    import pytorch_grad_cam

    if method not in CAM_METHODS:
        raise KeyError(f"unknown CAM method {method!r}, expected one of {list(CAM_METHODS)}")
    return getattr(pytorch_grad_cam, CAM_METHODS[method])


class DotProductTarget:
    """Backpropagate phi_g^T phi_p, the unnormalized similarity.

    Zhu et al. use this for metric learning so that the cosine denominators do
    not contribute gradients.
    """

    def __init__(self, reference: torch.Tensor):
        self.reference = reference

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        return torch.dot(self.reference, model_output)


class CosineSimilarityTarget:
    """Backpropagate the cosine similarity itself.

    ``detach_norms=True`` keeps the denominators constant, which is what
    ``CosineDistanceTarget`` does: it receives a ``norm_term`` computed once
    from the clean gallery and probe embeddings.  Since CAM evaluates the model
    on the same image, the two are the same function of the output.  Setting it
    to False reproduces ``CosineDistanceTargetwithGradient`` instead, which
    uses ``torch.nn.CosineSimilarity`` and lets the gradient flow through the
    embedding lengths.
    """

    def __init__(self, reference: torch.Tensor, detach_norms: bool = True, eps: float = 1e-8):
        self.reference = reference
        self.detach_norms = detach_norms
        self.eps = eps

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        norm_reference = self.reference.norm(p=2).clamp_min(self.eps)
        norm_output = model_output.norm(p=2).clamp_min(self.eps)
        if self.detach_norms:
            norm_reference = norm_reference.detach()
            norm_output = norm_output.detach()
        return torch.dot(self.reference, model_output) / (norm_reference * norm_output)


def classification_cam(
    model: torch.nn.Module,
    target_layers: Sequence[torch.nn.Module],
    image: torch.Tensor,
    class_indices: Sequence[int],
    method: str = "GradCAM",
) -> np.ndarray:
    """CAM attribution for a chosen class.

    Returns maps of shape (B, H, W), already upsampled to input resolution by
    ``pytorch_grad_cam``.
    """
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    cam_class = _resolve(method)
    targets = [ClassifierOutputTarget(int(index)) for index in class_indices]
    with cam_class(model=model, target_layers=list(target_layers)) as cam:
        return cam(input_tensor=image, targets=targets)


def verification_cam(
    model: torch.nn.Module,
    target_layers: Sequence[torch.nn.Module],
    image: torch.Tensor,
    reference_image: torch.Tensor,
    method: str = "GradCAM",
    target: str = "cosine",
) -> np.ndarray:
    """CAM attribution for one side of a verification pair.

    Args:
        image: the image being explained, shape (1, 3, H, W).
        reference_image: the other image of the pair, kept fixed.
        target: ``"cosine"`` or ``"dot"``.
    """
    with torch.no_grad():
        reference = model(reference_image)[0].detach()

    if target == "cosine":
        targets = [CosineSimilarityTarget(reference)]
    elif target == "dot":
        targets = [DotProductTarget(reference)]
    else:
        raise ValueError(f"unknown target {target!r}")

    cam_class = _resolve(method)
    with cam_class(model=model, target_layers=list(target_layers)) as cam:
        return cam(input_tensor=image, targets=targets)

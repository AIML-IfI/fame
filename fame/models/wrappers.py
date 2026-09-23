"""Wrappers that give every backbone the same interface for FAME.

:class:`NormalizedModel` folds mean/std normalization into the forward pass so
that a model can be driven with raw [0, 1] images.  Which space FAME perturbs is
a deliberate choice, not an implementation detail:

* **Normalized space** -- normalize the batch first and hand the normalized
  tensor to FAME with a bare backbone.  A step of 1/255 is then measured in
  normalized units, so its size in pixels differs per colour channel.  This is
  what the reported results use, and it is the default in the scripts.
* **Pixel space** -- hand FAME a [0, 1] image wrapped in
  :class:`NormalizedModel`.  A step of 1/255 is then exactly one grey level on
  every channel.

Face models take the second route either way, because their normalization is
a single shared 0.5/0.5 and so is uniform across channels.

:class:`FeatureMapExtractor` reaches intermediate activations with a forward
hook, which keeps the same code working for ResNet, VGG, ConvNeXt and IResNet.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# AdaFace and the other IResNet face models are trained on inputs scaled to
# [-1, 1], i.e. mean and std of 0.5 on every channel.
FACE_MEAN = (0.5, 0.5, 0.5)
FACE_STD = (0.5, 0.5, 0.5)


class NormalizedModel(nn.Module):
    """Prepend mean/std normalization to a backbone.

    The wrapped module therefore consumes images in [0, 1] and FAME can perturb
    exactly what the user sees.
    """

    def __init__(self, model: nn.Module, mean: Sequence[float], std: Sequence[float]):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


class FeatureMapExtractor(nn.Module):
    """Expose the activation of one intermediate layer.

    Calling the extractor runs the network and returns the target layer's
    output ``a`` of shape (B, C_a, H_a, W_a).  A hook is used rather than
    slicing the module list so that the same code works for ResNet, VGG,
    ConvNeXt and the IResNet face backbones.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        super().__init__()
        self.model = model
        self._activation: Optional[torch.Tensor] = None
        self._handle = target_layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        # Keep the tensor attached to the graph -- FAME differentiates it.
        self._activation = output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._activation = None
        self.model(x)
        if self._activation is None:
            raise RuntimeError("target layer was not reached during the forward pass")
        return self._activation

    def close(self) -> None:
        """Remove the hook.  Leaving hooks attached leaks memory across runs."""
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FaceEmbeddingModel(nn.Module):
    """Face recognition backbone returning embeddings for [0, 1] inputs.

    ``l2_normalize`` reproduces the ``adaface_norm`` variant of the original
    code.  It is off by default because FAME normalizes inside the similarity
    (see :func:`fame.losses.cosine_similarity`), so normalizing twice only
    changes the gradient magnitude, not the direction.
    """

    def __init__(self, backbone: nn.Module, l2_normalize: bool = False):
        super().__init__()
        self.backbone = NormalizedModel(backbone, FACE_MEAN, FACE_STD)
        self.l2_normalize = l2_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.backbone(x)
        if self.l2_normalize:
            embedding = embedding / embedding.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        return embedding

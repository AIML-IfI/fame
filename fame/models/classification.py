"""Pretrained image classification backbones used in the paper.

Also resolves the last convolutional layer per architecture.  The original
``code_cam.py`` hard-coded ``model.features[-1][-1].block[-1]``, which is a
ConvNeXt path and raises ``TypeError`` on ResNet and ``IndexError`` on VGG, so
CAM baselines could only be run one architecture at a time.
"""

from typing import Callable, Dict, Tuple

import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    VGG19_Weights,
    convnext_tiny,
    resnet34,
    resnet50,
    resnet101,
    vgg19,
)

from .wrappers import IMAGENET_MEAN, IMAGENET_STD, NormalizedModel

CLASSIFIERS: Dict[str, Tuple[Callable, object]] = {
    "ResNet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1),
    "ResNet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
    "ResNet101": (resnet101, ResNet101_Weights.IMAGENET1K_V2),
    "VGG19": (vgg19, VGG19_Weights.IMAGENET1K_V1),
    "ConvNeXt_Tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
}


def build_classifier(name: str, device: str = "cuda", normalized: bool = True) -> nn.Module:
    """Load a pretrained classifier in eval mode.

    With ``normalized=True`` the returned module takes [0, 1] images, which is
    what FAME needs; pass ``False`` when handing the model to an external CAM
    implementation that does its own preprocessing.
    """
    if name not in CLASSIFIERS:
        raise KeyError(f"unknown classifier {name!r}, expected one of {list(CLASSIFIERS)}")
    constructor, weights = CLASSIFIERS[name]
    model = constructor(weights=weights)
    if normalized:
        model = NormalizedModel(model, IMAGENET_MEAN, IMAGENET_STD)
    return model.to(device).eval()


def target_layer(model: nn.Module, name: str) -> nn.Module:
    """Return the last convolutional block, i.e. the layer producing ``a``.

    Accepts models with or without the :class:`NormalizedModel` wrapper.
    """
    backbone = model.model if isinstance(model, NormalizedModel) else model

    if name.startswith("ResNet"):
        return backbone.layer4[-1]
    if name.startswith("VGG"):
        # features[-1] is the final MaxPool; the convolution before it is the
        # last layer with spatial semantics.
        return backbone.features[-2]
    if name.startswith("ConvNeXt"):
        return backbone.features[-1][-1]
    raise KeyError(f"no target layer rule for {name!r}")

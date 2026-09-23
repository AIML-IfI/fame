"""Tests for backbone wiring.

Weights are never downloaded: the architectures are built untrained, which is
enough to check that target layers resolve and that the normalization wrappers
do what they claim.
"""

import pytest
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, resnet34, resnet50, resnet101, vgg19

from fame.models import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    FaceEmbeddingModel,
    FeatureMapExtractor,
    NormalizedModel,
)
from fame.models.classification import target_layer

ARCHITECTURES = {
    "ResNet34": resnet34,
    "ResNet50": resnet50,
    "ResNet101": resnet101,
    "VGG19": vgg19,
    "ConvNeXt_Tiny": convnext_tiny,
}


@pytest.mark.parametrize("name", list(ARCHITECTURES))
def test_target_layer_resolves_to_a_spatial_feature_map(name):
    """Regression test for the hard-coded ConvNeXt path in the original code.

    That path raised on ResNet and VGG, so only one architecture could be run.
    """
    model = NormalizedModel(ARCHITECTURES[name](weights=None), IMAGENET_MEAN, IMAGENET_STD).eval()
    layer = target_layer(model, name)

    with FeatureMapExtractor(model, layer) as extractor, torch.no_grad():
        activation = extractor(torch.rand(1, 3, 224, 224))

    assert activation.dim() == 4
    assert activation.shape[2] > 1 and activation.shape[3] > 1
    assert activation.shape[1] >= 512


def test_target_layer_rejects_an_unknown_architecture():
    with pytest.raises(KeyError):
        target_layer(nn.Identity(), "SomeNet")


def test_normalized_model_applies_the_statistics():
    inner = nn.Identity()
    model = NormalizedModel(inner, IMAGENET_MEAN, IMAGENET_STD)
    image = torch.rand(1, 3, 8, 8)

    expected = (image - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / torch.tensor(
        IMAGENET_STD
    ).view(1, 3, 1, 1)

    assert torch.allclose(model(image), expected, atol=1e-6)


def test_face_wrapper_normalizes_to_the_training_range():
    """AdaFace is trained on [-1, 1]; the original forward dropped this step."""
    model = FaceEmbeddingModel(nn.Identity())

    output = model(torch.tensor([[[[0.0]], [[0.5]], [[1.0]]]]))

    assert torch.allclose(output.flatten(), torch.tensor([-1.0, 0.0, 1.0]), atol=1e-6)


def test_face_wrapper_l2_normalizes_when_asked():
    backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 8))
    image = torch.rand(2, 3, 4, 4)

    plain = FaceEmbeddingModel(backbone, l2_normalize=False)(image)
    normalized = FaceEmbeddingModel(backbone, l2_normalize=True)(image)

    assert not torch.allclose(plain.norm(dim=1), torch.ones(2), atol=1e-3)
    assert torch.allclose(normalized.norm(dim=1), torch.ones(2), atol=1e-5)


def test_extractor_raises_when_the_layer_is_never_reached():
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 4))
    orphan = nn.Conv2d(3, 3, 1)

    with FeatureMapExtractor(model, orphan) as extractor:
        with pytest.raises(RuntimeError, match="not reached"):
            extractor(torch.rand(1, 3, 2, 2))


@pytest.mark.parametrize("name", ["ResNet34", "ResNet50", "ResNet101"])
def test_resnets_give_the_49_location_feature_map(name):
    """The feature map experiment is run on the ResNets, which are 7x7 at 224.

    Each image therefore yields 49 attribution maps that tile into a 7x7 sheet
    mirroring the feature map itself.
    """
    model = NormalizedModel(ARCHITECTURES[name](weights=None), IMAGENET_MEAN, IMAGENET_STD).eval()
    layer = target_layer(model, name)

    with FeatureMapExtractor(model, layer) as extractor, torch.no_grad():
        activation = extractor(torch.rand(1, 3, 224, 224))

    assert activation.shape[2:] == (7, 7)
    assert activation.shape[2] * activation.shape[3] == 49


def test_resnet_target_layer_output_is_non_negative():
    """layer4[-1] ends in a ReLU, so a[k] has no negative channels.

    That makes the absolute sum in L_a equal to a plain sum for these networks,
    even though the two differ in general.
    """
    model = NormalizedModel(ARCHITECTURES["ResNet34"](weights=None), IMAGENET_MEAN, IMAGENET_STD).eval()
    layer = target_layer(model, "ResNet34")

    with FeatureMapExtractor(model, layer) as extractor, torch.no_grad():
        activation = extractor(torch.rand(1, 3, 224, 224))

    assert activation.min() >= 0
    assert torch.allclose(activation.abs().sum(), activation.sum())


def test_vgg_does_not_give_a_7x7_map():
    """VGG19 produces 14x14, which is why the sweep warns about it."""
    model = NormalizedModel(ARCHITECTURES["VGG19"](weights=None), IMAGENET_MEAN, IMAGENET_STD).eval()

    with FeatureMapExtractor(model, target_layer(model, "VGG19")) as extractor, torch.no_grad():
        activation = extractor(torch.rand(1, 3, 224, 224))

    assert activation.shape[2:] == (14, 14)

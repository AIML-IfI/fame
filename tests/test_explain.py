"""Tests for the three task entry points of Sec. 3.4."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from fame import FameConfig, explain_classification, explain_feature_map, explain_verification
from fame.losses import cosine_similarity, verification_loss
from fame.models import FeatureMapExtractor

FAST = FameConfig(iterations=15, blur_kernel=11, blur_sigma=3.0)


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)
    np.random.seed(0)


class TinyNet(nn.Module):
    """Small convolutional stack with a reachable feature map."""

    def __init__(self, outputs: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, outputs))

    def forward(self, x):
        return self.head(self.features(x))


class TinyEmbedding(nn.Module):
    """Stand-in for a face backbone.

    It flattens the feature map rather than pooling it, as the face networks in
    Sec. 3 do.  Pooling averages away most of the spatial variation, which
    drives the cosine similarity of two random inputs to ~0.999 and leaves no
    headroom to test that L_- increases it.
    """

    def __init__(self, dimensions: int = 16, size: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        reduced = size // 4
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16 * reduced * reduced, dimensions))

    def forward(self, x):
        return self.head(self.features(x))


def test_classification_map_shape_and_range():
    model = TinyNet().eval()
    images = torch.rand(3, 3, 32, 32)

    maps = explain_classification(model, images, torch.tensor([1, 4, 9]), FAST)

    assert maps.shape == (3, 1, 32, 32)
    assert maps.min() >= 0.0 and maps.max() <= 1.0


def test_classification_map_depends_on_the_target_class():
    """Explaining a different class must produce a different map."""
    model = TinyNet().eval()
    image = torch.rand(1, 3, 32, 32)

    first = explain_classification(model, image, torch.tensor([0]), FAST)
    second = explain_classification(model, image, torch.tensor([7]), FAST)

    assert not torch.allclose(first, second, atol=1e-3)


def test_verification_returns_both_polarities():
    model = TinyEmbedding().eval()
    probe, gallery = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)

    maps = explain_verification(model, probe, gallery, FAST)

    assert set(maps) == {"similar", "dissimilar"}
    for value in maps.values():
        assert value.shape == (1, 1, 32, 32)
        assert value.min() >= 0.0 and value.max() <= 1.0


def test_verification_modes_move_similarity_in_opposite_directions():
    """L_+ must lower the similarity and L_- must raise it (Eq. 6)."""
    model = TinyEmbedding().eval()
    probe, gallery = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)

    with torch.no_grad():
        reference = model(gallery)
        start = cosine_similarity(reference, model(probe)).item()

    from fame.optimizer import fame

    lowered = fame(probe, verification_loss(model, reference, mode="similar"), iterations=40)
    raised = fame(probe, verification_loss(model, reference, mode="dissimilar"), iterations=40)
    with torch.no_grad():
        after_lowering = cosine_similarity(reference, model(lowered)).item()
        after_raising = cosine_similarity(reference, model(raised)).item()

    assert after_lowering < start
    assert after_raising > start


def test_verification_only_explains_the_image_it_is_given():
    """Swapping the roles of the pair must change the map."""
    model = TinyEmbedding().eval()
    probe, gallery = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)

    explaining_probe = explain_verification(model, probe, gallery, FAST, modes=("similar",))
    explaining_gallery = explain_verification(model, gallery, probe, FAST, modes=("similar",))

    assert not torch.allclose(explaining_probe["similar"], explaining_gallery["similar"], atol=1e-3)


def test_verification_selected_modes_are_respected():
    model = TinyEmbedding().eval()
    probe, gallery = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)

    maps = explain_verification(model, probe, gallery, FAST, modes=("similar",))

    assert set(maps) == {"similar"}


def test_feature_map_covers_every_location_by_default():
    model = TinyNet().eval()
    image = torch.rand(1, 3, 32, 32)
    layer = model.features[-2]

    with FeatureMapExtractor(model, layer) as extractor, torch.no_grad():
        _, _, height, width = extractor(image).shape

    maps = explain_feature_map(model, layer, image, config=FAST, batch_size=4)

    assert maps.shape == (height * width, 1, 32, 32)


def test_feature_map_respects_the_requested_order():
    model = TinyNet().eval()
    image = torch.rand(1, 3, 32, 32)
    locations = [(0, 0), (3, 3), (1, 2)]

    maps = explain_feature_map(model, model.features[-2], image, locations, FAST, batch_size=2)
    single = explain_feature_map(model, model.features[-2], image, [(3, 3)], FAST)

    assert maps.shape[0] == 3
    assert torch.allclose(maps[1], single[0], atol=1e-5)


def test_feature_map_locations_differ_from_each_other():
    """Different locations must attribute to different pixels.

    This is the measurement behind Sec. 4.1; if every location produced the
    same map the receptive field argument would not be testable.
    """
    model = TinyNet().eval()
    image = torch.rand(1, 3, 32, 32)

    maps = explain_feature_map(model, model.features[-2], image, [(0, 0), (7, 7)], FAST)

    assert not torch.allclose(maps[0], maps[1], atol=1e-3)


def test_feature_map_rejects_a_batch():
    model = TinyNet().eval()

    with pytest.raises(ValueError):
        explain_feature_map(model, model.features[-2], torch.rand(2, 3, 32, 32), config=FAST)


def test_extractor_hook_is_removed_on_exit():
    model = TinyNet().eval()
    layer = model.features[-2]
    before = len(layer._forward_hooks)

    with FeatureMapExtractor(model, layer):
        assert len(layer._forward_hooks) == before + 1

    assert len(layer._forward_hooks) == before


def test_feature_map_loss_is_the_l1_norm_over_channels():
    """Eq. (5) is ||a[k]||_1, i.e. a sum of absolute values over channels.

    The original driver script for the feature map experiments was lost; the
    channel reduction is confirmed to be the absolute sum. Pinning it here
    because the alternatives are easy to substitute by accident and only agree
    on non-negative activations.
    """
    from fame.losses import feature_map_loss

    class Fixed(nn.Module):
        """Returns a known activation, so the reduction can be read off."""

        def __init__(self, values):
            super().__init__()
            self.values = values
            self.scale = nn.Parameter(torch.ones(1))

        def forward(self, x):
            return self.values * self.scale

    # One location, four channels, deliberately mixing signs.
    activation = torch.tensor([[[[3.0]], [[-4.0]], [[0.0]], [[-1.0]]]])
    extractor = Fixed(activation)

    loss = feature_map_loss(extractor, [(0, 0)])(torch.rand(1, 3, 8, 8))

    assert loss.item() == pytest.approx(8.0), "|3| + |-4| + |0| + |-1|"
    # The alternatives would give different answers on the same input.
    assert loss.item() != pytest.approx(activation.sum().item())  # plain sum: -2
    assert loss.item() != pytest.approx(activation.norm().item())  # L2: ~5.1


def test_feature_map_loss_reductions_agree_after_a_relu():
    """With non-negative activations the absolute sum equals the plain sum."""
    from fame.losses import feature_map_loss

    class Fixed(nn.Module):
        def __init__(self, values):
            super().__init__()
            self.values = values
            self.scale = nn.Parameter(torch.ones(1))

        def forward(self, x):
            return torch.relu(self.values * self.scale)

    activation = torch.tensor([[[[3.0]], [[4.0]], [[0.0]], [[1.0]]]])

    loss = feature_map_loss(Fixed(activation), [(0, 0)])(torch.rand(1, 3, 8, 8))

    assert loss.item() == pytest.approx(8.0)
    assert loss.item() == pytest.approx(torch.relu(activation).sum().item())


def test_feature_map_loss_ignores_other_locations():
    """Only a[k] is penalized; the rest of the map is left free."""
    from fame.losses import feature_map_loss

    class Fixed(nn.Module):
        def __init__(self, values):
            super().__init__()
            self.values = values
            self.scale = nn.Parameter(torch.ones(1))

        def forward(self, x):
            return self.values * self.scale

    activation = torch.zeros(1, 2, 3, 3)
    activation[0, :, 1, 2] = torch.tensor([5.0, -2.0])
    activation[0, :, 0, 0] = torch.tensor([100.0, 100.0])  # must not contribute

    loss = feature_map_loss(Fixed(activation), [(1, 2)])(torch.rand(1, 3, 8, 8))

    assert loss.item() == pytest.approx(7.0)


def test_feature_map_over_a_7x7_map_yields_49_ordered_panels():
    """One image gives 49 maps, ordered row by row to tile a 7x7 sheet."""
    model = TinyNet().eval()
    image = torch.rand(1, 3, 28, 28)  # two stride-2 convs -> 7x7
    layer = model.features[-2]

    with FeatureMapExtractor(model, layer) as extractor, torch.no_grad():
        height, width = extractor(image).shape[2:]
    assert (height, width) == (7, 7)

    locations = [(row, col) for row in range(height) for col in range(width)]
    maps = explain_feature_map(model, layer, image, locations, FAST, batch_size=49)

    assert maps.shape == (49, 1, 28, 28)
    # Panel i belongs to location i, which is what the sheet layout relies on.
    single = explain_feature_map(model, layer, image, [(3, 4)], FAST)
    assert torch.allclose(maps[3 * width + 4], single[0], atol=1e-5)

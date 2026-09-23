"""Tests for the optimizer and the attribution post-processing.

These check properties that must hold for any input -- that the optimizer
actually descends its loss, that the step size is respected, that batch
elements stay independent -- rather than comparing against stored numbers.
They use tiny randomly initialized networks so the suite runs on CPU in
seconds and needs no downloaded weights.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from fame import FameConfig, fame, fame_attribution
from fame.attribution import normalize, to_grayscale
from fame.losses import classification_loss, cosine_similarity


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)
    np.random.seed(0)


class TinyClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def test_optimizer_descends_the_loss():
    model = TinyClassifier().eval()
    image = torch.rand(1, 3, 32, 32)
    loss_fn = classification_loss(model, torch.tensor([3]))

    before = loss_fn(image).item()
    after = loss_fn(fame(image, loss_fn, iterations=50)).item()

    assert after < before


def test_step_size_bounds_the_perturbation():
    """After n steps of size eta no pixel can have moved more than n * eta."""
    model = TinyClassifier().eval()
    image = torch.rand(1, 3, 32, 32)
    loss_fn = classification_loss(model, torch.tensor([0]))

    iterations, step_size = 20, 1.0 / 255.0
    perturbed = fame(image, loss_fn, step_size=step_size, iterations=iterations)

    assert (perturbed - image).abs().max().item() <= iterations * step_size + 1e-6


def test_largest_gradient_moves_by_exactly_the_step_size():
    """Gradient normalization makes the strongest pixel move by eta."""
    model = TinyClassifier().eval()
    image = torch.rand(1, 3, 32, 32)
    loss_fn = classification_loss(model, torch.tensor([1]))

    step_size = 0.01
    perturbed = fame(image, loss_fn, step_size=step_size, iterations=1)

    assert (perturbed - image).abs().max().item() == pytest.approx(step_size, rel=1e-5)


def test_batch_elements_are_independent():
    """A sample must get the same map alone as it does inside a batch.

    This is the regression test for the per-sample gradient normalization: with
    a single maximum taken over the whole batch, the loud sample would set the
    step size for the quiet one.
    """
    model = TinyClassifier().eval()
    images = torch.rand(2, 3, 32, 32)
    # Make one sample carry much larger gradients than the other.
    images[1] = images[1] * 0.01
    targets = torch.tensor([3, 7])

    batched = fame(images, classification_loss(model, targets), iterations=10)
    alone = torch.cat(
        [
            fame(images[i : i + 1], classification_loss(model, targets[i : i + 1]), iterations=10)
            for i in range(2)
        ]
    )

    assert torch.allclose(batched, alone, atol=1e-5)


def test_early_stopping_can_return_the_image_untouched():
    """A loss already below epsilon must stop before the first step.

    Worth pinning down because it is a real failure mode on impostor pairs,
    where the similarity starts below the default threshold and the resulting
    attribution map is empty.
    """
    model = TinyClassifier().eval()
    image = torch.rand(1, 3, 32, 32)
    loss_fn = classification_loss(model, torch.tensor([0]))

    perturbed = fame(image, loss_fn, iterations=100, epsilon=1e9)

    assert torch.equal(perturbed, image)


def test_clamp_keeps_the_image_in_range():
    model = TinyClassifier().eval()
    image = torch.rand(1, 3, 32, 32)
    loss_fn = classification_loss(model, torch.tensor([0]))

    perturbed = fame(image, loss_fn, step_size=0.5, iterations=20, clamp=(0.0, 1.0))

    assert perturbed.min() >= 0.0 and perturbed.max() <= 1.0


def test_attribution_is_single_channel_and_normalized():
    image = torch.rand(2, 3, 64, 64)
    perturbed = image + 0.05 * torch.randn_like(image)

    maps = fame_attribution(image, perturbed, blur_kernel=25, blur_sigma=5.0)

    assert maps.shape == (2, 1, 64, 64)
    assert maps.min() >= 0.0 and maps.max() <= 1.0
    for single in maps:
        assert single.max().item() == pytest.approx(1.0, abs=1e-5)


def test_attribution_ignores_the_sign_of_the_perturbation():
    """Eq. (4) takes the absolute difference, so direction must not matter."""
    image = torch.rand(1, 3, 32, 32)
    delta = 0.05 * torch.randn_like(image)

    positive = fame_attribution(image, image + delta, blur_kernel=11, blur_sigma=3.0)
    negative = fame_attribution(image, image - delta, blur_kernel=11, blur_sigma=3.0)

    assert torch.allclose(positive, negative, atol=1e-6)


def test_zero_perturbation_gives_an_empty_map_without_dividing_by_zero():
    image = torch.rand(1, 3, 32, 32)

    maps = fame_attribution(image, image.clone())

    assert torch.isfinite(maps).all()
    assert maps.abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_prescaling_does_not_change_the_attribution():
    """The original code divided by the max before blurring.

    Blurring is linear and normalization is scale invariant, so dropping that
    intermediate step leaves the result unchanged -- this pins that down.
    """
    image = torch.zeros(1, 3, 32, 32)
    difference = torch.rand(1, 3, 32, 32)

    direct = fame_attribution(image, difference, blur_kernel=11, blur_sigma=3.0)
    prescaled = fame_attribution(image, difference / difference.max(), blur_kernel=11, blur_sigma=3.0)

    assert torch.allclose(direct, prescaled, atol=1e-5)


def test_grayscale_matches_torchvision():
    import torchvision

    image = torch.rand(2, 3, 16, 16)
    expected = torchvision.transforms.Grayscale(num_output_channels=1)(image)

    assert torch.allclose(to_grayscale(image), expected, atol=1e-6)


@pytest.mark.parametrize("mode", ["max", "minmax"])
def test_normalization_modes_reach_one(mode):
    values = torch.rand(3, 1, 8, 8) + 0.5

    normalized = normalize(values, mode=mode)

    assert normalized.amax(dim=(-2, -1)).allclose(torch.ones(3, 1), atol=1e-5)
    if mode == "minmax":
        assert normalized.amin(dim=(-2, -1)).allclose(torch.zeros(3, 1), atol=1e-5)
    else:
        # Dividing by the max alone leaves the floor above zero.
        assert normalized.amin() > 0.0


def test_normalization_rejects_unknown_mode():
    with pytest.raises(ValueError):
        normalize(torch.rand(1, 1, 4, 4), mode="l2")


def test_cosine_similarity_detaches_only_the_denominator():
    """Footnote 4: gradients flow through the dot product, not the norms."""
    a = torch.rand(1, 8)
    b = torch.rand(1, 8, requires_grad=True)

    detached = cosine_similarity(a, b, detach_norms=True)
    full = cosine_similarity(a, b, detach_norms=False)

    # The value is the same either way; only the gradient differs.
    assert torch.allclose(detached, full, atol=1e-6)
    grad_detached = torch.autograd.grad(detached, b, retain_graph=True)[0]
    grad_full = torch.autograd.grad(full, b)[0]
    assert not torch.allclose(grad_detached, grad_full, atol=1e-4)


def test_config_defaults_match_the_paper():
    config = FameConfig()

    assert config.step_size == pytest.approx(1.0 / 255.0)
    assert config.iterations == 500
    assert config.blur_sigma == pytest.approx(7.7)
    # Unconstrained updates, as in the original implementation.
    assert config.clamp is None


def test_public_names_survive_importing_their_submodules():
    """Exported callables must not be shadowed by same-named submodules.

    A submodule overwrites the attribute of the same name on its package every
    time it is imported, so `from fame.baselines import corrise` used to hand
    back the module rather than the function depending on import order. The
    modules are named for their method families now; this pins that down.
    """
    import importlib

    for package_name, attribute in [
        ("fame", "fame"),
        ("fame.baselines", "corrise"),
        ("fame.baselines", "fggb"),
        ("fame.baselines", "classification_cam"),
        ("fame.metrics", "eer_threshold"),
        ("fame.data", "load_crop"),
    ]:
        package = importlib.import_module(package_name)
        assert callable(getattr(package, attribute)), f"{package_name}.{attribute} is not callable"

    # Importing every submodule explicitly must not change that.
    for module_name in [
        "fame.optimizer",
        "fame.attribution",
        "fame.explain",
        "fame.losses",
        "fame.baselines.cam",
        "fame.baselines.perturbation",
        "fame.baselines.gradient",
        "fame.metrics.classification",
        "fame.metrics.verification",
        "fame.data.imagenet",
        "fame.data.pairs",
    ]:
        importlib.import_module(module_name)

    import fame as package
    import fame.baselines as baselines

    assert callable(package.fame)
    assert callable(baselines.corrise)
    assert callable(baselines.fggb)


def test_every_exported_name_resolves():
    """Everything in __all__ must actually exist on the package."""
    import fame as package

    for name in package.__all__:
        assert hasattr(package, name), f"fame.__all__ lists {name} but it is missing"

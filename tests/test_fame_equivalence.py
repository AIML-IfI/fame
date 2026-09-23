"""Equivalence tests for FAME and the CAM baselines.

Same approach as ``test_baseline_equivalence.py``: the relevant part of each
original script is transcribed literally and the package version has to match
it numerically.  Sources:

    code_fame.py   lots, get_difference, lots_vis          (classification)
    utils.py       lots, Cosine_Loss, myCosine_Loss,       (face recognition)
                   get_difference, lots_vis
    code_cam.py    process_img                             (classification)
    cam.py         cam_dot_product, cam_cosine_similarity  (face recognition)
    targets.py     DotProductTarget, CosineDistanceTarget, compute_norm
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchvision

from fame import FameConfig, explain_classification, explain_verification, fame_attribution
from fame.losses import classification_loss, verification_loss
from fame.optimizer import fame as fame_optimize


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)
    np.random.seed(0)


class TinyClassifier(nn.Module):
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
    def __init__(self, dimensions: int = 32, size: int = 32):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(8 * (size // 2) ** 2, dimensions))

    def forward(self, x):
        return self.head(self.features(x))


# --------------------------------------------------------------------------
# Reference transcriptions


def reference_lots_classification(network, img, target, iterations, epsilon, stepwidth=1.0 / 255.0):
    """Transcription of lots() from code_fame.py, on CPU."""
    img = img.clone().detach().requires_grad_(True)
    loss_function = torch.nn.L1Loss(reduction="sum")

    for _ in range(iterations):
        network.zero_grad()
        prediction = network(img)
        loss = loss_function(prediction[0][target], torch.tensor(0.0))
        with torch.no_grad():
            if epsilon is not None and loss.mean() < epsilon:
                break
        loss.backward(retain_graph=True)
        gradient = img.grad.detach()
        with torch.no_grad():
            gradient_step = gradient * (stepwidth / torch.max(torch.abs(gradient)))
            img = (img - gradient_step).detach()
            img.requires_grad_(True)
    return img.detach()


class ReferenceCosineLoss(torch.nn.Module):
    """Transcription of Cosine_Loss and compute_norm from utils.py."""

    def __init__(self, invert=False):
        super().__init__()
        self.invert = invert

    def forward(self, x, y):
        epsilon = 1e-8
        dot_product = torch.sum(x * y, dim=1)
        norm_x1 = torch.maximum(torch.norm(x, p=2, dim=-1), torch.tensor(epsilon))
        norm_x2 = torch.maximum(torch.norm(y, p=2, dim=-1), torch.tensor(epsilon))
        similarity_score = dot_product / (norm_x1 * norm_x2)
        return 1 - similarity_score if self.invert else similarity_score


def reference_lots_verification(network, img, target_features, loss_func, iterations, epsilon=0.01):
    """Transcription of lots() from utils.py."""
    stepwidth = 1.0 / 255.0
    img = img.clone().detach().requires_grad_(True)

    for _ in range(iterations):
        network.zero_grad()
        features = network(img)
        loss = loss_func(features, target_features)
        with torch.no_grad():
            if loss.mean() < epsilon:
                break
        loss = loss.sum()
        loss.backward(retain_graph=True)
        gradient = img.grad.detach()
        with torch.no_grad():
            gradient_step = gradient * (stepwidth / torch.max(torch.abs(gradient)))
            img = (img - gradient_step).detach()
            img.requires_grad_(True)
    return img.detach()


def reference_get_difference(adversarial_batch, x_batch):
    """Transcription of get_difference, shared by code_fame.py and utils.py."""
    adversarial_batch = torchvision.transforms.Grayscale(num_output_channels=1)(adversarial_batch)
    x_batch = torchvision.transforms.Grayscale(num_output_channels=1)(x_batch)
    diff_batch = abs(adversarial_batch - x_batch)
    max_vals = diff_batch.amax(dim=(2, 3), keepdim=True)
    return diff_batch / max_vals


def reference_lots_vis(diff_batch):
    """Transcription of lots_vis, the FAME variant with kernel 49 / sigma 7.7."""
    blur_tf = torchvision.transforms.GaussianBlur(kernel_size=49, sigma=7.7)
    pert_img = blur_tf(diff_batch)
    max_vals = pert_img.amax(dim=(2, 3), keepdim=True)
    min_vals = pert_img.amin(dim=(2, 3), keepdim=True)
    return (pert_img - min_vals) / (max_vals - min_vals + 1e-8)


# --------------------------------------------------------------------------
# FAME post-processing


def test_attribution_matches_get_difference_and_lots_vis():
    """The whole post-processing chain, against the original two functions."""
    image = torch.rand(1, 3, 64, 64)
    perturbed = image + 0.03 * torch.randn_like(image)

    reference = reference_lots_vis(reference_get_difference(perturbed, image))
    ours = fame_attribution(image, perturbed, blur_kernel=49, blur_sigma=7.7)

    assert torch.allclose(ours, reference, atol=1e-6)


def test_greying_before_subtracting_is_not_the_same_as_after():
    """|gray(a) - gray(b)| differs from gray(|a - b|).

    The absolute value sits between the two linear steps, so the order matters
    whenever the colour channels move in opposite directions. The original
    greys first; getting this backwards changes every map.
    """
    from fame.attribution import smooth_and_normalize, to_grayscale

    image = torch.rand(1, 3, 32, 32)
    delta = torch.zeros_like(image)
    delta[0, 0] = 0.1  # Red moves up,
    delta[0, 2] = -0.1  # blue moves down.
    perturbed = image + delta

    grey_first = fame_attribution(image, perturbed, blur_kernel=None)
    subtract_first = smooth_and_normalize(
        to_grayscale((perturbed - image).abs()), blur_kernel=None
    )

    assert not torch.allclose(grey_first, subtract_first, atol=1e-3)


def test_prescaling_inside_get_difference_is_redundant():
    """get_difference divides by the max before lots_vis normalizes again.

    Blur is linear and normalization is scale invariant, so dropping the first
    division leaves the result unchanged.
    """
    image = torch.rand(1, 3, 32, 32)
    perturbed = image + 0.05 * torch.randn_like(image)

    with_prescale = reference_lots_vis(reference_get_difference(perturbed, image))
    without = fame_attribution(image, perturbed, blur_kernel=49, blur_sigma=7.7)

    assert torch.allclose(with_prescale, without, atol=1e-6)


# --------------------------------------------------------------------------
# FAME for image classification


def test_fame_classification_matches_the_original_optimizer():
    model = TinyClassifier().eval()
    image = torch.rand(1, 3, 64, 64)
    target = 3
    iterations = 30

    reference_adv = reference_lots_classification(model, image, target, iterations, epsilon=1e-3)
    reference_map = reference_lots_vis(reference_get_difference(reference_adv, image))

    config = FameConfig(iterations=iterations, epsilon=1e-3, blur_kernel=49, blur_sigma=7.7)
    ours = explain_classification(model, image, torch.tensor([target]), config)

    assert torch.allclose(ours, reference_map, atol=1e-5)


def test_absolute_logit_matches_the_raw_logit_while_it_stays_positive():
    """code_fame.py minimizes |z_o|; Sec. 3.4 minimizes z_o.

    Identical as long as the logit is positive, which holds for the ground
    truth class of a correctly classified image.
    """
    model = TinyClassifier().eval()
    image = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        target = int(model(image).argmax())
        assert model(image)[0, target] > 0

    raw = fame_optimize(image, classification_loss(model, torch.tensor([target])), iterations=5)
    absolute = fame_optimize(
        image, classification_loss(model, torch.tensor([target]), absolute=True), iterations=5
    )

    assert torch.allclose(raw, absolute, atol=1e-6)


def test_absolute_logit_reverses_once_the_logit_is_negative():
    """For a class the network rejects, the two losses pull opposite ways."""
    model = TinyClassifier().eval()
    image = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        logits = model(image)[0]
        target = int(logits.argmin())
        if logits[target] >= 0:
            pytest.skip("no negative logit in this random model")

    def logit_of(x):
        with torch.no_grad():
            return model(x)[0, target].item()

    start = logit_of(image)
    raw = logit_of(fame_optimize(image, classification_loss(model, torch.tensor([target])), iterations=20))
    absolute = logit_of(
        fame_optimize(
            image, classification_loss(model, torch.tensor([target]), absolute=True), iterations=20
        )
    )

    assert raw < start, "minimizing z_o pushes the logit down"
    assert absolute > start, "minimizing |z_o| pulls a negative logit up towards zero"


# --------------------------------------------------------------------------
# FAME for face recognition


@pytest.mark.parametrize("mode,invert", [("similar", False), ("dissimilar", True)])
def test_fame_verification_matches_the_original_optimizer(mode, invert):
    model = TinyEmbedding(size=64).eval()
    probe, gallery = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
    iterations = 25

    with torch.no_grad():
        target_features = model(gallery).detach()

    reference_adv = reference_lots_verification(
        model, probe, target_features, ReferenceCosineLoss(invert=invert), iterations, epsilon=0.01
    )
    reference_map = reference_lots_vis(reference_get_difference(reference_adv, probe))

    config = FameConfig(iterations=iterations, epsilon=0.01, blur_kernel=49, blur_sigma=7.7)
    ours = explain_verification(model, probe, gallery, config, modes=(mode,))[mode]

    assert torch.allclose(ours, reference_map, atol=1e-5)


def test_verification_loss_keeps_the_norms_differentiable():
    """Cosine_Loss uses compute_norm, which is an ordinary differentiable op.

    Detaching the denominators changes the gradient, so it must not be the
    default.
    """
    model = TinyEmbedding().eval()
    probe, gallery = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        reference = model(gallery)

    attached = fame_optimize(probe, verification_loss(model, reference, detach_norms=False), iterations=10)
    detached = fame_optimize(probe, verification_loss(model, reference, detach_norms=True), iterations=10)

    assert not torch.allclose(attached, detached, atol=1e-6)


def test_dissimilar_loss_is_one_minus_the_similar_loss():
    """myCosine_Loss is exactly 1 - Cosine_Loss."""
    model = TinyEmbedding().eval()
    probe, gallery = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        reference = model(gallery)

    similar = verification_loss(model, reference, mode="similar")(probe)
    dissimilar = verification_loss(model, reference, mode="dissimilar")(probe)

    assert torch.allclose(similar + dissimilar, torch.ones_like(similar), atol=1e-6)


# --------------------------------------------------------------------------
# CAM baselines


@pytest.mark.parametrize("method", ["GradCAM", "GradCAMElementWise", "HiResCAM"])
def test_classification_cam_matches_process_img(method):
    """Transcription of process_img from code_cam.py."""
    pytest.importorskip("pytorch_grad_cam")
    import pytorch_grad_cam
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    from fame.baselines import classification_cam

    model = TinyClassifier().eval()
    image = torch.rand(1, 3, 64, 64)
    target = 5
    layers = [model.features[-2]]

    cam_class = getattr(pytorch_grad_cam, method)
    with cam_class(model=model, target_layers=layers) as cam:
        reference = cam(image, [ClassifierOutputTarget(target)])

    ours = classification_cam(model, layers, image, [target], method=method)

    assert np.allclose(ours, reference, atol=1e-5)


@pytest.mark.parametrize("method", ["GradCAM", "GradCAMElementWise"])
def test_verification_cam_matches_cam_cosine_similarity(method):
    """Transcription of cam_cosine_similarity from cam.py.

    CosineDistanceTarget divides the dot product by a norm product computed
    once from the clean images, so the denominator is a constant. Our
    CosineSimilarityTarget with detached norms is the same function of the
    output, since CAM evaluates the model on the same image.
    """
    pytest.importorskip("pytorch_grad_cam")
    import pytorch_grad_cam

    from fame.baselines import verification_cam

    class CosineDistanceTarget:
        def __init__(self, features, norm_term):
            self.features = features
            self.norm_term = norm_term

        def __call__(self, model_output):
            return torch.dot(self.features, model_output) / self.norm_term

    model = TinyEmbedding(size=64).eval()
    gallery, probe = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
    layers = [model.features[-2]]

    with torch.no_grad():
        features_g = model(gallery).detach()[0]
        features_p = model(probe).detach()[0]
        epsilon = 1e-8
        norm_term = torch.maximum(
            features_g.norm(p=2, dim=-1), torch.tensor(epsilon)
        ) * torch.maximum(features_p.norm(p=2, dim=-1), torch.tensor(epsilon))

    cam_class = getattr(pytorch_grad_cam, method)
    with cam_class(model=model, target_layers=layers) as cam:
        reference = cam(probe, [CosineDistanceTarget(features_g, norm_term)])

    ours = verification_cam(model, layers, probe, gallery, method=method, target="cosine")

    assert np.allclose(ours, reference, atol=1e-5)


def test_verification_cam_dot_target_matches_dot_product_target():
    """Transcription of cam_dot_product from cam.py."""
    pytest.importorskip("pytorch_grad_cam")
    import pytorch_grad_cam

    from fame.baselines import verification_cam

    class DotProductTarget:
        def __init__(self, features):
            self.features = features

        def __call__(self, model_output):
            return torch.dot(self.features, model_output)

    model = TinyEmbedding(size=64).eval()
    gallery, probe = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
    layers = [model.features[-2]]

    with torch.no_grad():
        features_g = model(gallery).detach()[0]

    with pytorch_grad_cam.GradCAM(model=model, target_layers=layers) as cam:
        reference = cam(probe, [DotProductTarget(features_g)])

    ours = verification_cam(model, layers, probe, gallery, method="GradCAM", target="dot")

    assert np.allclose(ours, reference, atol=1e-5)


def test_cosine_and_dot_cam_agree_because_the_denominator_is_constant():
    """CosineDistanceTarget only rescales DotProductTarget.

    CAM normalizes its output map, so a constant factor drops out and the two
    targets give the same attribution.
    """
    pytest.importorskip("pytorch_grad_cam")

    from fame.baselines import verification_cam

    model = TinyEmbedding(size=64).eval()
    gallery, probe = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
    layers = [model.features[-2]]

    cosine = verification_cam(model, layers, probe, gallery, method="GradCAM", target="cosine")
    dot = verification_cam(model, layers, probe, gallery, method="GradCAM", target="dot")

    assert np.allclose(cosine, dot, atol=1e-5)


def test_cam_upsampling_is_backend_independent():
    """The original vendored a tensor-computation rewrite of grad-cam.

    The one place such a rewrite could change results is the upsampling from
    the feature map to input resolution: pytorch_grad_cam does it with
    cv2.resize, while a tensor version would use F.interpolate. They agree to
    float32 epsilon, and the percentile masks that deletion and insertion
    actually consume are identical, so the CAM tests above transfer.

    Note that argsort order does differ: bilinear upsampling produces many
    exactly equal values and float noise reorders the ties. Ranking by
    percentile threshold, which is what the metrics do, is unaffected.
    """
    cv2 = pytest.importorskip("cv2")
    import torch.nn.functional as F

    from fame.metrics import top_percent_mask

    rng = np.random.default_rng(0)
    for shape in [(7, 7), (14, 14)]:
        cam = rng.random(shape).astype(np.float32)

        with_cv2 = cv2.resize(cam, (224, 224))
        with_torch = (
            F.interpolate(
                torch.from_numpy(cam)[None, None], size=(224, 224), mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()
        )

        assert np.abs(with_cv2 - with_torch).max() < 1e-6
        for percentage in (10.0, 30.0, 50.0):
            assert np.array_equal(
                top_percent_mask(with_cv2, percentage), top_percent_mask(with_torch, percentage)
            )

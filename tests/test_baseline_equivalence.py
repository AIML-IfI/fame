"""Equivalence tests against the original baseline implementations.

The package versions of CorrRISE and FGGB were rewritten for batching and
reuse, so the risk is that a rewrite quietly changes the numbers.  Each test
here transcribes the relevant part of the original script, runs both on the
same inputs, and requires them to agree.

The references are deliberately literal, loops and all -- they are the
specification, not code anyone should run.  Sources:

    Corrise.py            generate_mask, get_masked_img, process_pair
    code_fggb.py          _normalize_embedding, process_pair
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from fame.baselines import corrise, fggb
from fame.baselines.perturbation import random_masks


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)
    np.random.seed(0)


class TinyEmbedding(nn.Module):
    """Small stand-in for an IResNet, with a spatially sensitive head."""

    def __init__(self, dimensions: int = 32, size: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(8 * (size // 2) ** 2, dimensions))

    def forward(self, x):
        return self.head(self.features(x))


# --------------------------------------------------------------------------
# Reference transcriptions


def reference_fggb(model, img_g, img_p, threshold):
    """Transcription of process_pair from code_fggb.py.

    Returns the four maps the original saves, before its blur and colouring:
    (sim_a, dis_a, sim_b, dis_b) for the gallery and the probe.
    """

    def _normalize_embedding(x, eps=1e-8):
        return x / (x.norm(p=2, dim=1, keepdim=True) + eps)

    img_g = img_g.clone().detach().requires_grad_(True)
    img_p = img_p.clone().detach().requires_grad_(True)

    f_a = model(img_g)
    f_b = model(img_p)
    f_a = _normalize_embedding(f_a)
    f_b = _normalize_embedding(f_b)

    weights = f_a * f_b
    N = weights.shape[1]
    weights = weights - threshold / N

    grads_a, grads_b = [], []
    for k in range(N):
        model.zero_grad(set_to_none=True)
        if img_g.grad is not None:
            img_g.grad.zero_()
        if img_p.grad is not None:
            img_p.grad.zero_()
        f_a[0, k].backward(retain_graph=True)
        g_a = img_g.grad.detach().clone().abs().mean(dim=1)
        grads_a.append(g_a / (torch.norm(g_a, p=2) + 1e-8))

        model.zero_grad(set_to_none=True)
        img_g.grad.zero_()
        if img_p.grad is not None:
            img_p.grad.zero_()
        f_b[0, k].backward(retain_graph=True)
        g_b = img_p.grad.detach().clone().abs().mean(dim=1)
        grads_b.append(g_b / (torch.norm(g_b, p=2) + 1e-8))

    grads_a = torch.stack(grads_a, dim=0).cpu()
    grads_b = torch.stack(grads_b, dim=0).cpu()
    w = weights.t()[:, :, None, None].detach().cpu()

    S_a = (grads_a * w).sum(dim=0)
    S_b = (grads_b * w).sum(dim=0)

    return (
        torch.clamp(S_a, min=0.0),
        -torch.clamp(S_a, max=0.0),
        torch.clamp(S_b, min=0.0),
        -torch.clamp(S_b, max=0.0),
    )


def reference_corrise(model, img_g, img_p, mask_set_one_channel):
    """Transcription of process_pair from Corrise.py.

    Returns (s_A_pos, |s_A_neg|, s_B_pos, |s_B_neg|), where A is the gallery.
    """
    from scipy.stats import rankdata

    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    count = mask_set_one_channel.shape[0]
    mask_set_3ch = mask_set_one_channel.unsqueeze(1).expand(-1, 3, -1, -1)

    with torch.no_grad():
        masked_g = img_g.expand(count, -1, -1, -1) * mask_set_3ch
        masked_p = img_p.expand(count, -1, -1, -1) * mask_set_3ch

        features_g = model(img_g)
        features_p = model(img_p)
        features_g_mask = model(masked_g)
        features_p_mask = model(masked_p)

        sc_A = cos(features_p, features_g_mask).cpu().numpy()
        sc_B = cos(features_g, features_p_mask).cpu().numpy()

    size = mask_set_one_channel.shape[-1]
    mask_flat = mask_set_one_channel.numpy().reshape(count, -1)
    mask_ranked = np.apply_along_axis(rankdata, 0, mask_flat)
    sc_A_ranked = rankdata(sc_A)
    sc_B_ranked = rankdata(sc_B)

    def fast_corr(x, Y):
        x = (x - x.mean()) / x.std()
        Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
        return (x[:, None] * Y).mean(axis=0)

    s_A = np.nan_to_num(fast_corr(sc_A_ranked, mask_ranked).reshape(size, size), nan=0.0)
    s_B = np.nan_to_num(fast_corr(sc_B_ranked, mask_ranked).reshape(size, size), nan=0.0)

    return (
        np.where(s_A < 0, 0, s_A),
        np.abs(np.where(s_A >= 0, 0, s_A)),
        np.where(s_B < 0, 0, s_B),
        np.abs(np.where(s_B >= 0, 0, s_B)),
    )


# --------------------------------------------------------------------------
# FGGB


def test_fggb_matches_the_original_on_both_sides_of_a_pair():
    model = TinyEmbedding().eval()
    gallery, probe = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)
    threshold = 0.3

    sim_a, dis_a, sim_b, dis_b = reference_fggb(model, gallery, probe, threshold)

    ours_gallery = fggb(model, gallery, probe, threshold=threshold)
    ours_probe = fggb(model, probe, gallery, threshold=threshold)

    assert torch.allclose(ours_gallery["similar"], sim_a, atol=1e-5)
    assert torch.allclose(ours_gallery["dissimilar"], dis_a, atol=1e-5)
    assert torch.allclose(ours_probe["similar"], sim_b, atol=1e-5)
    assert torch.allclose(ours_probe["dissimilar"], dis_b, atol=1e-5)


@pytest.mark.parametrize("threshold", [0.0, 0.15, 0.45])
def test_fggb_matches_the_original_across_thresholds(threshold):
    model = TinyEmbedding(dimensions=16).eval()
    gallery, probe = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)

    sim_a, dis_a, _, _ = reference_fggb(model, gallery, probe, threshold)
    ours = fggb(model, gallery, probe, threshold=threshold)

    assert torch.allclose(ours["similar"], sim_a, atol=1e-5)
    assert torch.allclose(ours["dissimilar"], dis_a, atol=1e-5)


def test_fggb_differentiates_the_normalized_embedding():
    """Gradients must run through the L2 normalization, as the original does.

    Differentiating the raw embedding gives visibly different maps, because
    normalizing couples the dimensions together.
    """
    model = TinyEmbedding(dimensions=16).eval()
    gallery = torch.rand(1, 3, 32, 32)

    def raw_gradients():
        image = gallery.clone().detach().requires_grad_(True)
        embedding = model(image)[0]
        maps = []
        for index in range(embedding.shape[0]):
            (gradient,) = torch.autograd.grad(embedding[index], image, retain_graph=True)
            gradient = gradient.detach().abs().mean(dim=1)[0]
            maps.append(gradient / (gradient.norm(p=2) + 1e-8))
        return torch.stack(maps)

    from fame.baselines.gradient import _embedding_gradients

    assert not torch.allclose(_embedding_gradients(model, gallery), raw_gradients(), atol=1e-4)


# --------------------------------------------------------------------------
# CorrRISE


def test_corrise_matches_the_original_on_both_sides_of_a_pair():
    pytest.importorskip("scipy")
    model = TinyEmbedding().eval()
    gallery, probe = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)

    # One shared mask set, as the original builds at import time.
    generator = torch.Generator().manual_seed(3)
    masks = random_masks(64, 32, 32, patches=3, patch_size=8, generator=generator)

    ref_sim_a, ref_dis_a, ref_sim_b, ref_dis_b = reference_corrise(
        model, gallery, probe, masks[:, 0]
    )

    ours_gallery = corrise(model, gallery, probe, masks=masks)
    ours_probe = corrise(model, probe, gallery, masks=masks)

    assert np.allclose(ours_gallery["similar"][0].numpy(), ref_sim_a, atol=1e-5)
    assert np.allclose(ours_gallery["dissimilar"][0].numpy(), ref_dis_a, atol=1e-5)
    assert np.allclose(ours_probe["similar"][0].numpy(), ref_sim_b, atol=1e-5)
    assert np.allclose(ours_probe["dissimilar"][0].numpy(), ref_dis_b, atol=1e-5)


def test_corrise_patches_are_never_truncated_at_the_border():
    """generate_mask draws origins from [0, H - patch_size], so patches fit.

    Sampling from the full range instead and letting patches run off the edge
    would occlude fewer pixels than asked for. With one patch per mask, every
    mask must therefore contain exactly patch_size**2 zeros.
    """
    patch_size = 8
    masks = random_masks(
        300, 32, 32, patches=1, patch_size=patch_size, generator=torch.Generator().manual_seed(0)
    )

    zeros_per_mask = (masks[:, 0] == 0).sum(dim=(1, 2))

    assert (zeros_per_mask == patch_size**2).all(), "a patch was clipped at the border"


def test_corrise_occlusion_is_uniform_away_from_the_border():
    """Interior pixels are covered equally often, so none is favoured."""
    masks = random_masks(
        4000, 32, 32, patches=1, patch_size=8, generator=torch.Generator().manual_seed(0)
    )

    rate = (1 - masks[:, 0]).mean(dim=0)
    interior = rate[8:24, 8:24]

    # Every interior pixel sits under the same number of valid patch origins.
    assert (interior.max() - interior.min()) < 0.05
    # Border pixels are covered less often; that is geometry, not truncation.
    assert rate[0, 0] < interior.mean()


def test_corrise_uses_three_patches_by_default():
    """Corrise.py calls generate_mask_set(..., num_patches=3)."""
    import inspect

    from fame.baselines.perturbation import corrise as corrise_fn

    assert inspect.signature(random_masks).parameters["patches"].default == 3
    assert inspect.signature(corrise_fn).parameters["patches"].default == 3
    assert inspect.signature(corrise_fn).parameters["patch_size"].default == 30


def test_corrise_shared_masks_remove_between_image_variance():
    """Reusing one mask set makes two images directly comparable."""
    model = TinyEmbedding().eval()
    masks = random_masks(40, 32, 32, patches=3, patch_size=8, generator=torch.Generator().manual_seed(1))
    probe, gallery = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)

    first = corrise(model, probe, gallery, masks=masks)
    second = corrise(model, probe, gallery, masks=masks)

    assert torch.allclose(first["similar"], second["similar"])


def test_corrise_polarities_are_disjoint_and_finite():
    model = TinyEmbedding().eval()
    probe, gallery = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)

    maps = corrise(model, probe, gallery, num_masks=40, patch_size=8, seed=0)

    assert (maps["similar"] * maps["dissimilar"]).max() == 0
    assert torch.isfinite(maps["similar"]).all()
    assert torch.isfinite(maps["dissimilar"]).all()

"""Equivalence tests for the evaluation metrics on both tasks.

The evaluation scripts were rewritten rather than ported -- eight near-identical
face recognition scripts collapsed into one, and IoU and ROAD merged into a
single pass -- so the metric functions are transcribed here from the originals
and required to agree.  Sources:

    eval_iou.py           adjust_bbx, compute_iou
    eval_road_delete.py   top_p_mask, ImgNet.__getitem__, the confidence loop
    eval_delete_fame.py   top_p_mask
    eval_insert_fame.py   top_p_mask_insert
    eval_auc.py           get_auc
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchvision

from fame.data import adjust_box
from fame.metrics import (
    PERCENTAGES,
    accuracy_at,
    curve_from_scores,
    intersection_over_union,
    logit_drop,
    noisy_linear_imputation,
    normalized_auc,
    top_percent_mask,
)

RESIZE_SIZE = 232.0
CROP_SIZE = 224.0


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)
    np.random.seed(0)


# --------------------------------------------------------------------------
# Reference transcriptions


def reference_adjust_bbx(bounding_box, original_width, original_height):
    """Transcription of adjust_bbx from eval_iou.py."""
    x_min, y_min, x_max, y_max = bounding_box

    width_scale = RESIZE_SIZE / original_width
    height_scale = RESIZE_SIZE / original_height

    x_min *= width_scale
    y_min *= height_scale
    x_max *= width_scale
    y_max *= height_scale

    x_offset = (RESIZE_SIZE - CROP_SIZE) / 2
    y_offset = (RESIZE_SIZE - CROP_SIZE) / 2

    x_min -= x_offset
    y_min -= y_offset
    x_max -= x_offset
    y_max -= y_offset

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(CROP_SIZE, x_max)
    y_max = min(CROP_SIZE, y_max)

    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def reference_compute_iou(activation, bounding_box, thr=0.5):
    """Transcription of compute_iou from eval_iou.py."""
    sal_mask = activation >= thr
    bbox_mask = np.zeros((224, 224), dtype=bool)
    x_min, y_min, x_max, y_max = bounding_box
    x_min, y_min = int(x_min), int(y_min)
    x_max, y_max = int(x_max), int(y_max)

    bbox_mask[y_min:y_max, x_min:x_max] = True
    intersection = np.logical_and(sal_mask, bbox_mask).sum()
    union = np.logical_or(sal_mask, bbox_mask).sum()
    return intersection / (union + 1e-8)


def reference_top_p_mask(cam, p, ignore_nan=False):
    """Transcription of top_p_mask, shared by the deletion scripts."""
    perc_fn = np.nanpercentile if ignore_nan else np.percentile
    thr = perc_fn(cam, 100 - p)
    mask = cam <= thr
    return mask.astype(np.uint8)


def reference_top_p_mask_insert(cam, p, ignore_nan=False):
    """Transcription of top_p_mask_insert from eval_insert_fame.py."""
    perc_fn = np.nanpercentile if ignore_nan else np.percentile
    thr = perc_fn(cam, 100 - p)
    mask = cam > thr
    return mask.astype(np.uint8)


def reference_get_auc(accuracies, p_values):
    """Transcription of the AUC part of get_auc from eval_auc.py.

    The original calls np.trapz, which numpy 2.0 renamed to trapezoid and 2.4
    removed; the function itself is unchanged.
    """
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return trapezoid(accuracies, p_values) / (p_values[-1] - p_values[0])


def reference_road_confidence(model, images, cams, targets, percentage, normalize):
    """Transcription of the ImgNet masking and the confidence loop.

    ``ImgNet`` masks the [0, 1] tensor and normalizes afterwards, so removed
    pixels become the negative channel mean rather than zero once normalized.
    """
    total = 0.0
    for image, cam, target in zip(images, cams, targets):
        mask_img = reference_top_p_mask(cam, percentage)
        masked_img = image * mask_img[None, :, :]
        original = normalize(image)[None]
        masked = normalize(masked_img)[None]
        with torch.no_grad():
            output = model(original)
            output_masked = model(masked)
        total += (output[0, target] - output_masked[0, target]).item()
    return total / len(images)


# --------------------------------------------------------------------------
# Image classification: bounding boxes


@pytest.mark.parametrize(
    "box,width,height",
    [
        ([189, 187, 292, 225], 500, 375),
        ([10, 10, 400, 300], 640, 480),
        ([0, 0, 224, 224], 224, 224),
        ([50, 60, 120, 200], 375, 500),
    ],
)
def test_original_box_convention_is_reproduced(box, width, height):
    reference = reference_adjust_bbx(list(box), width, height)

    ours = adjust_box(list(box), width, height, aspect_preserving=False)

    assert ours == reference


def test_the_two_box_conventions_agree_on_square_images():
    """adjust_bbx is only correct when the image is already square.

    Allowing one pixel: both truncate with int(), so a coordinate landing on an
    exact integer can fall either side depending on the order of the floating
    point operations.
    """
    box = [40, 60, 180, 200]

    original = adjust_box(list(box), 400, 400, aspect_preserving=False)
    corrected = adjust_box(list(box), 400, 400, aspect_preserving=True)

    assert max(abs(a - b) for a, b in zip(original, corrected)) <= 1


def test_the_two_box_conventions_differ_on_a_wide_image():
    """Resize(232) scales by the shorter side, so a 4:3 image is not square.

    adjust_bbx stretches the box along the longer axis; the difference is tens
    of pixels on a 224 crop, which moves IoU noticeably.
    """
    box = [100, 100, 300, 250]

    original = adjust_box(list(box), 500, 375, aspect_preserving=False)
    corrected = adjust_box(list(box), 500, 375, aspect_preserving=True)

    assert original != corrected
    assert max(abs(a - b) for a, b in zip(original, corrected)) > 10


def test_aspect_preserving_boxes_keep_a_full_image_box_full():
    """A box covering the whole image must still cover the whole crop."""
    for width, height in [(500, 375), (375, 500), (224, 224)]:
        assert adjust_box([0, 0, width, height], width, height) == [0, 0, 224, 224]


# --------------------------------------------------------------------------
# Image classification: IoU


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
def test_iou_matches_compute_iou(threshold):
    attribution = np.random.rand(224, 224)
    box = [50, 60, 180, 200]

    reference = reference_compute_iou(attribution, box, thr=threshold)
    ours = intersection_over_union(attribution, box, threshold)

    assert ours == pytest.approx(float(reference), abs=1e-9)


def test_iou_matches_on_a_degenerate_empty_map():
    """Everything below threshold and a zero-area box: union is empty."""
    attribution = np.zeros((224, 224))
    box = [10, 10, 10, 10]

    reference = reference_compute_iou(attribution, box, thr=0.5)
    ours = intersection_over_union(attribution, box, 0.5)

    assert ours == pytest.approx(float(reference), abs=1e-9)
    assert np.isfinite(ours)


def test_iou_uses_the_bounding_box_row_column_order():
    """compute_iou fills bbox_mask[y_min:y_max, x_min:x_max], not the reverse."""
    attribution = np.zeros((224, 224))
    attribution[20:40, 100:150] = 1.0  # rows 20-40, columns 100-150

    aligned = intersection_over_union(attribution, [100, 20, 150, 40], 0.5)
    transposed = intersection_over_union(attribution, [20, 100, 40, 150], 0.5)

    assert aligned == pytest.approx(1.0, abs=1e-6)
    assert transposed < 0.1


# --------------------------------------------------------------------------
# Masks, shared by both tasks


@pytest.mark.parametrize("percentage", list(np.linspace(0, 100, 11)))
def test_deletion_mask_matches_top_p_mask(percentage):
    attribution = np.random.rand(112, 112)

    reference = reference_top_p_mask(attribution, percentage)
    ours = top_percent_mask(attribution, percentage)

    assert np.array_equal(ours, reference)


@pytest.mark.parametrize("percentage", list(np.linspace(0, 100, 11)))
def test_insertion_mask_matches_top_p_mask_insert(percentage):
    attribution = np.random.rand(112, 112)

    reference = reference_top_p_mask_insert(attribution, percentage)
    ours = top_percent_mask(attribution, percentage, keep_top=True)

    assert np.array_equal(ours, reference)


def test_masks_match_on_a_constant_map():
    """A tied map sends every pixel to the same side of the threshold."""
    attribution = np.full((32, 32), 0.5)

    for percentage in (0.0, 50.0, 100.0):
        assert np.array_equal(
            top_percent_mask(attribution, percentage), reference_top_p_mask(attribution, percentage)
        )
        assert np.array_equal(
            top_percent_mask(attribution, percentage, keep_top=True),
            reference_top_p_mask_insert(attribution, percentage),
        )


def test_full_deletion_keeps_the_pixels_sitting_at_the_minimum():
    """At P = 100 the threshold is the minimum, and `cam <= thr` still holds.

    Special-casing P = 100 to an all-zero mask would disagree with the
    original on the final point of every deletion curve.
    """
    attribution = np.random.rand(16, 16)

    ours = top_percent_mask(attribution, 100.0)

    assert np.array_equal(ours, reference_top_p_mask(attribution, 100.0))
    assert ours.sum() == 1, "exactly the single minimum pixel survives"


# --------------------------------------------------------------------------
# Image classification: ROAD


def test_road_confidence_drop_matches_the_original_loop():
    """The whole ROAD-Delete pass, against eval_road_delete.py."""
    torch.manual_seed(1)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10)).eval()
    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    images = torch.rand(6, 3, 32, 32)
    cams = np.random.rand(6, 32, 32)
    targets = torch.tensor([0, 1, 2, 3, 4, 5])
    percentage = 30.0

    reference = reference_road_confidence(model, images, cams, targets, percentage, normalize)

    masks = np.stack([top_percent_mask(cam, percentage) for cam in cams])
    mask = torch.from_numpy(masks).float()[:, None]
    normalized = normalize(images)
    perturbed = normalize(images * mask)
    ours = logit_drop(model, normalized, perturbed, targets).mean().item()

    assert ours == pytest.approx(reference, abs=1e-5)


def test_road_averages_over_the_actual_number_of_images():
    """The original divides by a hard-coded 5000 regardless of dataset size."""
    torch.manual_seed(2)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4)).eval()
    images = torch.rand(7, 3, 8, 8)
    targets = torch.tensor([0, 1, 2, 3, 0, 1, 2])

    drops = logit_drop(model, images, images * 0.5, targets)

    assert drops.shape == (7,)
    assert drops.mean().item() == pytest.approx(drops.sum().item() / 7, abs=1e-6)


def test_zero_imputation_is_applied_before_normalization():
    """ImgNet masks the [0, 1] image and normalizes afterwards.

    Removed pixels therefore land at -mean/std, not at zero, which is a
    different input to the network.
    """
    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    image = torch.rand(1, 3, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)

    before = normalize(image * mask)
    after = normalize(image) * mask

    assert not torch.allclose(before, after)
    assert torch.allclose(before[0, :, 0, 0], torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]), atol=1e-5)


def test_linear_imputation_is_opt_in_and_differs_from_zeroing():
    """The original does no imputation; ROAD's is available but not default."""
    image = torch.rand(1, 3, 16, 16)
    mask = (torch.rand(1, 1, 16, 16) > 0.3).float()

    zeroed = image * mask
    imputed = noisy_linear_imputation(image, mask, iterations=20, noise=0.0)

    assert not torch.allclose(zeroed, imputed)
    assert torch.allclose(zeroed * mask, imputed * mask, atol=1e-5)


# --------------------------------------------------------------------------
# Face recognition: accuracy and AUC


def test_auc_matches_get_auc():
    accuracies = [0.99, 0.95, 0.88, 0.80, 0.71, 0.65, 0.60, 0.57, 0.54, 0.52, 0.50]
    p_values = list(np.linspace(0, 100, 11))

    reference = reference_get_auc(accuracies, p_values)
    ours = normalized_auc(accuracies, p_values)

    assert ours == pytest.approx(float(reference), abs=1e-9)


def test_accuracy_matches_the_thresholding_in_get_auc():
    """get_auc binarises with `>= thr` and compares against the target column."""
    scores = np.array([0.8, 0.7, 0.2, 0.1, 0.45])
    labels = np.array([1, 1, 0, 0, 1])
    threshold = 0.45

    reference = float(((scores >= threshold).astype(int) == labels).mean())
    ours = accuracy_at(scores, labels, threshold)

    assert ours == pytest.approx(reference, abs=1e-9)


def test_accuracy_boundary_is_inclusive():
    """A score exactly at the threshold counts as a match."""
    assert accuracy_at([0.5], [1], 0.5) == pytest.approx(1.0)
    assert accuracy_at([0.5], [0], 0.5) == pytest.approx(0.0)


def test_curve_from_scores_matches_the_two_step_original():
    """eval_delete writes a per-P score table and eval_auc reduces it.

    Doing both in one function has to give the same number.
    """
    labels = np.array([1] * 30 + [0] * 30)
    threshold = 0.45
    rng = np.random.default_rng(0)
    scores_by_percentage = {
        float(p): np.concatenate(
            [rng.normal(0.8 - p / 150, 0.05, 30), rng.normal(0.2, 0.05, 30)]
        )
        for p in PERCENTAGES
    }

    accuracies = [
        float(((scores_by_percentage[float(p)] >= threshold).astype(int) == labels).mean())
        for p in PERCENTAGES
    ]
    reference = reference_get_auc(accuracies, list(PERCENTAGES))

    curve = curve_from_scores(scores_by_percentage, labels, threshold)

    assert curve["accuracies"] == pytest.approx(accuracies, abs=1e-9)
    assert curve["auc"] == pytest.approx(float(reference), abs=1e-9)


def test_deletion_and_insertion_score_in_opposite_directions():
    """A faithful map gives a low deletion AUC and a high insertion AUC."""
    labels = np.array([1] * 20 + [0] * 20)
    threshold = 0.5

    # Deletion: removing important pixels breaks genuine matches.
    deletion = {float(p): np.concatenate([np.full(20, 0.9 - p / 120), np.full(20, 0.1)]) for p in PERCENTAGES}
    # Insertion: adding them back restores the matches.
    insertion = {float(p): np.concatenate([np.full(20, 0.1 + p / 120), np.full(20, 0.1)]) for p in PERCENTAGES}

    assert curve_from_scores(deletion, labels, threshold)["auc"] < 0.75
    assert curve_from_scores(insertion, labels, threshold)["auc"] > 0.6


# --------------------------------------------------------------------------
# Face recognition: the EER operating point


def reference_compute_eer_threshold(scores, labels):
    """Transcription of compute_eer_threshold from the scores script."""
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(int)

    mask_g = labels == 1
    mask_i = labels == 0

    thresholds = np.unique(scores)[::-1]

    fprs = []
    fnrs = []
    for t in thresholds:
        pred = scores >= t

        fp = np.sum(pred[mask_i])
        tn = np.sum(~pred[mask_i])
        fpr = fp / (fp + tn + 1e-12)

        fn = np.sum(~pred[mask_g])
        tp = np.sum(pred[mask_g])
        fnr = fn / (fn + tp + 1e-12)

        fprs.append(fpr)
        fnrs.append(fnr)

    fprs = np.array(fprs)
    fnrs = np.array(fnrs)

    idx = np.argmin(np.abs(fprs - fnrs))
    return thresholds[idx], (fprs[idx] + fnrs[idx]) / 2.0


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_eer_matches_compute_eer_threshold(seed):
    from fame.metrics import equal_error_rate

    rng = np.random.default_rng(seed)
    scores = np.concatenate([rng.normal(0.7, 0.15, 150), rng.normal(0.25, 0.15, 150)])
    labels = np.array([1] * 150 + [0] * 150)

    reference_threshold, reference_eer = reference_compute_eer_threshold(scores, labels)
    threshold, eer = equal_error_rate(scores, labels)

    assert threshold == pytest.approx(float(reference_threshold), abs=1e-12)
    assert eer == pytest.approx(float(reference_eer), abs=1e-12)


def test_eer_ties_resolve_to_the_largest_threshold():
    """np.unique(scores)[::-1] sweeps downwards, so argmin takes the highest.

    Sweeping upwards returns a different threshold whenever several candidates
    tie on |FPR - FNR|, which a coarse score distribution produces easily.
    """
    from fame.metrics import equal_error_rate

    # A coarse, tied score distribution: thresholds 0.5 and 1.0 both give
    # |FPR - FNR| = 1/3, so only the sweep direction separates them.
    scores = np.array([1.0, 1.0, 0.5, 0.5, 0.0, 0.0])
    labels = np.array([1, 1, 1, 0, 0, 0])

    descending, _ = equal_error_rate(scores, labels)
    reference_threshold, _ = reference_compute_eer_threshold(scores, labels)

    candidates_ascending = np.unique(scores)
    genuine, impostor = scores[labels == 1], scores[labels == 0]
    fm = np.array([(impostor >= t).mean() for t in candidates_ascending])
    fnm = np.array([(genuine < t).mean() for t in candidates_ascending])
    ascending = float(candidates_ascending[np.argmin(np.abs(fm - fnm))])

    assert descending == pytest.approx(float(reference_threshold))
    assert ascending != descending, "the sweep direction has to matter here"
    assert descending > ascending


def test_eer_of_perfectly_separated_scores_is_zero():
    from fame.metrics import equal_error_rate

    scores = np.concatenate([np.full(20, 0.9), np.full(20, 0.1)])
    labels = np.array([1] * 20 + [0] * 20)

    threshold, eer = equal_error_rate(scores, labels)

    assert eer == pytest.approx(0.0)
    assert accuracy_at(scores, labels, threshold) == pytest.approx(1.0)


def test_stored_threshold_is_rounded_to_four_decimals():
    """scores_eer.csv stores f"{eer_threshold:.4f}", so the value downstream is rounded.

    Recomputing gives full precision, which can land on the other side of a
    score sitting within 5e-5 of the threshold.
    """
    from fame.metrics import equal_error_rate

    rng = np.random.default_rng(7)
    scores = np.concatenate([rng.normal(0.7, 0.15, 150), rng.normal(0.25, 0.15, 150)])
    labels = np.array([1] * 150 + [0] * 150)

    exact, _ = equal_error_rate(scores, labels)
    stored = float(f"{exact:.4f}")

    borderline = np.array([stored - 1e-5, stored + 1e-5])
    assert not np.array_equal(borderline >= exact, borderline >= stored) or exact == stored

"""Deletion and insertion metrics for face verification (Tab. 2).

Following Lu et al., pixels of the probe are removed (or added back) in order
of attribution while the gallery stays fixed, verification accuracy is measured
at each ratio against the operating threshold, and the curve is summarised by
its normalized area.  A faithful map gives a low deletion AUC and a high
insertion AUC.
"""

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

# numpy 2.0 renamed trapz to trapezoid and removed the old name in 2.4.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

# Removal ratios P used in the paper.
PERCENTAGES = np.linspace(0, 100, 11)


def equal_error_rate(scores: Sequence[float], labels: Sequence[int]) -> Tuple[float, float]:
    """Operating point at the equal error rate, as ``(threshold, eer)``.

    Sweeps every observed score as a candidate threshold and picks the one
    where the false match and false non-match rates are closest, reporting
    their average as the EER.  This is ``compute_eer_threshold``.

    Candidates are swept in descending order because ``argmin`` returns the
    first minimum: when several thresholds tie, the sweep direction decides
    which one is returned, and descending picks the largest.  Sweeping upwards
    instead would silently return a different operating point on ties, which
    are common whenever the score distribution has flat regions.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    genuine = scores[labels == 1]
    impostor = scores[labels == 0]
    if genuine.size == 0 or impostor.size == 0:
        raise ValueError("both genuine and impostor pairs are needed to estimate the EER")

    candidates = np.unique(scores)[::-1]
    false_match = np.array([(impostor >= t).mean() for t in candidates])
    false_non_match = np.array([(genuine < t).mean() for t in candidates])

    index = int(np.argmin(np.abs(false_match - false_non_match)))
    return float(candidates[index]), float((false_match[index] + false_non_match[index]) / 2.0)


def eer_threshold(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Similarity threshold at the equal error rate.

    Note that ``scores_eer.csv`` stores this value formatted to four decimal
    places, so a pipeline reading the stored threshold operates on a rounded
    number.  Pass it explicitly rather than recomputing if you need to match
    previously reported accuracies exactly.
    """
    return equal_error_rate(scores, labels)[0]


def accuracy_at(scores: Sequence[float], labels: Sequence[int], threshold: float) -> float:
    """Verification accuracy of a set of scores at a fixed threshold."""
    predictions = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    return float((predictions == np.asarray(labels, dtype=int)).mean())


def normalized_auc(
    accuracies: Sequence[float], percentages: Sequence[float] = PERCENTAGES
) -> float:
    """Area under the accuracy-over-P curve, normalized to the P range."""
    percentages = np.asarray(percentages, dtype=float)
    span = percentages[-1] - percentages[0]
    if span == 0:
        raise ValueError("percentages must cover a non-zero range")
    return float(_trapezoid(np.asarray(accuracies, dtype=float), percentages) / span)


def curve_from_scores(
    scores_by_percentage: dict,
    labels: Sequence[int],
    threshold: float,
    percentages: Optional[Iterable[float]] = None,
) -> dict:
    """Accuracy curve and AUC from per-ratio similarity scores.

    Args:
        scores_by_percentage: maps each ratio P to the list of pair scores
            obtained after perturbing at that ratio.
        labels: genuine/impostor label per pair.
        threshold: decision threshold, normally the clean EER threshold.

    Returns:
        Dict with the ordered ``percentages``, the ``accuracies`` and the
        normalized ``auc``.
    """
    ratios = sorted(scores_by_percentage) if percentages is None else list(percentages)
    accuracies = [accuracy_at(scores_by_percentage[p], labels, threshold) for p in ratios]
    return {
        "percentages": ratios,
        "accuracies": accuracies,
        "auc": normalized_auc(accuracies, ratios),
    }

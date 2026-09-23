"""FAME: Feature Activation Map Explanation.

Reference implementation of

    Xinyi Zhang and Manuel Guenther, "FAME: Feature Activation Map Explanation
    on Image Classification and Face Recognition", CVPR Workshops 2026.
"""

from .attribution import fame_attribution, normalize, smooth_and_normalize, to_grayscale
from .explain import FameConfig, explain_classification, explain_feature_map, explain_verification
from .optimizer import fame

__version__ = "0.1.0"

__all__ = [
    "FameConfig",
    "explain_classification",
    "explain_feature_map",
    "explain_verification",
    "fame",
    "fame_attribution",
    "normalize",
    "smooth_and_normalize",
    "to_grayscale",
]

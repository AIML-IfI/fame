"""Face recognition backbones and checkpoint loading."""

from typing import Dict

import torch

from .iresnet import build_model
from .wrappers import FaceEmbeddingModel

# Maps the names used throughout the scripts to the AdaFace architecture names.
FACE_MODELS: Dict[str, str] = {
    "Adaface_ir_18": "ir_18",
    "Adaface_ir_50": "ir_50",
    "Adaface_ir_101": "ir_101",
}


def load_adaface_state_dict(backbone, checkpoint_path: str):
    """Load an AdaFace Lightning checkpoint into a bare backbone."""
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    # The checkpoint stores the backbone under a "model." prefix alongside the
    # margin head, which is not needed for inference.
    weights = {key[6:]: value for key, value in state_dict.items() if key.startswith("model.")}
    backbone.load_state_dict(weights)
    return backbone


def build_face_model(
    name: str,
    checkpoint_path: str,
    device: str = "cuda",
    l2_normalize: bool = False,
) -> FaceEmbeddingModel:
    """Build a face embedding model that consumes [0, 1] BGR images.

    The channel order matters: AdaFace was trained on BGR, so the pair loader
    swaps channels before the image reaches this model.  See
    ``fame.data.pairs``.
    """
    if name not in FACE_MODELS:
        raise KeyError(f"unknown face model {name!r}, expected one of {list(FACE_MODELS)}")
    backbone = build_model(model_name=FACE_MODELS[name])
    backbone = load_adaface_state_dict(backbone, checkpoint_path)
    model = FaceEmbeddingModel(backbone, l2_normalize=l2_normalize)
    return model.to(device).eval()


def face_target_layer(model: FaceEmbeddingModel):
    """Last convolutional block of an IResNet backbone, used by CAM baselines."""
    return model.backbone.model.body[-1]

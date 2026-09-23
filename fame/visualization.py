"""Heatmap colouring and overlays.

Attribution maps are stored as single-channel float arrays; colouring happens
only at display time.  The original code saved the jet-coloured RGB array to
``.npy`` and the evaluation scripts then reloaded it as if it were a raw map,
blurring and thresholding a colormap, so the deletion order did not follow the
attribution values.  Keeping colour out of the saved artefacts avoids that.
"""

from typing import Union

import numpy as np
import torch
from matplotlib import colormaps

Array = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: Array) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def heatmap(attribution: Array, cmap: str = "jet") -> np.ndarray:
    """Colour a single (H, W) attribution map, returning float RGB in [0, 1]."""
    values = np.squeeze(_to_numpy(attribution))
    if values.ndim != 2:
        raise ValueError(f"expected a single 2D map, got shape {values.shape}")
    return colormaps[cmap](values)[..., :3]


def overlay(image: Array, attribution: Array, alpha: float = 0.5, cmap: str = "jet") -> np.ndarray:
    """Blend a heatmap over an image, returning uint8 RGB of shape (H, W, 3).

    Args:
        image: (3, H, W) or (H, W, 3) image with values in [0, 1].
        attribution: single-channel map in [0, 1].
        alpha: weight of the heatmap; 0 shows the image, 1 shows the heatmap.
    """
    picture = _to_numpy(image)
    if picture.ndim == 3 and picture.shape[0] == 3:
        picture = picture.transpose(1, 2, 0)

    blended = (1.0 - alpha) * picture + alpha * heatmap(attribution, cmap=cmap)
    return np.uint8(255 * np.clip(blended, 0.0, 1.0))


def bgr_to_rgb(image: Array) -> np.ndarray:
    """Flip channel order of a (3, H, W) tensor, for displaying face crops."""
    picture = _to_numpy(image)
    return picture[::-1] if picture.shape[0] == 3 else picture[..., ::-1]


def contact_sheet(
    panels,
    row_labels=None,
    column_labels=None,
    title: str = None,
    panel_size: float = 1.4,
):
    """Lay out a grid of already-rendered images and return the figure.

    Used for every figure in the supplemental material: the feature map sheets
    of Fig. 7 and 8, where a panel is one feature map location, and the method
    comparisons of Fig. 9 to 12, where rows are networks and columns are XAI
    methods.

    Args:
        panels: nested sequence, ``panels[row][column]``, of uint8 or float RGB
            images.  ``None`` leaves a cell blank, which is how a method that
            produces no dissimilar map is shown.
        row_labels: label drawn to the left of each row.
        column_labels: label drawn above the top row.
        title: figure heading.
        panel_size: size of one panel in inches.

    Returns:
        A matplotlib figure; the caller saves and closes it.
    """
    import matplotlib.pyplot as plt

    rows = len(panels)
    columns = max(len(row) for row in panels)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(panel_size * columns, panel_size * rows),
        squeeze=False,
    )

    for row_index in range(rows):
        for column_index in range(columns):
            axis = axes[row_index][column_index]
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)

            panel = None
            if column_index < len(panels[row_index]):
                panel = panels[row_index][column_index]
            if panel is not None:
                axis.imshow(_to_numpy(panel))

            if column_labels is not None and row_index == 0 and column_index < len(column_labels):
                axis.set_title(column_labels[column_index], fontsize=8)
            if column_index == 0 and row_labels is not None and row_index < len(row_labels):
                axis.set_ylabel(row_labels[row_index], fontsize=8, rotation=90)

    if title:
        figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    return figure


def mark_receptive_field(image: Array, location, feature_shape, colour=1.0, width: int = 1):
    """Outline the pixels lying underneath one feature map location.

    Fig. 1 draws these white borders to make the point visible: the attribution
    routinely extends well outside the box that CAM's upsampling assumes.

    Args:
        image: (H, W, 3) RGB panel to draw on.
        location: ``(row, col)`` index into the feature map.
        feature_shape: ``(H_a, W_a)`` size of the feature map.
        colour: value written into all three channels for the border.
        width: border thickness in pixels.
    """
    picture = _to_numpy(image).copy()
    height, width_px = picture.shape[:2]
    rows, columns = feature_shape
    row, column = location

    top = int(round(row * height / rows))
    bottom = int(round((row + 1) * height / rows))
    left = int(round(column * width_px / columns))
    right = int(round((column + 1) * width_px / columns))

    value = colour * (255 if picture.dtype == np.uint8 else 1.0)
    picture[top : top + width, left:right] = value
    picture[max(bottom - width, 0) : bottom, left:right] = value
    picture[top:bottom, left : left + width] = value
    picture[top:bottom, max(right - width, 0) : right] = value
    return picture

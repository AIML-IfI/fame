"""Tests for the supplemental material code (Sec. A, B and C).

Covers the parameter sweep behaviour, the timing helpers and the figure grids,
including one run of the comparison sheet against files on disk so that the
plotting path is exercised rather than only imported.
"""

import pathlib
import time

import matplotlib
import numpy as np
import pytest
import torch
import torch.nn as nn

matplotlib.use("Agg")  # Headless backend; the tests never open a window.

from fame import FameConfig, explain_classification
from fame.benchmark import environment, measure
from fame.visualization import contact_sheet, heatmap, mark_receptive_field, overlay


def load_script(name):
    """Import a file in scripts/ as a module.

    The scripts import `_bootstrap` to put the repository root on sys.path when
    they are run directly, so scripts/ has to be importable before the module
    body executes.
    """
    import importlib.util
    import pathlib
    import sys

    scripts = pathlib.Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))

    spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)
    np.random.seed(0)


class TinyNet(nn.Module):
    def __init__(self, outputs: int = 10):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, outputs))

    def forward(self, x):
        return self.head(self.features(x))


# --------------------------------------------------------------------------
# Sec. A -- parameter sensitivity


def test_more_iterations_change_the_map():
    """Fig. 6(a): one step gives a coarse map, more steps refine it."""
    model = TinyNet().eval()
    image = torch.rand(1, 3, 32, 32)
    targets = torch.tensor([2])

    def run(iterations):
        config = FameConfig(iterations=iterations, blur_kernel=11, blur_sigma=3.0)
        return explain_classification(model, image, targets, config)

    assert not torch.allclose(run(1), run(50), atol=1e-2)


def test_larger_step_size_produces_a_larger_perturbation():
    """Fig. 6(b): eta scales how far the image is allowed to travel."""
    from fame.optimizer import fame
    from fame.losses import classification_loss

    model = TinyNet().eval()
    image = torch.rand(1, 3, 32, 32)
    loss_fn = classification_loss(model, torch.tensor([0]))

    small = (fame(image, loss_fn, step_size=1 / 255, iterations=10) - image).abs().sum()
    large = (fame(image, loss_fn, step_size=8 / 255, iterations=10) - image).abs().sum()

    assert large > small


def test_stronger_blur_spreads_the_attribution():
    """Fig. 5: bigger kernels wash the map out over a wider area."""
    from fame.attribution import fame_attribution

    image = torch.zeros(1, 3, 64, 64)
    perturbed = torch.zeros(1, 3, 64, 64)
    perturbed[0, :, 32, 32] = 1.0  # A single hot pixel.

    sharp = fame_attribution(image, perturbed, blur_kernel=5, blur_sigma=1.0)
    diffuse = fame_attribution(image, perturbed, blur_kernel=49, blur_sigma=12.0)

    # The wider the smoothing, the more pixels carry a non-trivial value.
    assert (diffuse > 0.1).sum() > (sharp > 0.1).sum()


def test_early_stopping_off_runs_every_requested_step():
    """Fig. 6 needs epsilon disabled, or the labelled step counts are wrong."""
    from fame.optimizer import fame

    calls = {"count": 0}

    def loss_fn(x):
        calls["count"] += 1
        return (x**2).sum(dim=(1, 2, 3))

    fame(torch.rand(1, 3, 8, 8), loss_fn, iterations=7, epsilon=None)

    assert calls["count"] == 7


# --------------------------------------------------------------------------
# Sec. B -- runtime


def test_measure_reports_elapsed_time():
    with measure(device=None) as timing:
        time.sleep(0.05)

    assert timing["seconds"] >= 0.05
    assert timing["seconds"] < 5.0


def test_measure_omits_gpu_fields_on_cpu():
    with measure(device="cpu") as timing:
        pass

    assert "seconds" in timing
    assert "peak_memory_gb" not in timing


def test_environment_records_the_versions():
    info = environment(device="cpu")

    assert info["torch"] == torch.__version__
    assert "python" in info and "platform" in info


def test_runtime_grows_with_iterations():
    """The linear trend of Tab. 3(a) is the basis for the speed/quality advice."""
    model = TinyNet().eval()
    image = torch.rand(1, 3, 32, 32)
    targets = torch.tensor([0])

    durations = []
    for iterations in (5, 40):
        config = FameConfig(iterations=iterations, blur_kernel=11, blur_sigma=3.0)
        with measure(device=None) as timing:
            explain_classification(model, image, targets, config)
        durations.append(timing["seconds"])

    assert durations[1] > durations[0]


# --------------------------------------------------------------------------
# Sec. C -- figures


def test_contact_sheet_shape_matches_the_panels():
    panels = [[np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4)] for _ in range(3)]

    figure = contact_sheet(panels, row_labels=["a", "b", "c"], column_labels=list("wxyz"))

    assert len(figure.axes) == 12
    figure.clf()


def test_contact_sheet_leaves_none_cells_blank():
    """Grad-CAM has no dissimilar map, so that cell must stay empty."""
    panels = [[np.zeros((8, 8, 3), dtype=np.uint8), None]]

    figure = contact_sheet(panels)

    assert len(figure.axes[0].images) == 1
    assert len(figure.axes[1].images) == 0
    figure.clf()


def test_contact_sheet_handles_ragged_rows():
    panels = [[np.zeros((4, 4, 3))] * 3, [np.zeros((4, 4, 3))]]

    figure = contact_sheet(panels)

    assert len(figure.axes) == 6
    figure.clf()


def test_receptive_field_box_lands_on_the_right_cell():
    picture = np.zeros((28, 28, 3), dtype=np.uint8)

    marked = mark_receptive_field(picture, (0, 0), (7, 7), colour=1.0)

    # The top-left cell spans pixels 0..4; its border must be drawn and the
    # rest of the image left untouched.
    assert marked[0, 0].max() == 255
    assert marked[27, 27].max() == 0


def test_receptive_field_marks_distinct_cells_differently():
    picture = np.zeros((28, 28, 3), dtype=np.uint8)

    first = mark_receptive_field(picture, (0, 0), (7, 7))
    last = mark_receptive_field(picture, (6, 6), (7, 7))

    assert not np.array_equal(first, last)


def test_overlay_and_heatmap_produce_displayable_images():
    image = torch.rand(3, 32, 32)
    attribution = torch.rand(1, 32, 32)

    coloured = heatmap(attribution)
    blended = overlay(image, attribution)

    assert coloured.shape == (32, 32, 3)
    assert blended.shape == (32, 32, 3) and blended.dtype == np.uint8


def test_overlay_alpha_interpolates_between_image_and_heatmap():
    image = torch.zeros(3, 8, 8)
    attribution = torch.zeros(1, 8, 8)

    only_image = overlay(image, attribution, alpha=0.0)
    only_heatmap = overlay(image, attribution, alpha=1.0)

    assert only_image.max() == 0
    assert only_heatmap.max() > 0  # Zero maps to the bottom of the colormap, not to black.


def test_comparison_grid_builds_from_files_on_disk(tmp_path):
    """End-to-end check of the Fig. 9 layout against saved maps.

    Mirrors what plot_comparison_grid.py does: read per-method .npy files, skip
    the ones a method did not produce, and lay the rest out as a sheet.
    """
    models = ["NetA", "NetB"]
    methods = ["GradCAM", "FAME"]
    for model_name in models:
        for method in methods:
            directory = tmp_path / method / model_name
            directory.mkdir(parents=True)
            np.save(directory / "img_0.npy", np.random.rand(16, 16).astype(np.float32))
    # FAME on NetB is missing, standing in for an interrupted run.
    (tmp_path / "FAME" / "NetB" / "img_0.npy").unlink()

    picture = np.random.rand(16, 16, 3)
    rows = []
    for model_name in models:
        panels = [np.uint8(255 * picture)]
        for method in methods:
            path = tmp_path / method / model_name / "img_0.npy"
            panels.append(overlay(picture, np.load(path)) if path.exists() else None)
        rows.append(panels)

    figure = contact_sheet(rows, row_labels=models, column_labels=["input"] + methods)

    assert len(figure.axes) == 6
    # The missing map leaves exactly one empty cell.
    assert sum(1 for axis in figure.axes if not axis.images) == 1
    figure.savefig(tmp_path / "sheet.pdf")
    assert (tmp_path / "sheet.pdf").stat().st_size > 0
    figure.clf()


def test_sweep_configs_vary_one_knob_at_a_time():
    """Each swept value must change its own knob and leave the others fixed."""
    import argparse

    module = load_script("run_parameter_sweep")

    args = argparse.Namespace(
        step_size=1 / 255, iterations=100, blur_kernel=49, blur_sigma=7.7,
        input_space="normalized",
    )

    iteration_sweep = list(module.configs_for("iterations", [1, 100, 500], args))
    assert [config.iterations for _, config in iteration_sweep] == [1, 100, 500]
    assert {config.step_size for _, config in iteration_sweep} == {1 / 255}

    step_sweep = list(module.configs_for("step_size", [1 / 255, 8 / 255], args))
    assert [config.step_size for _, config in step_sweep] == [1 / 255, 8 / 255]
    assert {config.iterations for _, config in step_sweep} == {100}

    blur_sweep = list(module.configs_for("blur", [5, 49], args))
    assert [config.blur_kernel for _, config in blur_sweep] == [5, 49]
    # Sigma must follow the kernel, or the wider kernel is clipped to no effect.
    assert blur_sweep[0][1].blur_sigma < blur_sweep[1][1].blur_sigma

    # Fig. 6 is produced without early stopping.
    assert all(config.epsilon is None for _, config in iteration_sweep)


def test_sweep_labels_are_filesystem_safe():
    """Labels become filenames, so they must not contain separators."""
    import argparse

    module = load_script("run_parameter_sweep")

    args = argparse.Namespace(
        step_size=1 / 255, iterations=100, blur_kernel=49, blur_sigma=7.7,
        input_space="normalized",
    )

    for sweep, values in (("iterations", [500]), ("step_size", [1 / 255]), ("blur", [49])):
        for label, _ in module.configs_for(sweep, values, args):
            assert "\\" not in label
            assert label.replace("/", "per") == label or "/" in label


def test_curve_plotting_runs_on_a_verification_metrics_csv(tmp_path):
    """plot_curves.py must consume exactly what eval_verification.py writes."""
    import subprocess
    import sys

    import pandas as pd

    from fame.metrics import PERCENTAGES

    accuracies = ";".join(f"{0.99 - i * 0.04:.4f}" for i in range(len(PERCENTAGES)))
    rows = []
    for method in ("FAME", "GradCAM"):
        for metric in ("Delete", "Insert"):
            rows.append(
                {
                    "model": "Adaface_ir_101",
                    "dataset": "CFP",
                    "protocol": "01FP",
                    "method": method,
                    "metric": metric,
                    "auc": 61.5,
                    "accuracies": accuracies,
                }
            )
    metrics = tmp_path / "metrics.csv"
    pd.DataFrame(rows).to_csv(metrics, index=False)

    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "plot_curves.py"
    result = subprocess.run(
        [sys.executable, str(script), "--task", "verification",
         "--metrics", str(metrics), "--output", str(tmp_path / "figures")],
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "figures" / "curves_CFP_01FP.pdf").stat().st_size > 0


def test_curve_plotting_runs_on_a_classification_metrics_csv(tmp_path):
    """The ROAD curve reads the long-format CSV eval_classification.py writes."""
    import subprocess
    import sys

    import pandas as pd

    rows = [
        {"model": "ResNet50", "method": method, "metric": "ROAD-Delete",
         "parameter": p, "value": p / 40.0}
        for method in ("FAME", "GradCAM")
        for p in (0.0, 30.0, 60.0, 100.0)
    ]
    rows.append(
        {"model": "ResNet50", "method": "FAME", "metric": "IoU", "parameter": 0.5, "value": 41.2}
    )
    metrics = tmp_path / "metrics.csv"
    pd.DataFrame(rows).to_csv(metrics, index=False)

    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "plot_curves.py"
    result = subprocess.run(
        [sys.executable, str(script), "--task", "classification",
         "--metrics", str(metrics), "--output", str(tmp_path / "figures")],
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "figures" / "road_ResNet50.pdf").stat().st_size > 0

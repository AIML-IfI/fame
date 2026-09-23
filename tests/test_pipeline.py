"""Tests for the data layer, the metrics and the baselines.

The pipeline test at the end is the one that matters most: it writes maps to
disk exactly as ``run_fame_verification.py`` does and reads them back exactly
as ``eval_verification.py`` does, which is where the original code silently
disagreed with itself about what a saved ``.npy`` contains.
"""

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from fame import FameConfig, explain_verification
from fame.baselines import corrise, fggb
from fame.data import VerificationPairs, adjust_box, load_synset_mapping, parse_prediction_string
from fame.metrics import (
    PERCENTAGES,
    accuracy_at,
    curve_from_scores,
    eer_threshold,
    intersection_over_union,
    logit_drop,
    noisy_linear_imputation,
    normalized_auc,
    top_percent_mask,
)


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)
    np.random.seed(0)


class TinyEmbedding(nn.Module):
    def __init__(self, dimensions: int = 16, size: int = 32):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.ReLU())
        reduced = size // 2
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(8 * reduced * reduced, dimensions))

    def forward(self, x):
        return self.head(self.features(x))


@pytest.fixture
def protocol(tmp_path):
    """A small protocol with G/P/T columns and matching HDF5 crops."""
    crops = tmp_path / "crops"
    crops.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for index in range(6):
        for side in ("g", "p"):
            relative = f"id{index}/{side}.jpg"
            flattened = relative.replace("/", "_")[:-3] + "h5"
            with h5py.File(crops / flattened, "w") as handle:
                handle["data"] = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
        rows.append({"G": f"id{index}/g.jpg", "P": f"id{index}/p.jpg", "T": index % 2})

    csv = tmp_path / "protocol.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return str(csv), str(crops)


# --------------------------------------------------------------------------
# Data


def test_pairs_reads_named_columns(protocol):
    csv, crops = protocol
    dataset = VerificationPairs(csv, crops)

    item = dataset[0]

    assert len(dataset) == 6
    assert item["gallery"].shape == (3, 32, 32)
    assert 0.0 <= item["gallery"].min() and item["gallery"].max() <= 1.0
    assert list(dataset.labels) == [0, 1, 0, 1, 0, 1]


def test_pairs_rejects_a_protocol_without_gpt_columns(tmp_path):
    csv = tmp_path / "wrong.csv"
    pd.DataFrame([{"gallery": "a.jpg", "probe": "b.jpg", "label": 1}]).to_csv(csv, index=False)

    with pytest.raises(ValueError, match="G"):
        VerificationPairs(str(csv), str(tmp_path))


def test_pairs_bgr_flag_swaps_channels(protocol):
    csv, crops = protocol

    rgb = VerificationPairs(csv, crops, to_bgr=False)[0]["gallery"]
    bgr = VerificationPairs(csv, crops, to_bgr=True)[0]["gallery"]

    assert torch.allclose(bgr, rgb.flip(0))


def test_pairs_genuine_only_filters_impostors(protocol):
    csv, crops = protocol

    dataset = VerificationPairs(csv, crops, genuine_only=True)

    assert len(dataset) == 3
    assert set(dataset.labels) == {1}


def test_prediction_string_parsing():
    synsets, box = parse_prediction_string("n01440764 189 187 292 225 ")

    assert synsets == ["n01440764"]
    assert box == [189, 187, 292, 225]


def test_prediction_string_handles_multiple_objects():
    synsets, box = parse_prediction_string("n01 10 10 20 20 n02 30 30 40 40 ")

    assert synsets == ["n01", "n02"]
    # The box belongs to the first object, which is the one being explained.
    assert box == [10, 10, 20, 20]


def test_box_adjustment_uses_one_scale_for_both_axes():
    """Resize(232) scales by the shorter side, so the aspect ratio is kept.

    A box covering the whole image must therefore still cover the full crop
    along the shorter dimension after resizing and cropping.
    """
    landscape = adjust_box([0, 0, 500, 375], 500, 375)
    portrait = adjust_box([0, 0, 375, 500], 375, 500)

    assert landscape == [0, 0, 224, 224]
    assert portrait == [0, 0, 224, 224]


def test_box_adjustment_clips_to_the_crop():
    box = adjust_box([0, 0, 10, 10], 500, 375)

    assert box[0] >= 0 and box[1] >= 0
    assert box[2] <= 224 and box[3] <= 224


def test_synset_mapping_indexes_in_file_order(tmp_path):
    path = tmp_path / "synsets.txt"
    path.write_text("n01440764 tench\nn01443537 goldfish\n")

    mapping = load_synset_mapping(str(path))

    assert mapping == {"n01440764": 0, "n01443537": 1}


# --------------------------------------------------------------------------
# Metrics


def test_iou_of_a_perfect_map_is_one():
    attribution = np.zeros((224, 224))
    attribution[50:150, 50:150] = 1.0

    assert intersection_over_union(attribution, [50, 50, 150, 150], 0.5) == pytest.approx(1.0)


def test_iou_of_a_disjoint_map_is_zero():
    attribution = np.zeros((224, 224))
    attribution[0:40, 0:40] = 1.0

    assert intersection_over_union(attribution, [100, 100, 150, 150], 0.5) == pytest.approx(0.0)


def test_iou_falls_as_the_threshold_rises_for_a_graded_map():
    attribution = np.tile(np.linspace(0, 1, 224), (224, 1))
    box = [0, 0, 224, 224]

    loose = intersection_over_union(attribution, box, 0.3)
    strict = intersection_over_union(attribution, box, 0.7)

    assert loose > strict


@pytest.mark.parametrize("percent", [0, 10, 30, 50, 100])
def test_masks_select_the_requested_fraction(percent):
    attribution = np.random.rand(100, 100)

    deleted = top_percent_mask(attribution, percent)
    inserted = top_percent_mask(attribution, percent, keep_top=True)

    assert deleted.mean() == pytest.approx(1 - percent / 100, abs=0.02)
    assert inserted.mean() == pytest.approx(percent / 100, abs=0.02)


def test_deletion_and_insertion_masks_are_complementary():
    attribution = np.random.rand(64, 64)

    deleted = top_percent_mask(attribution, 30)
    inserted = top_percent_mask(attribution, 30, keep_top=True)

    assert np.array_equal(deleted + inserted, np.ones_like(deleted))


def test_masks_remove_the_highest_ranked_pixels():
    attribution = np.arange(100).reshape(10, 10).astype(float)

    mask = top_percent_mask(attribution, 10)

    # The ten largest values are the last row.
    assert mask[-1].sum() == 0
    assert mask[:-1].all()


def test_imputation_keeps_known_pixels_and_fills_the_rest():
    image = torch.rand(1, 3, 16, 16)
    mask = (torch.rand(1, 1, 16, 16) > 0.3).float()

    filled = noisy_linear_imputation(image, mask, iterations=20, noise=0.0)

    assert torch.allclose(filled * mask, image * mask, atol=1e-5)
    assert torch.isfinite(filled).all()
    # Imputed pixels take neighbourhood values rather than staying at zero.
    assert not torch.allclose(filled * (1 - mask), torch.zeros_like(filled))


def test_logit_drop_is_zero_without_perturbation():
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 5)).eval()
    images = torch.rand(4, 3, 8, 8)

    drop = logit_drop(model, images, images.clone(), torch.tensor([0, 1, 2, 3]))

    assert torch.allclose(drop, torch.zeros(4), atol=1e-6)


def test_eer_threshold_separates_well_separated_scores():
    scores = np.concatenate([np.full(50, 0.8), np.full(50, 0.2)])
    labels = np.array([1] * 50 + [0] * 50)

    threshold = eer_threshold(scores, labels)

    assert 0.2 < threshold <= 0.8
    assert accuracy_at(scores, labels, threshold) == pytest.approx(1.0)


def test_eer_threshold_needs_both_classes():
    with pytest.raises(ValueError):
        eer_threshold([0.5, 0.6], [1, 1])


def test_auc_of_a_flat_curve_equals_its_level():
    assert normalized_auc([0.75] * 11, PERCENTAGES) == pytest.approx(0.75)


def test_deletion_curve_falls_when_scores_degrade():
    labels = np.array([1] * 50 + [0] * 50)
    clean = np.concatenate([np.full(50, 0.8), np.full(50, 0.2)])
    threshold = eer_threshold(clean, labels)

    # Genuine scores collapse as more of the probe is removed.
    scores = {p: np.concatenate([np.full(50, 0.8 - p / 100), np.full(50, 0.2)]) for p in PERCENTAGES}
    curve = curve_from_scores(scores, labels, threshold)

    assert curve["accuracies"][0] > curve["accuracies"][-1]
    assert 0.0 <= curve["auc"] <= 1.0


# --------------------------------------------------------------------------
# Baselines


def test_baselines_return_both_polarities_at_input_resolution():
    model = TinyEmbedding().eval()
    probe, gallery = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)

    for maps in (
        fggb(model, probe, gallery, threshold=0.1),
        corrise(model, probe, gallery, num_masks=20, patch_size=8, seed=0),
    ):
        assert set(maps) == {"similar", "dissimilar"}
        for value in maps.values():
            assert value.shape == (1, 32, 32)
            assert torch.isfinite(value).all()
            assert value.min() >= 0.0


def test_corrise_is_reproducible_with_a_seed():
    model = TinyEmbedding().eval()
    probe, gallery = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)

    first = corrise(model, probe, gallery, num_masks=20, patch_size=8, seed=7)
    second = corrise(model, probe, gallery, num_masks=20, patch_size=8, seed=7)

    assert torch.allclose(first["similar"], second["similar"])


def test_fggb_threshold_shifts_the_split_between_polarities():
    model = TinyEmbedding().eval()
    probe, gallery = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)

    low = fggb(model, probe, gallery, threshold=0.0)
    high = fggb(model, probe, gallery, threshold=0.5)

    assert not torch.allclose(low["similar"], high["similar"], atol=1e-6)


# --------------------------------------------------------------------------
# End to end


def test_saved_maps_are_grayscale_and_need_no_reprocessing(protocol, tmp_path):
    """Write maps the way the run script does, read them the way eval does.

    The saved array must be a single-channel map already in [0, 1], so that the
    evaluation can rank pixels directly.  The original pipeline saved a
    jet-coloured RGB array here and re-blurred it on load.
    """
    csv, crops = protocol
    dataset = VerificationPairs(csv, crops)
    model = TinyEmbedding().eval()
    config = FameConfig(iterations=10, blur_kernel=11, blur_sigma=3.0)
    destination = tmp_path / "maps"
    destination.mkdir()

    for index in range(len(dataset)):
        item = dataset[index]
        maps = explain_verification(
            model, item["probe"][None], item["gallery"][None], config, modes=("similar",)
        )
        np.save(destination / f"{index}_p_pos.npy", maps["similar"][0].numpy()[0])

    loaded = np.load(destination / "0_p_pos.npy")
    assert loaded.ndim == 2, "saved map must be single channel, not coloured RGB"
    assert loaded.dtype == np.float32
    assert 0.0 <= loaded.min() and loaded.max() <= 1.0 + 1e-6

    # And the full deletion sweep runs on what was written.
    @torch.no_grad()
    def similarity(gallery, probe):
        a, b = model(gallery), model(probe)
        a = a / a.norm(dim=1, keepdim=True)
        b = b / b.norm(dim=1, keepdim=True)
        return float((a * b).sum())

    clean = [
        similarity(dataset[i]["gallery"][None], dataset[i]["probe"][None])
        for i in range(len(dataset))
    ]
    threshold = eer_threshold(clean, dataset.labels)

    scores = {}
    for percentage in PERCENTAGES:
        step = []
        for index in range(len(dataset)):
            item = dataset[index]
            attribution = np.load(destination / f"{index}_p_pos.npy")
            mask = torch.from_numpy(top_percent_mask(attribution, percentage)).float()
            step.append(similarity(item["gallery"][None], item["probe"][None] * mask))
        scores[float(percentage)] = step

    curve = curve_from_scores(scores, dataset.labels, threshold)
    assert len(curve["accuracies"]) == len(PERCENTAGES)
    assert 0.0 <= curve["auc"] <= 1.0
    # Removing everything must not leave the probe untouched.
    assert scores[100.0] != scores[0.0]


def test_fggb_threshold_is_spread_across_the_embedding():
    """v_i sum to the cosine similarity, so theta must be divided by D.

    Subtracting a whole EER threshold from every v_i drives all the weights
    negative, which empties e_+ and makes e_- cover the image. The default must
    not do that.
    """
    model = TinyEmbedding(dimensions=64).eval()
    probe, gallery = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)

    default = fggb(model, probe, gallery, threshold=0.3)
    degenerate = fggb(model, probe, gallery, threshold=0.3, per_dimension=True)

    assert default["similar"].max() > 0, "e_+ must survive a realistic threshold"
    assert degenerate["similar"].max() == 0, "per-dimension subtraction should empty e_+"


def test_fggb_polarities_are_disjoint():
    """A pixel supports the match or opposes it, never both."""
    model = TinyEmbedding().eval()
    probe, gallery = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)

    maps = fggb(model, probe, gallery, threshold=0.1)

    assert (maps["similar"] * maps["dissimilar"]).max() == 0


def test_corrise_rank_transform_changes_the_result():
    """The original correlates ranked scores, making this Spearman."""
    model = TinyEmbedding().eval()
    probe, gallery = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)

    spearman = corrise(model, probe, gallery, num_masks=40, patch_size=8, seed=1, rank_based=True)
    pearson = corrise(model, probe, gallery, num_masks=40, patch_size=8, seed=1, rank_based=False)

    assert not torch.allclose(spearman["similar"], pearson["similar"], atol=1e-4)
    assert torch.isfinite(spearman["similar"]).all()


def test_baseline_smoothing_matches_the_original_settings():
    """FGGB maps were blurred with kernel 25 / sigma 5; CorrRISE was not."""
    from fame import smooth_and_normalize

    attribution = torch.zeros(1, 32, 32)
    attribution[0, 16, 16] = 1.0

    blurred = smooth_and_normalize(attribution, blur_kernel=25, blur_sigma=5.0)
    unblurred = smooth_and_normalize(attribution, blur_kernel=None)

    assert (blurred > 0.05).sum() > (unblurred > 0.05).sum()
    assert blurred.max() == pytest.approx(1.0, abs=1e-5)
    assert unblurred.max() == pytest.approx(1.0, abs=1e-5)


def test_verification_scores_script_produces_the_expected_tables(protocol, tmp_path, monkeypatch):
    """compute_verification_scores.py must write both output tables.

    Exercised with a stub backbone so no checkpoint is needed; what is being
    checked is the file layout and the columns, which the FGGB threshold and
    the deletion evaluation both read.
    """
    import importlib.util
    import pathlib
    import sys

    csv, crops = protocol
    protocol_dir = tmp_path / "protocols" / "CFP"
    protocol_dir.mkdir(parents=True)
    for name in ("01FF", "01FP"):
        (protocol_dir / f"{name}.csv").write_text(pathlib.Path(csv).read_text())

    scripts = pathlib.Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "compute_verification_scores", scripts / "compute_verification_scores.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    stub = TinyEmbedding().eval()
    monkeypatch.setattr(module, "build_face_model", lambda *a, **k: stub)
    output = tmp_path / "scores"
    monkeypatch.setattr(
        sys, "argv",
        ["compute_verification_scores.py", "--images-root", crops,
         "--protocol-dir", str(tmp_path / "protocols"),
         "--checkpoints", "Adaface_ir_101=/unused", "--datasets", "CFP",
         "--output", str(output), "--device", "cpu"],
    )

    module.main()

    per_protocol = output / "Adaface_ir_101" / "CFP_01FP.csv"
    assert per_protocol.exists()
    scores = pd.read_csv(per_protocol)
    assert list(scores.columns) == ["Sim-Score", "targets"]
    assert len(scores) == 6

    summary = pd.read_csv(output / "scores_eer.csv")
    assert set(["model_name", "data_name", "protocol", "thr", "eer"]).issubset(summary.columns)
    assert len(summary) == 2  # both CFP protocols
    # The threshold is stored rounded to four decimals, as scores_eer.csv does.
    assert all(len(str(t).split(".")[1]) <= 4 for t in summary["thr"])

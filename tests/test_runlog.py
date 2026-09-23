"""Tests for run logging.

Every script writes a log and a manifest, so these check the things that make
a result directory self-explanatory: that the log lands where it says, that
stage timings are recorded, and that a crashed run still leaves a record of how
far it got.
"""

import argparse
import json
import os
import time

import pytest

from fame.runlog import RunLogger, _format_duration


def read_log(logger):
    with open(logger.log_path) as handle:
        return handle.read()


def read_manifest(logger):
    with open(logger.manifest_path) as handle:
        return json.load(handle)


def test_log_and_manifest_are_written_under_the_output_directory(tmp_path):
    logger = RunLogger(str(tmp_path), "demo", device="cpu")
    logger.finish()

    assert logger.log_path.startswith(os.path.join(str(tmp_path), "logs"))
    assert os.path.exists(logger.log_path)
    assert os.path.exists(logger.manifest_path)


def test_a_csv_destination_logs_beside_it(tmp_path):
    """Evaluation scripts pass a file path, not a directory."""
    destination = tmp_path / "results" / "metrics.csv"
    destination.parent.mkdir()

    logger = RunLogger(str(destination), "eval", device="cpu")
    logger.finish()

    assert logger.log_path.startswith(str(destination.parent))


def test_stages_are_timed_and_recorded(tmp_path):
    logger = RunLogger(str(tmp_path), "demo", device="cpu")

    with logger.stage("ResNet50"):
        time.sleep(0.05)
    logger.finish()

    manifest = read_manifest(logger)
    assert "ResNet50" in manifest["stages"]
    assert manifest["stages"]["ResNet50"]["seconds"] >= 0.05
    assert manifest["total_seconds"] >= 0.05
    assert "[ResNet50] done in" in read_log(logger)


def test_item_counts_give_a_per_item_rate(tmp_path):
    logger = RunLogger(str(tmp_path), "demo", device="cpu")

    with logger.stage("VGG19"):
        time.sleep(0.02)
    logger.count("VGG19", 50)
    logger.finish()

    assert read_manifest(logger)["stages"]["VGG19"]["items"] == 50
    assert "50 items" in read_log(logger)


def test_arguments_and_environment_land_in_the_manifest(tmp_path):
    args = argparse.Namespace(models=["ResNet50"], iterations=500)

    logger = RunLogger(str(tmp_path), "demo", args, device="cpu")
    logger.finish()

    manifest = read_manifest(logger)
    assert manifest["arguments"]["iterations"] == "500"
    assert manifest["environment"]["device"] == "cpu"
    assert "torch" in manifest["environment"]
    assert manifest["command"]


def test_a_failed_run_still_records_what_it_finished(tmp_path):
    """A partial result directory must explain why it is partial."""
    with pytest.raises(FileNotFoundError):
        with RunLogger(str(tmp_path), "demo", device="cpu") as logger:
            with logger.stage("ARface/frontal"):
                time.sleep(0.01)
            raise FileNotFoundError("missing 12_p_pos.npy")

    manifest = read_manifest(logger)
    assert manifest["status"] == "failed"
    assert "ARface/frontal" in manifest["stages"]
    assert "missing 12_p_pos.npy" in read_log(logger)


def test_a_stage_that_raises_is_still_timed(tmp_path):
    with pytest.raises(RuntimeError):
        with RunLogger(str(tmp_path), "demo", device="cpu") as logger:
            with logger.stage("crashing"):
                raise RuntimeError("out of memory")

    assert read_manifest(logger)["stages"]["crashing"]["seconds"] >= 0


def test_two_runs_do_not_overwrite_each_other(tmp_path):
    first = RunLogger(str(tmp_path), "demo", device="cpu")
    first.finish()
    time.sleep(1.01)  # The stamp has one-second resolution.
    second = RunLogger(str(tmp_path), "demo", device="cpu")
    second.finish()

    assert first.log_path != second.log_path
    assert len(os.listdir(os.path.join(str(tmp_path), "logs"))) == 4


def test_handlers_are_released_on_finish(tmp_path):
    """Long sessions must not accumulate open file handles."""
    logger = RunLogger(str(tmp_path), "demo", device="cpu")
    logger.finish()

    assert logger._logger.handlers == []


@pytest.mark.parametrize(
    "seconds,expected",
    [(0.5, "0.50s"), (45.0, "45.00s"), (90.0, "1m 30s"), (3725.0, "1h 02m 05s")],
)
def test_durations_read_naturally(seconds, expected):
    assert _format_duration(seconds) == expected

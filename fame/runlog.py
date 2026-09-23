"""Run logging shared by every script.

Every entry point writes the same three things, so that a result directory
explains itself months later:

* a human-readable log, to the console and to ``<output>/logs/<name>_<stamp>.log``
* timings for each stage and for the run as a whole
* a manifest JSON recording the exact arguments, the git commit and the machine

Stage timings use :func:`fame.benchmark.measure`, so they synchronise CUDA
before reading the clock and report peak GPU memory alongside the seconds.
"""

import json
import logging
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

from .benchmark import environment, measure


def _git_commit() -> Optional[str]:
    """Current commit, or None outside a repository.

    Recorded so that a result directory can be traced back to the code that
    produced it; ``-dirty`` marks uncommitted changes.
    """
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=root, stderr=subprocess.DEVNULL
        )
        return f"{commit}-dirty" if dirty else commit
    except (subprocess.SubprocessError, OSError):
        return None


def _format_duration(seconds: float) -> str:
    """Render seconds as ``1h 02m 03s``, since runs go from seconds to hours."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {whole_seconds:02d}s"
    if minutes:
        return f"{minutes}m {whole_seconds:02d}s"
    return f"{seconds:.2f}s"


class RunLogger:
    """Console and file logging with per-stage timing.

    Args:
        output_dir: where results are written; logs go to ``logs/`` beneath it.
            Pass a file path and its directory is used, so the evaluation
            scripts can hand over their CSV destination directly.
        name: short identifier for the run, used in the log filename.
        args: parsed ``argparse.Namespace``, recorded in the manifest.
        device: device string, so GPU memory is reported and identified.
    """

    def __init__(self, output_dir: str, name: str, args=None, device: Optional[str] = None):
        directory = output_dir
        if os.path.splitext(directory)[1]:
            directory = os.path.dirname(os.path.abspath(directory))
        self.output_dir = directory or "."
        self.name = name
        self.device = device
        self.timings = {}
        self._start = time.perf_counter()

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{name}_{stamp}.log")
        self.manifest_path = os.path.join(log_dir, f"{name}_{stamp}.json")

        self._logger = logging.getLogger(f"fame.{name}.{stamp}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
        for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(self.log_path)):
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

        self.environment = environment(device)
        self.environment["git_commit"] = _git_commit()
        self.args = {} if args is None else {k: str(v) for k, v in vars(args).items()}

        self.info(f"start {name}")
        self.info(f"command: {' '.join(sys.argv)}")
        self.info(f"log:     {self.log_path}")
        for key, value in self.environment.items():
            if value is not None:
                self.info(f"  {key}: {value}")

    def info(self, message: str) -> None:
        self._logger.info(message)

    def warning(self, message: str) -> None:
        self._logger.warning(message)

    @contextmanager
    def stage(self, label: str):
        """Time one stage and log its duration on the way out.

        Timings are kept even when the block raises, so a crashed run still
        shows how far it got and how long that took.
        """
        self.info(f"[{label}] start")
        entry = {}
        self.timings[label] = entry
        try:
            with measure(self.device) as timing:
                yield entry
        finally:
            entry.update(timing)
            memory = entry.get("peak_memory_gb")
            suffix = f", peak {memory:.2f} GB" if memory else ""
            self.info(f"[{label}] done in {_format_duration(entry['seconds'])}{suffix}")

    def count(self, label: str, items: int) -> None:
        """Record how many items a stage processed, to get a per-item rate."""
        entry = self.timings.setdefault(label, {})
        entry["items"] = items
        if entry.get("seconds"):
            self.info(f"[{label}] {items} items, {entry['seconds'] / items:.4f} s/item")

    def finish(self, status: str = "completed") -> float:
        """Log the total runtime and write the manifest.  Returns the seconds."""
        total = time.perf_counter() - self._start
        self.info(f"{status} in {_format_duration(total)} (total {total:.2f} s)")

        manifest = {
            "name": self.name,
            "status": status,
            "command": sys.argv,
            "arguments": self.args,
            "environment": self.environment,
            "total_seconds": round(total, 3),
            "stages": {
                label: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in entry.items()}
                for label, entry in self.timings.items()
            },
        }
        with open(self.manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2)
        self.info(f"manifest: {self.manifest_path}")

        for handler in list(self._logger.handlers):
            handler.close()
            self._logger.removeHandler(handler)
        return total

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            # Log the failure before the traceback reaches the console, so the
            # log file records why a partial result directory is partial.
            self._logger.error(f"failed: {exc_type.__name__}: {exc}")
            self.finish(status="failed")
        else:
            self.finish()
        return False

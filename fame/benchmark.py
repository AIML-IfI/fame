"""Timing helpers for the runtime evaluation of the supplemental material.

Tab. 3 reports wall-clock seconds for whole runs, and states that all methods
were measured sequentially so that the comparison is fair -- FAME batches
naturally while the reference implementations of CorrRISE and FGGB do not, so
batching only FAME would flatter it.  :func:`measure` therefore takes a mode
argument rather than always doing the fastest thing.
"""

import platform
import time
from contextlib import contextmanager
from typing import Optional

import torch


@contextmanager
def measure(device: Optional[str] = None):
    """Measure wall-clock time and peak GPU memory of a block.

    Yields a dict that is filled in on exit with ``seconds`` and, when a CUDA
    device is given, ``peak_memory_gb``.  CUDA work is asynchronous, so the
    timer synchronises before reading the clock; without that the measurement
    would end while kernels are still queued.
    """
    result = {}
    use_cuda = device is not None and str(device).startswith("cuda") and torch.cuda.is_available()

    if use_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    try:
        yield result
    finally:
        if use_cuda:
            torch.cuda.synchronize(device)
        result["seconds"] = time.perf_counter() - start
        if use_cuda:
            result["peak_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3


def environment(device: Optional[str] = None) -> dict:
    """Describe the machine, so runtime tables can be compared across hosts."""
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": str(device),
    }
    if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(device)
        info["gpu"] = properties.name
        info["gpu_memory_gb"] = round(properties.total_memory / 1024**3, 2)
    return info

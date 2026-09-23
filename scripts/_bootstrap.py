"""Path setup for running the scripts directly.

Calling `ensure_repository_on_path()` before the `fame` imports lets
`python scripts/run_fame_classification.py ...` work from a fresh checkout,
without setting PYTHONPATH or installing anything. `pip install -e .` also
works and makes the call a no-op.
"""

import os
import sys


def ensure_repository_on_path() -> str:
    """Prepend the repository root to sys.path and return it."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root

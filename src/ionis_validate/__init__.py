"""IONIS V20 HF Propagation Model — Validation Suite"""
__version__ = "0.3.0"

import os as _os

_PKG_DIR = _os.path.dirname(_os.path.abspath(__file__))


def _data_path(filename):
    """Return absolute path to a package data file."""
    return _os.path.join(_PKG_DIR, "data", filename)

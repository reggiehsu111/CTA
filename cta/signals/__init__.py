"""
mtx.cta.signals — one file per signal, auto-discovered on import.

Usage:
    from cta import signals as _  # populates SIGNAL_REGISTRY
    from cta.signals._base import SIGNAL_REGISTRY, VARIANT_REGISTRY
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from ._base import SIGNAL_REGISTRY, VARIANT_REGISTRY, Signal, register  # noqa: F401


def _discover():
    """Import every *.py sibling so their @register calls run."""
    pkg_dir = Path(__file__).resolve().parent
    for m in pkgutil.iter_modules([str(pkg_dir)]):
        if m.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{m.name}")


_discover()

__all__ = ["SIGNAL_REGISTRY", "VARIANT_REGISTRY", "Signal", "register"]

"""Test isolation for optional hardware SDKs.

The protocol unit tests exercise parameter and correction logic only.  Stub SDKs
that are unavailable in lightweight development environments so importing an
operation does not require instrument-control packages.
"""

from __future__ import annotations

import importlib.util
import sys
from unittest.mock import MagicMock


def _stub_missing_package(root: str, modules: tuple[str, ...]) -> None:
    if importlib.util.find_spec(root) is not None:
        return
    for module in modules:
        sys.modules[module] = MagicMock(name=module)


_stub_missing_package("Pyro4", ("Pyro4",))
_stub_missing_package(
    "qm",
    (
        "qm",
        "qm.qua",
        "qm.octave",
        "qm.QuantumMachinesManager",
    ),
)
_stub_missing_package(
    "qualang_tools",
    (
        "qualang_tools",
        "qualang_tools.loops",
        "qualang_tools.results",
        "qualang_tools.units",
    ),
)

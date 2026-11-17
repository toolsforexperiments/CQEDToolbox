from __future__ import annotations

from dataclasses import dataclass

import pytest

from labcore.protocols.base import PlatformTypes
from cqedtoolbox.protocols import parameters


@dataclass
class ScalarParameter:
    value: object

    def __call__(self, value=None):
        if value is None:
            return self.value
        self.value = value
        return value


def resolve_nested(root, path: str):
    node = root
    try:
        for part in path.split("."):
            node = node[part]
    except KeyError as exc:
        raise AttributeError(path) from exc
    return node


@pytest.fixture
def qick_params(monkeypatch):
    params = {
        "active": {"qubit": ScalarParameter("q01")},
        "q01": {
            "pulses": {
                "pi": {
                    "flattop_gain": ScalarParameter(0.2),
                    "flattop_length": ScalarParameter(0.1),
                    "gaussian_amp": ScalarParameter(0.3),
                    "gaussian_sigma": ScalarParameter(0.04),
                }
            }
        },
    }
    monkeypatch.setattr(parameters, "nestedAttributeFromString", resolve_nested)
    return params


def make_qick_parameter(cls, params):
    return cls(params, platform_type=PlatformTypes.QICK)


def test_missing_shape_defaults_to_flattop(qick_params):
    gain = make_qick_parameter(parameters.QubitGain, qick_params)
    length = make_qick_parameter(parameters.PiLength, qick_params)

    assert gain() == 0.2
    assert length() == 0.1

    gain(0.25)
    length(0.12)

    pi = qick_params["q01"]["pulses"]["pi"]
    assert pi["flattop_gain"]() == 0.25
    assert pi["flattop_length"]() == 0.12
    assert pi["gaussian_amp"]() == 0.3
    assert pi["gaussian_sigma"]() == 0.04


def test_parameters_follow_shape_changes_after_construction(qick_params):
    pi = qick_params["q01"]["pulses"]["pi"]
    pi["pulse_shape"] = ScalarParameter("flattop")
    gain = make_qick_parameter(parameters.QubitGain, qick_params)
    length = make_qick_parameter(parameters.PiLength, qick_params)

    assert gain() == 0.2
    assert length() == 0.1

    pi["pulse_shape"](" Gaussian ")
    assert gain() == 0.3
    assert length() == 0.04

    gain(0.35)
    length(0.05)
    assert pi["gaussian_amp"]() == 0.35
    assert pi["gaussian_sigma"]() == 0.05
    assert pi["flattop_gain"]() == 0.2
    assert pi["flattop_length"]() == 0.1


def test_invalid_shape_raises(qick_params):
    qick_params["q01"]["pulses"]["pi"]["pulse_shape"] = ScalarParameter("square")
    gain = make_qick_parameter(parameters.QubitGain, qick_params)

    with pytest.raises(ValueError, match="Unknown pi pulse shape"):
        gain()


def test_opx_pi_length_uses_pi_len(monkeypatch):
    params = {
        "active": {"qubit": ScalarParameter("q01")},
        "q01": {"pulses": {"pi": {"len": ScalarParameter(40)}}},
    }
    monkeypatch.setattr(parameters, "nestedAttributeFromString", resolve_nested)
    length = parameters.PiLength(params, platform_type=PlatformTypes.OPX)

    assert length() == 40
    length(48)
    assert params["q01"]["pulses"]["pi"]["len"]() == 48

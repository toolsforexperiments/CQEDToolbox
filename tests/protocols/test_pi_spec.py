from __future__ import annotations

from pathlib import Path

import pytest

from labcore.protocols.base import PlatformTypes

from cqedtoolbox.measurement_lib.qick import single_transmon_v2
from cqedtoolbox.measurement_lib.qick.single_transmon_v2 import (
    PI_SPEC_PULSE_FACTOR,
    add_pi_envelope,
)
from cqedtoolbox.protocols.operations.single_qubit import pi_spec


class MutableParameter:
    def __init__(self, value):
        self.value = value

    def __call__(self, value=None):
        if value is None:
            return self.value
        self.value = value
        return value


def make_operation():
    operation = object.__new__(pi_spec.PiSpectroscopy)
    operation.name = "PiSpectroscopy"
    operation.qubit_gain = MutableParameter(0.2)
    operation.pi_length = MutableParameter(0.1)
    return operation


class RecordingProgram:
    def __init__(self):
        self.gaussians = []
        self.cosines = []

    def add_gauss(self, **kwargs):
        self.gaussians.append(kwargs)

    def add_cosine(self, **kwargs):
        self.cosines.append(kwargs)


def test_qick_pi_spec_does_not_mutate_protocol_parameters(monkeypatch):
    operation = make_operation()
    sweep = object()
    monkeypatch.setattr(pi_spec, "PiSpecProgram", lambda: sweep)

    def run(actual_sweep, dataset_name, operation_name):
        assert actual_sweep is sweep
        assert operation.qubit_gain() == pytest.approx(0.2)
        assert operation.pi_length() == pytest.approx(0.1)
        return Path("data"), object()

    monkeypatch.setattr(pi_spec, "run_and_save_sweep", run)

    assert operation._measure_qick() == Path("data")
    assert operation.qubit_gain() == pytest.approx(0.2)
    assert operation.pi_length() == pytest.approx(0.1)


def test_qick_pi_spec_leaves_parameters_unchanged_after_measurement_error(monkeypatch):
    operation = make_operation()
    monkeypatch.setattr(pi_spec, "PiSpecProgram", lambda: object())

    def fail(*args, **kwargs):
        raise RuntimeError("measurement failed")

    monkeypatch.setattr(pi_spec, "run_and_save_sweep", fail)

    with pytest.raises(RuntimeError, match="measurement failed"):
        operation._measure_qick()

    assert operation.qubit_gain() == pytest.approx(0.2)
    assert operation.pi_length() == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("length_scale", "expected_sigma", "expected_gain"),
    [
        (10, 1.0, 0.02),
        (1, 0.1, 0.2),
    ],
)
def test_gaussian_length_scale(length_scale, expected_sigma, expected_gain, caplog):
    program = RecordingProgram()
    cfg = {
        "q_pi_pulse_shape": "gaussian",
        "q_pi_gaussian_sigma": 0.1,
        "q_pi_gaussian_nsigma": 4,
        "q_pi_gaussian_amp": 0.2,
    }

    with caplog.at_level("INFO"):
        pulse_kwargs, gain = add_pi_envelope(
            program, cfg, ch=2, prefix="test", label="Test pulse",
            length_scale=length_scale,
        )

    assert program.gaussians == [{
        "ch": 2,
        "name": f"test_gauss",
        "sigma": pytest.approx(expected_sigma),
        "length": pytest.approx(4 * expected_sigma),
        "even_length": True,
    }]
    assert pulse_kwargs == {"style": "arb", "envelope": f"test_gauss"}
    assert gain == pytest.approx(expected_gain)
    assert f"gain={expected_gain}" in caplog.text


@pytest.mark.parametrize(
    ("length_scale", "expected_length", "expected_gain"),
    [
        (10, 1.0, 0.02),
        (1, 0.1, 0.2),
    ],
)
def test_flattop_length_scale(length_scale, expected_length, expected_gain, caplog):
    program = RecordingProgram()
    cfg = {
        "q_pi_pulse_shape": "flattop",
        "q_pi_flattop_ramp": 0.05,
        "q_pi_flattop_length": 0.1,
        "q_pi_flattop_gain": 0.2,
    }

    with caplog.at_level("INFO"):
        pulse_kwargs, gain = add_pi_envelope(
            program, cfg, ch=2, prefix="test", label="Test pulse",
            length_scale=length_scale,
        )

    assert program.cosines == [{
        "ch": 2,
        "name": f"test_flattop_cosine",
        "length": pytest.approx(0.1),
        "even_length": True,
    }]
    assert pulse_kwargs == {
        "style": "flat_top",
        "envelope": f"test_flattop_cosine",
        "length": pytest.approx(expected_length),
    }
    assert gain == pytest.approx(expected_gain)
    assert f"gain={expected_gain}" in caplog.text


def test_prefix_alone_does_not_scale():
    program = RecordingProgram()
    cfg = {
        "q_pi_pulse_shape": "gaussian",
        "q_pi_gaussian_sigma": 0.1,
        "q_pi_gaussian_nsigma": 4,
        "q_pi_gaussian_amp": 0.2,
    }

    _, gain = add_pi_envelope(program, cfg, ch=2, prefix="pi_spec", label="Test pulse")

    assert program.gaussians[0]["sigma"] == pytest.approx(0.1)
    assert gain == pytest.approx(0.2)


def test_length_scale_does_not_touch_swept_gain():
    program = RecordingProgram()
    cfg = {
        "q_pi_pulse_shape": "gaussian",
        "q_pi_gaussian_sigma": 0.1,
        "q_pi_gaussian_nsigma": 4,
    }

    _, gain = add_pi_envelope(
        program, cfg, ch=2, prefix="test", label="Test pulse",
        require_gain=False, length_scale=10,
    )

    assert gain is None
    assert program.gaussians[0]["sigma"] == pytest.approx(1.0)


@pytest.mark.parametrize("length_scale", [0, -1])
def test_length_scale_must_be_positive(length_scale):
    cfg = {
        "q_pi_pulse_shape": "gaussian",
        "q_pi_gaussian_sigma": 0.1,
        "q_pi_gaussian_nsigma": 4,
        "q_pi_gaussian_amp": 0.2,
    }

    with pytest.raises(ValueError, match="length_scale"):
        add_pi_envelope(
            RecordingProgram(), cfg, ch=2, prefix="test", label="Test pulse",
            length_scale=length_scale,
        )


def test_pi_spec_program_passes_pulse_factor(monkeypatch):
    calls = []

    def fake_add_pi_envelope(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("stop after envelope")

    monkeypatch.setattr(single_transmon_v2, "add_pi_envelope", fake_add_pi_envelope)

    class FakeProgram:
        cfg = {"pi_spec_freq_steps": 3}

        def declare_gen(self, **kwargs):
            pass

        def declare_readout(self, **kwargs):
            pass

        def add_loop(self, *args):
            pass

    cfg = {"ro_adc_ch": 0, "ro_dac_ch": 1, "q_dac_ch": 2, "q_nqz": 1, "ro_nqz": 1, "ro_len": 1.0}
    # QickBoardSweep replaces the class with a sweep factory; the program class
    # is held in that factory's closure.
    program_cls = next(
        cell.cell_contents for cell in single_transmon_v2.PiSpecProgram.__closure__
        if isinstance(cell.cell_contents, type)
        and hasattr(cell.cell_contents, "_initialize")
    )
    initialize = program_cls._initialize

    with pytest.raises(RuntimeError, match="stop after envelope"):
        initialize(FakeProgram(), cfg)

    assert calls == [{"prefix": "pi_spec", "label": "Pi spectroscopy",
                      "length_scale": PI_SPEC_PULSE_FACTOR}]


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        (PlatformTypes.QICK, "pi_length 0.1 → 1.0, qubit_gain 0.2 → 0.02"),
        (PlatformTypes.OPX, None),
    ],
)
def test_report_describes_scaled_drive_pulse(platform, expected):
    operation = make_operation()
    operation.platform_type = platform

    text = operation._drive_pulse_report()

    if expected is None:
        assert text == ""
    else:
        assert expected in text
        assert f"×{PI_SPEC_PULSE_FACTOR}" in text

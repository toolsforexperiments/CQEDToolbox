from __future__ import annotations

from types import SimpleNamespace

import pytest

from labcore.protocols.base import EvaluateResult, OperationStatus, ProtocolOperation
from cqedtoolbox.protocols.operations.single_qubit import (
    pi_spec,
    power_rabi,
    res_spec,
    res_spec_vs_gain,
    sat_spec,
    t1,
    t2e,
    t2r,
)


class MutableParameter:
    def __init__(self, value):
        self.value = value

    def __call__(self, value=None):
        if value is None:
            return self.value
        self.value = value
        return value


class ResettableCorrection:
    def reset(self):
        pass


def simple_factory(cls):
    return lambda reps, factor, maximum: cls(reps, factor, maximum)


def nested_factory(cls):
    return lambda reps, factor, maximum: cls(
        reps, ResettableCorrection(), factor, maximum
    )


CORRECTION_FACTORIES = [
    simple_factory(pi_spec.IncreaseAveragingCorrection),
    simple_factory(power_rabi.IncreaseAveragingCorrection),
    nested_factory(res_spec.IncreaseAveragingCorrection),
    simple_factory(res_spec_vs_gain.IncreaseRepetitionsCorrection),
    nested_factory(sat_spec.IncreaseAveragingCorrection),
    nested_factory(t1.IncreaseAveragingCorrection),
    simple_factory(t2r.IncreaseAveragingCorrection),
    simple_factory(t2e.IncreaseAveragingCorrection),
]


@pytest.mark.parametrize("factory", CORRECTION_FACTORIES)
def test_averaging_correction_restore_returns_original_and_resets(factory):
    repetitions = MutableParameter(100)
    correction = factory(repetitions, MutableParameter(2), MutableParameter(3))

    correction.apply()
    correction.apply()
    assert repetitions() == 400

    correction.restore()
    assert repetitions() == 100
    assert correction._original_reps is None
    assert correction._count == 0

    correction.apply()
    assert repetitions() == 200


@pytest.mark.parametrize(
    ("status", "restore_calls"),
    [
        (OperationStatus.RETRY, 0),
        (OperationStatus.SUCCESS, 1),
        (OperationStatus.FAILURE, 1),
    ],
)
def test_resonator_spec_restores_only_after_terminal_status(status, restore_calls):
    operation = object.__new__(res_spec.ResonatorSpectroscopy)
    ProtocolOperation.__init__(operation)
    operation._increase_averaging = SimpleNamespace(restore=lambda: None)
    calls = 0

    def restore():
        nonlocal calls
        calls += 1

    operation._increase_averaging.restore = restore
    operation.correct(EvaluateResult(status))

    assert calls == restore_calls

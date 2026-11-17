from __future__ import annotations

from types import SimpleNamespace

from cqedtoolbox.protocols.configs import qick_config


class ScalarParameter:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


def test_default_power_rabi_timing_comes_from_pi_pulse(monkeypatch):
    values = {
        "active.qubit": "q01",
        "q01.pulses.pi.flattop_ramp": 0.05,
        "q01.pulses.pi.flattop_length": 0.1,
        "q01.scripts.power_rabi.flattop_ramp": 0.5,
        "q01.scripts.power_rabi.flattop_length": 1.0,
    }

    def getp(params, name, default=None, raise_if_missing=False):
        return values.get(name, 1)

    monkeypatch.setattr(qick_config, "getp", getp)
    monkeypatch.setattr(qick_config, "QickSweep1D", lambda *args: args)

    params = SimpleNamespace(
        qick=SimpleNamespace(
            default_reps=ScalarParameter(100),
            trig_time=ScalarParameter(0.1),
            final_delay=ScalarParameter(1.0),
            soft_avgs=ScalarParameter(1),
        )
    )
    config = object.__new__(qick_config.QickConfig)
    config.params = params

    cfg = config.config_()

    assert cfg["amprabi_flattop_ramp"] == 0.05
    assert cfg["amprabi_flattop_length"] == 0.1

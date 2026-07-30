"""
TODO LIST:

* Detuning word is now being used for both the artificial detuning and chi
* ROCalibration should store the centers of both g and e states in the parameter manager and have ProtocolParameters for them.

"""
from dataclasses import dataclass, field
from labcore.protocols.base import ProtocolParameterBase
from instrumentserver.helpers import nestedAttributeFromString


def _opx_readout_window(params):
    """(start, end) of the OPX readout frequency sweep window, absolute Hz.

    The window is not stored directly: center = readout LO + IF, and the span
    is scripts.qubit_tuneup.resonator_spec_range in units of the readout bandwidth.
    """
    q = nestedAttributeFromString(params, "active.qubit")()
    center = (nestedAttributeFromString(params, f"{q}.readout.LO")()
              + nestedAttributeFromString(params, f"{q}.readout.IF")())
    span = (nestedAttributeFromString(params, "scripts.qubit_tuneup.resonator_spec_range")()
            * nestedAttributeFromString(params, f"{q}.readout.bandwidth")())
    return center - span / 2, center + span / 2


def _opx_set_readout_window(params, start, end):
    """Write a readout window (absolute Hz) back as center (readout IF) + span (range in BW units)."""
    q = nestedAttributeFromString(params, "active.qubit")()
    lo = nestedAttributeFromString(params, f"{q}.readout.LO")()
    bw = nestedAttributeFromString(params, f"{q}.readout.bandwidth")()
    nestedAttributeFromString(params, f"{q}.readout.IF")((start + end) / 2 - lo)
    nestedAttributeFromString(params, "scripts.qubit_tuneup.resonator_spec_range")((end - start) / bw)


def _opx_qubit_window(params, range_path):
    q = nestedAttributeFromString(params, "active.qubit")()
    center = nestedAttributeFromString(params, f"{q}.IF")()
    span = nestedAttributeFromString(params, range_path)()
    return center - span / 2, center + span / 2


def _opx_set_qubit_window(params, range_path, start, end):
    q = nestedAttributeFromString(params, "active.qubit")()
    nestedAttributeFromString(params, f"{q}.IF")((start + end) / 2)
    nestedAttributeFromString(params, range_path)(end - start)


def _opx_pi_spec_step(params):
    duration = nestedAttributeFromString(params, "scripts.qubit_tuneup.pispec_pulselen")()
    linewidth = int(1.5 / duration * 1e9)
    return max(1, int(linewidth / 5))


def _opx_t2_steps(params, t2_path, oscillations_path):
    t2 = nestedAttributeFromString(params, t2_path)()
    oscillations = nestedAttributeFromString(params, oscillations_path)()
    period = max(1, int(t2 / oscillations))
    nppp = nestedAttributeFromString(params, "scripts.qubit_tuneup.t2_nppp")()
    step = max(1, period // nppp)
    return int((t2 * 3) // step)


@dataclass
class Repetition(ProtocolParameterBase):
    name: str = field(default="reps", init=False)
    description: str = field(default="Number of shots a measurement performs", init=False)

    def _dummy_getter(self):
        return self.params.reps()

    def _dummy_setter(self, value):
        return self.params.reps(value)

    def _qick_getter(self):
        return self.params.qick.default_reps()

    def _qick_setter(self, value):
        return self.params.qick.default_reps(value)

    def _opx_getter(self):
        return int(self.params.opx.default_reps())

    def _opx_setter(self, value):
        return self.params.opx.default_reps(int(value))


@dataclass
class ResonatorSpecSteps(ProtocolParameterBase):
    name: str = field(default="frequency_steps", init=False)
    description: str = field(default="Number of frequency steps for resonator spec", init=False)

    def _dummy_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.res_spec_steps")()

    def _dummy_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.res_spec_steps")(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.res_spec.steps")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.res_spec.steps")(value)

    def _opx_getter(self):
        q = nestedAttributeFromString(self.params, "active.qubit")()
        bandwidth = nestedAttributeFromString(self.params, f"{q}.readout.bandwidth")()
        step = bandwidth / 8
        span = bandwidth * nestedAttributeFromString(self.params, "scripts.qubit_tuneup.resonator_spec_range")()
        return int(abs(round(span / step)))

    def _opx_setter(self, value):
        points = int(value)
        return nestedAttributeFromString(self.params, "scripts.qubit_tuneup.resonator_spec_range")(points / 8)


@dataclass
class ReadoutFrequency(ProtocolParameterBase):
    name: str = field(default="readout_frequency", init=False)
    description: str = field(default="Frequency of the readout pulse", init=False)

    def _dummy_getter(self):
        return self.params.readout.f()

    def _dummy_setter(self, value):
        return self.params.readout.f(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.readout.freq")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.readout.freq")(value)

    def _opx_getter(self):
        q = nestedAttributeFromString(self.params, "active.qubit")()
        return (nestedAttributeFromString(self.params, f"{q}.readout.LO")()
                + nestedAttributeFromString(self.params, f"{q}.readout.IF")())

    def _opx_setter(self, value):
        q = nestedAttributeFromString(self.params, "active.qubit")()
        lo = nestedAttributeFromString(self.params, f"{q}.readout.LO")()
        return nestedAttributeFromString(self.params, f"{q}.readout.IF")(value - lo)

@dataclass
class ReadoutLength(ProtocolParameterBase):
    name: str = field(default="readout_length", init=False)
    description: str = field(default="Length of the readout pulse", init=False)

    def _dummy_getter(self):
        return self.params.readout.length()

    def _dummy_setter(self, value):
        return self.params.readout.length(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.readout.len")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.readout.len")(value)

    def _opx_getter(self):
        q = nestedAttributeFromString(self.params, "active.qubit")()
        readout_pulse = nestedAttributeFromString(self.params, "opx.readout_pulse")()
        return nestedAttributeFromString(self.params, f"{q}.readout.{readout_pulse}.len")()

    def _opx_setter(self, value):
        q = nestedAttributeFromString(self.params, "active.qubit")()
        readout_pulse = nestedAttributeFromString(self.params, "opx.readout_pulse")()
        return nestedAttributeFromString(self.params, f"{q}.readout.{readout_pulse}.len")(value)


@dataclass
class ReadoutGain(ProtocolParameterBase):
    name: str = field(default="readout_gain", init=False)
    description: str = field(default="Gain of the readout pulse", init=False)

    def _dummy_getter(self):
        return self.params.readout.gain()

    def _dummy_setter(self, value):
        return self.params.readout.gain(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.readout.gain")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.readout.gain")(value)

    def _opx_getter(self):
        q = nestedAttributeFromString(self.params, "active.qubit")()
        readout_pulse = nestedAttributeFromString(self.params, "opx.readout_pulse")()
        return nestedAttributeFromString(self.params, f"{q}.readout.{readout_pulse}.amp")()

    def _opx_setter(self, value):
        q = nestedAttributeFromString(self.params, "active.qubit")()
        readout_pulse = nestedAttributeFromString(self.params, "opx.readout_pulse")()
        return nestedAttributeFromString(self.params, f"{q}.readout.{readout_pulse}.amp")(value)

@dataclass
class StartReadoutFrequency(ProtocolParameterBase):
    name: str = field(default="initial_readout_frequency", init=False)
    description: str = field(default="Initial frequency of a readout frequency sweep", init=False)

    def _dummy_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.res_spec_start")()

    def _dummy_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.res_spec_start")(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.res_spec.start_f")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.res_spec.start_f")(value)

    def _opx_getter(self):
        return _opx_readout_window(self.params)[0]

    def _opx_setter(self, value):
        _, end = _opx_readout_window(self.params)
        _opx_set_readout_window(self.params, value, end)

@dataclass
class EndReadoutFrequency(ProtocolParameterBase):
    name: str = field(default="final_readout_frequency", init=False)
    description: str = field(default="Final frequency of a readout frequency sweep", init=False)

    def _dummy_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.res_spec_end")()

    def _dummy_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.res_spec_end")(value)
    
    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.res_spec.end_f")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.res_spec.end_f")(value)

    def _opx_getter(self):
        return _opx_readout_window(self.params)[1]

    def _opx_setter(self, value):
        start, _ = _opx_readout_window(self.params)
        _opx_set_readout_window(self.params, start, value)


@dataclass
class Detuning(ProtocolParameterBase):
    name: str = field(default="detuning", init=False)
    description: str = field(default="Dispersive shift (chi) - frequency difference of resonator with qubit in ground vs excited state", init=False)

    def _dummy_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.detuning")()

    def _dummy_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.detuning")(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.qubit.chi")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.qubit.chi")(value)
    
    def _opx_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.chi")()

    def _opx_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.chi")(value)


@dataclass
class Delay(ProtocolParameterBase):
    name: str = field(default="delay", init=False)
    description: str = field(default="Length of time that the machine waits between shots", init=False)

    def _dummy_getter(self):
        return self.params.msmt_params.delay()

    def _dummy_setter(self, value):
        return self.params.msmt_params.delay(value)

    def _qick_getter(self):
        return self.params.qick.final_delay()

    def _qick_setter(self, value):
        self.params.qick.final_delay(value)

    def _opx_getter(self):
        return self.params.opx.default_rep_delay()

    def _opx_setter(self, value):
        return self.params.opx.default_rep_delay(value)

@dataclass
class ResonatorSpecVsGainSteps(ProtocolParameterBase):
    name: str = field(default="resonator_spec_vs_gain_steps", init=False)
    description: str = field(default="Number of steps for resonator spectroscopy vs gain", init=False)

    def _dummy_getter(self):
        return self.params.readout.resonator_spec_vs_gain_steps()

    def _dummy_setter(self, value):
        return self.params.readout.resonator_spec_vs_gain_steps(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.res_spec_vs_gain.steps")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.res_spec_vs_gain.steps")(value)

    def _opx_getter(self):
        return int(nestedAttributeFromString(self.params, "scripts.qubit_tuneup.resonator_spec_vs_amp_na")())

    def _opx_setter(self, value):
        return nestedAttributeFromString(self.params, "scripts.qubit_tuneup.resonator_spec_vs_amp_na")(int(value))


@dataclass
class StartReadoutGain(ProtocolParameterBase):
    name: str = field(default="initial_readout_gain", init=False)
    description: str = field(default="Gain of the readout pulse", init=False)

    def _dummy_getter(self):
        return self.params.readout.start_g()

    def _dummy_setter(self, value):
        return self.params.readout.start_g(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.res_spec_vs_gain.start_g")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.res_spec_vs_gain.start_g")(value)

    def _opx_getter(self):
        return nestedAttributeFromString(self.params, "scripts.qubit_tuneup.resonator_spec_vs_amp_a0")()

    def _opx_setter(self, value):
        return nestedAttributeFromString(self.params, "scripts.qubit_tuneup.resonator_spec_vs_amp_a0")(value)


@dataclass
class EndReadoutGain(ProtocolParameterBase):
    name: str = field(default="final_readout_gain", init=False)
    description: str = field(default="Gain of the readout pulse", init=False)

    def _dummy_getter(self):
        return self.params.readout.end_g()

    def _dummy_setter(self, value):
        return self.params.readout.end_g(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.res_spec_vs_gain.end_g")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.res_spec_vs_gain.end_g")(value)

    def _opx_getter(self):
        return nestedAttributeFromString(self.params, "scripts.qubit_tuneup.resonator_spec_vs_amp_a1")()

    def _opx_setter(self, value):
        return nestedAttributeFromString(self.params, "scripts.qubit_tuneup.resonator_spec_vs_amp_a1")(value)


@dataclass
class SaturationSpecSteps(ProtocolParameterBase):
    name: str = field(default="sat_spec_steps", init=False)
    description: str = field(default="Number of steps for saturation spectroscopy", init=False)

    def _dummy_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.sat_spec_steps")()

    def _dummy_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.sat_spec_steps")(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.sat_spec.steps")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.sat_spec.steps")(value)

    def _opx_getter(self):
        span = nestedAttributeFromString(self.params, "scripts.qubit_tuneup.saturation_spec_range")()
        step = nestedAttributeFromString(self.params, "scripts.qubit_tuneup.saturation_spec_step")()
        return int(abs(round(span / step)))

    def _opx_setter(self, value):
        span = nestedAttributeFromString(self.params, "scripts.qubit_tuneup.saturation_spec_range")()
        return nestedAttributeFromString(self.params, "scripts.qubit_tuneup.saturation_spec_step")(span / int(value))


@dataclass
class StartSaturationSpecFrequency(ProtocolParameterBase):
    name: str = field(default="start_qubit_frequency", init=False)
    description: str = field(default="Initial frequency of a qubit frequency sweep", init=False)

    def _dummy_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.sat_spec_start")()

    def _dummy_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.sat_spec_start")(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.sat_spec.start_f")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.sat_spec.start_f")(value)

    def _opx_getter(self):
        return _opx_qubit_window(self.params, "scripts.qubit_tuneup.saturation_spec_range")[0]

    def _opx_setter(self, value):
        _, end = _opx_qubit_window(self.params, "scripts.qubit_tuneup.saturation_spec_range")
        _opx_set_qubit_window(self.params, "scripts.qubit_tuneup.saturation_spec_range", value, end)


@dataclass
class EndSaturationSpecFrequency(ProtocolParameterBase):
    name: str = field(default="end_qubit_frequency", init=False)
    description: str = field(default="Final frequency of a qubit frequency sweep", init=False)

    def _dummy_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.sat_spec_end")()

    def _dummy_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.sat_spec_end")(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.sat_spec.end_f")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.sat_spec.end_f")(value)

    def _opx_getter(self):
        return _opx_qubit_window(self.params, "scripts.qubit_tuneup.saturation_spec_range")[1]

    def _opx_setter(self, value):
        start, _ = _opx_qubit_window(self.params, "scripts.qubit_tuneup.saturation_spec_range")
        _opx_set_qubit_window(self.params, "scripts.qubit_tuneup.saturation_spec_range", start, value)


@dataclass
class SaturationSpecDriveGain(ProtocolParameterBase):
    name: str = field(default="sat_spec_drive_gain", init=False)
    description: str = field(default="Drive gain for the saturation spectroscopy pump pulse", init=False)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.pulses.const.gain")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.pulses.const.gain")(value)

    def _opx_getter(self):
        q = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{q}.pulses.long.amp")()

    def _opx_setter(self, value):
        q = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{q}.pulses.long.amp")(value)


@dataclass
class StartQubitGain(ProtocolParameterBase):
    name: str = field(default="start_qubit_gain", init=False)
    description: str = field(default="Initial gain of the qubit drive pulse for a gain sweep", init=False)

    def _dummy_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.power_rabi_start")()

    def _dummy_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.power_rabi_start")(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.power_rabi.start_g")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.power_rabi.start_g")(value)

    def _opx_getter(self):
        return -nestedAttributeFromString(self.params, "scripts.qubit_tuneup.rabi_range")()

    def _opx_setter(self, value):
        return nestedAttributeFromString(self.params, "scripts.qubit_tuneup.rabi_range")(abs(value))


@dataclass
class EndQubitGain(ProtocolParameterBase):
    name: str = field(default="end_qubit_gain", init=False)
    description: str = field(default="Final gain of the qubit drive pulse for a gain sweep", init=False)

    def _dummy_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.power_rabi_end")()

    def _dummy_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.power_rabi_end")(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.power_rabi.end_g")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.power_rabi.end_g")(value)

    def _opx_getter(self):
        return nestedAttributeFromString(self.params, "scripts.qubit_tuneup.rabi_range")()

    def _opx_setter(self, value):
        return nestedAttributeFromString(self.params, "scripts.qubit_tuneup.rabi_range")(abs(value))
    

@dataclass
class T1Steps(ProtocolParameterBase):
    name: str = field(default="t1_steps", init=False)
    description: str = field(default="Number of time steps for T1 measurement", init=False)

    def _dummy_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.t1_steps")()

    def _dummy_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.t1_steps")(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.t1.steps")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.t1.steps")(value)

    def _opx_getter(self):
        return int(5 * nestedAttributeFromString(self.params, "scripts.qubit_tuneup.pts_per_t1")() + 1)

    def _opx_setter(self, value):
        pts_per_t1 = max(1, int(round((int(value) - 1) / 5)))
        return nestedAttributeFromString(self.params, "scripts.qubit_tuneup.pts_per_t1")(pts_per_t1)


@dataclass
class T2ESteps(ProtocolParameterBase):
    name: str = field(default="t2e_steps", init=False)
    description: str = field(default="Number of time steps for T2 echo measurement", init=False)

    def _dummy_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.t2e_steps")()

    def _dummy_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.t2e_steps")(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.t2e.steps")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.t2e.steps")(value)

    def _opx_getter(self):
        q = nestedAttributeFromString(self.params, "active.qubit")()
        return _opx_t2_steps(self.params, f"{q}.T2E", "scripts.qubit_tuneup.oscillations_per_t2e")

    def _opx_setter(self, value):
        q = nestedAttributeFromString(self.params, "active.qubit")()
        t2 = nestedAttributeFromString(self.params, f"{q}.T2E")()
        oscillations = nestedAttributeFromString(self.params, "scripts.qubit_tuneup.oscillations_per_t2e")()
        period = max(1, int(t2 / oscillations))
        step = max(1, int((t2 * 3) // int(value)))
        nppp = max(1, int(round(period / step)))
        return nestedAttributeFromString(self.params, "scripts.qubit_tuneup.t2_nppp")(nppp)


@dataclass
class T2RSteps(ProtocolParameterBase):
    name: str = field(default="t2r_steps", init=False)
    description: str = field(default="Number of time steps for T2 Ramsey measurement", init=False)

    def _dummy_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.t2r_steps")()

    def _dummy_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.t2r_steps")(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.t2r.steps")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.t2r.steps")(value)

    def _opx_getter(self):
        q = nestedAttributeFromString(self.params, "active.qubit")()
        return _opx_t2_steps(self.params, f"{q}.T2R", "scripts.qubit_tuneup.oscillations_per_t2r")

    def _opx_setter(self, value):
        q = nestedAttributeFromString(self.params, "active.qubit")()
        t2 = nestedAttributeFromString(self.params, f"{q}.T2R")()
        oscillations = nestedAttributeFromString(self.params, "scripts.qubit_tuneup.oscillations_per_t2r")()
        period = max(1, int(t2 / oscillations))
        step = max(1, int((t2 * 3) // int(value)))
        nppp = max(1, int(round(period / step)))
        return nestedAttributeFromString(self.params, "scripts.qubit_tuneup.t2_nppp")(nppp)


@dataclass
class PiSpecSteps(ProtocolParameterBase):
    name: str = field(default="pi_spec_steps", init=False)
    description: str = field(default="Number of frequency steps for pi spectroscopy", init=False)

    def _dummy_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.pi_spec_steps")()

    def _dummy_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.pi_spec_steps")(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.pi_spec.steps")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.pi_spec.steps")(value)

    def _opx_getter(self):
        span = nestedAttributeFromString(self.params, "scripts.qubit_tuneup.qubit_spec_range")()
        return int(abs(round(span / _opx_pi_spec_step(self.params))))

    def _opx_setter(self, value):
        span = nestedAttributeFromString(self.params, "scripts.qubit_tuneup.qubit_spec_range")()
        target_step = max(1, int(span / int(value)))
        duration = int(round(1.5e9 / (5 * target_step)))
        return nestedAttributeFromString(self.params, "scripts.qubit_tuneup.pispec_pulselen")(duration)


@dataclass
class StartPiSpecFrequency(ProtocolParameterBase):
    name: str = field(default="start_pi_spec_frequency", init=False)
    description: str = field(default="Initial frequency of a pi spectroscopy frequency sweep", init=False)

    def _dummy_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.pi_spec_start")()

    def _dummy_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.pi_spec_start")(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.pi_spec.start_f")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.pi_spec.start_f")(value)

    def _opx_getter(self):
        return _opx_qubit_window(self.params, "scripts.qubit_tuneup.qubit_spec_range")[0]

    def _opx_setter(self, value):
        _, end = _opx_qubit_window(self.params, "scripts.qubit_tuneup.qubit_spec_range")
        _opx_set_qubit_window(self.params, "scripts.qubit_tuneup.qubit_spec_range", value, end)


@dataclass
class EndPiSpecFrequency(ProtocolParameterBase):
    name: str = field(default="end_pi_spec_frequency", init=False)
    description: str = field(default="Final frequency of a pi spectroscopy frequency sweep", init=False)

    def _dummy_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.pi_spec_end")()

    def _dummy_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.pi_spec_end")(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.pi_spec.end_f")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.pi_spec.end_f")(value)

    def _opx_getter(self):
        return _opx_qubit_window(self.params, "scripts.qubit_tuneup.qubit_spec_range")[1]

    def _opx_setter(self, value):
        start, _ = _opx_qubit_window(self.params, "scripts.qubit_tuneup.qubit_spec_range")
        _opx_set_qubit_window(self.params, "scripts.qubit_tuneup.qubit_spec_range", start, value)


@dataclass
class NumGainSteps(ProtocolParameterBase):
    name: str = field(default="num_gain_steps", init=False)
    description: str = field(default="Number of gain steps for a qubit drive gain sweep", init=False)

    def _dummy_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.gain_steps")()

    def _dummy_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.gain_steps")(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.power_rabi.steps")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.power_rabi.steps")(value)

    def _opx_getter(self):
        rng = nestedAttributeFromString(self.params, "scripts.qubit_tuneup.rabi_range")()
        step = nestedAttributeFromString(self.params, "scripts.qubit_tuneup.rabi_step")()
        return int(abs(round((2 * rng) / step)))

    def _opx_setter(self, value):
        rng = nestedAttributeFromString(self.params, "scripts.qubit_tuneup.rabi_range")()
        return nestedAttributeFromString(self.params, "scripts.qubit_tuneup.rabi_step")((2 * rng) / int(value))


@dataclass
class QubitGain(ProtocolParameterBase):
    name: str = field(default="qubit_gain", init=False)
    description: str = field(default="Gain of the qubit drive pulse", init=False)

    def _dummy_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.pulses.pi.amp")()

    def _dummy_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.pulses.pi.amp")(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.pulses.pi.gain")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.pulses.pi.gain")(value)

    def _opx_getter(self):
        q = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{q}.pulses.pi.amp")()

    def _opx_setter(self, value):
        q = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{q}.pulses.pi.amp")(value)


@dataclass
class QubitFrequency(ProtocolParameterBase):
    name: str = field(default="qubit_frequency", init=False)
    description: str = field(default="Intermediate frequency of the qubit", init=False)

    def _dummy_getter(self):
        return self.params.qubit.f()

    def _dummy_setter(self, value):
        return self.params.qubit.f(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.qubit.freq")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.qubit.freq")(value)

    def _opx_getter(self):
        q = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{q}.IF")()

    def _opx_setter(self, value):
        q = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{q}.IF")(value)


@dataclass
class T1(ProtocolParameterBase):
    name: str = field(default="T1", init=False)
    description: str = field(default="T1 relaxation time - characteristic time for qubit to decay from excited to ground state", init=False)

    def _dummy_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.T1")()

    def _dummy_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.T1")(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.qubit.T1")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.qubit.T1")(value)

    def _opx_getter(self):
        q = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{q}.T1")()

    def _opx_setter(self, value):
        q = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{q}.T1")(value)


@dataclass
class T2R(ProtocolParameterBase):
    name: str = field(default="T2R", init=False)
    description: str = field(default="T2 Ramsey time - dephasing time measured without echo pulses", init=False)

    def _dummy_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.T2R")()

    def _dummy_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.T2R")(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.qubit.T2R")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.qubit.T2R")(value)

    def _opx_getter(self):
        q = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{q}.T2R")()

    def _opx_setter(self, value):
        q = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{q}.T2R")(value)


@dataclass
class T2E(ProtocolParameterBase):
    name: str = field(default="T2E", init=False)
    description: str = field(default="T2 Echo time - dephasing time measured with echo pulses", init=False)

    def _dummy_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.T2E")()

    def _dummy_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.T2E")(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.qubit.T2E")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.qubit.T2E")(value)

    def _opx_getter(self):
        q = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{q}.T2E")()

    def _opx_setter(self, value):
        q = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{q}.T2E")(value)


@dataclass
class NEchos(ProtocolParameterBase):
    name: str = field(default="n_echos", init=False)
    description: str = field(default="Number of echo pulses in T2 measurement", init=False)

    def _dummy_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.n_echo")()

    def _dummy_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.n_echo")(value)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.t2e.n_echoes")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.scripts.t2e.n_echoes")(value)


@dataclass
class StartFlux(ProtocolParameterBase):
    name: str = field(default="start_flux", init=False)
    description: str = field(default="Initial flux value for flux sweep", init=False)

    def _dummy_getter(self):
        return self.params.flux.start()

    def _dummy_setter(self, value):
        return self.params.flux.start(value)


@dataclass
class EndFlux(ProtocolParameterBase):
    name: str = field(default="end_flux", init=False)
    description: str = field(default="Final flux value for flux sweep", init=False)

    def _dummy_getter(self):
        return self.params.flux.end()

    def _dummy_setter(self, value):
        return self.params.flux.end(value)


@dataclass
class FluxSteps(ProtocolParameterBase):
    name: str = field(default="flux_steps", init=False)
    description: str = field(default="Number of flux steps in flux sweep", init=False)

    def _dummy_getter(self):
        return self.params.flux.steps()

    def _dummy_setter(self, value):
        return self.params.flux.steps(value)


@dataclass
class ZeroFluxCurrent(ProtocolParameterBase):
    name: str = field(default="zero_flux_current", init=False)
    description: str = field(default="Current value corresponding to zero flux", init=False)

    def _dummy_getter(self):
        return self.params.flux.zero_current()

    def _dummy_setter(self, value):
        return self.params.flux.zero_current(value)


@dataclass
class GainPulseDuration(ProtocolParameterBase):
    name: str = field(default="rabi_pulse_duration", init=False)
    description: str = field(default="Longest duration of applying the Rabi pulse", init=False)
        
    def _dummy_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.pulses.duration")()

    def _dummy_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.pulses.duration")(value)


@dataclass
class ECParam(ProtocolParameterBase):
    name: str = field(default="EC", init=False)
    description: str = field(default="Charging energy for fluxonium (GHz)", init=False)

    def _dummy_getter(self):
        return self.params.qubit.EC()

    def _dummy_setter(self, value):
        return self.params.qubit.EC(value)


@dataclass
class ELParam(ProtocolParameterBase):
    name: str = field(default="EL", init=False)
    description: str = field(default="Inductive energy for fluxonium (GHz)", init=False)

    def _dummy_getter(self):
        return self.params.qubit.EL()

    def _dummy_setter(self, value):
        return self.params.qubit.EL(value)


@dataclass
class EJParam(ProtocolParameterBase):
    name: str = field(default="EJ", init=False)
    description: str = field(default="Josephson energy for fluxonium (GHz)", init=False)

    def _dummy_getter(self):
        return self.params.qubit.EJ()

    def _dummy_setter(self, value):
        return self.params.qubit.EJ(value)


@dataclass
class CouplingG(ProtocolParameterBase):
    name: str = field(default="g", init=False)
    description: str = field(default="Coupling strength between qubit and resonator (GHz)", init=False)

    def _dummy_getter(self):
        return self.params.coupling.g()

    def _dummy_setter(self, value):
        return self.params.coupling.g(value)


@dataclass
class ResonatorFr(ProtocolParameterBase):
    name: str = field(default="fr", init=False)
    description: str = field(default="Bare resonator frequency (GHz)", init=False)

    def _dummy_getter(self):
        return self.params.readout.fr()

    def _dummy_setter(self, value):
        return self.params.readout.fr(value)

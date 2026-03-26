"""
TODO LIST:

* Detuning word is now being used for both the artificial detuning and chi
* ROCalibration should store the centers of both g and e states in the parameter manager and have ProtocolParameters for them.

"""
from dataclasses import dataclass, field
from labcore.protocols.base import ProtocolParameterBase
from instrumentserver.helpers import nestedAttributeFromString


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


@dataclass
class Delay(ProtocolParameterBase):
    name: str = field(default="delay", init=False)
    description: str = field(default="Length of time that the machine waits between shots", init=False)

    def _dummy_getter(self):
        return self.params.msmt_params.delay()

    def _dummy_setter(self, value):
        return self.params.params.msmt_params.delay(value)

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

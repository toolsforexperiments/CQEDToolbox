
from qick.asm_v2 import QickSweep1D

from instrumentserver.client import Client
from instrumentserver.helpers import nestedAttributeFromString

from cqedtoolbox.instruments.qick.config import QBoardConfig


def getp(params, name: str, default=None, raise_if_missing=False):
    if params.parameters is None:
        print("No parameter manager defined. cannot get/set params!")
        return None

    try:
        p = nestedAttributeFromString(params, name)
        return p()
    except AttributeError:
        if raise_if_missing:
            raise
        else:
            return default


class QickConfig(QBoardConfig):
    def config_(self):
        params = self.params

        active_qubit = getp(self.params, "active.qubit")

        cfg = {
            # General parameters
            'reps': params.qick.default_reps(),
            'trig_time': params.qick.trig_time(),
            'final_delay': params.qick.final_delay(),
            'soft_avgs': params.qick.soft_avgs(),

            # Readout resonator
            'ro_adc_ch': getp(params, f"{active_qubit}.readout.adc_ch"),
            'ro_dac_ch': getp(params, f"{active_qubit}.readout.dac_ch"),
            'ro_nqz': getp(params, f"{active_qubit}.readout.nqz"),
            'ro_freq': getp(params, f"{active_qubit}.readout.freq"),
            'ro_len': getp(params, f"{active_qubit}.readout.len"),
            'ro_phase': getp(params, f"{active_qubit}.readout.phase"),
            'ro_gain': getp(params, f"{active_qubit}.readout.gain"),

            # Qubit
            'q_dac_ch': getp(params, f"{active_qubit}.qubit.dac_ch"),
            'q_nqz': getp(params, f"{active_qubit}.qubit.nqz"),
            'q_freq': getp(params, f"{active_qubit}.qubit.freq"),
            'q_detuning': getp(params, f"{active_qubit}.qubit.detuning"),

            # Qubit pulses

            # Constant long pulse
            'q_const_len': getp(params, f"{active_qubit}.pulses.const.len"),
            'q_const_gain': getp(params, f"{active_qubit}.pulses.const.gain"),
            'q_const_phase': getp(params, f"{active_qubit}.pulses.const.phase"),

            # Pi pulse
            'q_pi_sigma': getp(params, f"{active_qubit}.pulses.pi.sigma"),
            'q_pi_n_sigma': getp(params, f"{active_qubit}.pulses.pi.n_sigma"),
            'q_pi_gain': getp(params, f"{active_qubit}.pulses.pi.gain"),
            'q_pi_phase': getp(params, f"{active_qubit}.pulses.pi.phase"),

            # Measurements

            # Resonator spec
            'ro_freq_loop_var': QickSweep1D("ro_freqs_loop", getp(params, f"{active_qubit}.scripts.res_spec.start_f"), getp(params, f"{active_qubit}.scripts.res_spec.end_f")),
            'ro_freq_steps': getp(params, f"{active_qubit}.scripts.res_spec.steps"),

            # Resonator spec vs gain
            'ro_gain_loop_var': QickSweep1D("ro_gains_loop", getp(params, f"{active_qubit}.scripts.res_spec_vs_gain.start_g"), getp(params, f"{active_qubit}.scripts.res_spec_vs_gain.end_g")),
            'ro_gain_steps': getp(params, f"{active_qubit}.scripts.res_spec_vs_gain.steps"),

            # Saturation spec
            'sat_spec_freq_loop_var': QickSweep1D("sat_spec_freqs_loop", getp(params, f"{active_qubit}.scripts.sat_spec.start_f"), getp(params, f"{active_qubit}.scripts.sat_spec.end_f")),
            'sat_spec_freq_steps': getp(params, f"{active_qubit}.scripts.sat_spec.steps"),

            # Power rabi
            'q_gain_loop_var': QickSweep1D("q_gains_loop", getp(params, f"{active_qubit}.scripts.power_rabi.start_g"), getp(params, f"{active_qubit}.scripts.power_rabi.end_g")),
            'q_gain_steps': getp(params, f"{active_qubit}.scripts.power_rabi.steps"),

            # Pi spec
            'pi_spec_freq_loop_var': QickSweep1D("pi_spec_freqs_loop", getp(params, f"{active_qubit}.scripts.pi_spec.start_f"), getp(params, f"{active_qubit}.scripts.pi_spec.end_f")),
            'pi_spec_freq_steps': getp(params, f"{active_qubit}.scripts.pi_spec.steps"),

            # T1
            't1_wait_time_loop_var': QickSweep1D("t1_wait_time_loop", 0, getp(params, f"{active_qubit}.scripts.t1.n_T1") * getp(params, f"{active_qubit}.qubit.T1")),
            't1_steps': getp(params, f"{active_qubit}.scripts.t1.steps"),

            # T2 Ramsey
            't2r_wait_time_loop_var': QickSweep1D("t2r_wait_time_loop", 0, getp(params, f"{active_qubit}.scripts.t2r.n_T2R") * getp(params, f"{active_qubit}.qubit.T2R")),
            't2r_steps': getp(params, f"{active_qubit}.scripts.t2r.steps"),

            # T2 Echo
            't2e_wait_time_loop_var': QickSweep1D("t2e_wait_time_loop", 0, getp(params, f"{active_qubit}.scripts.t2e.n_T2E") * getp(params, f"{active_qubit}.qubit.T2E")),
            't2e_steps': getp(params, f"{active_qubit}.scripts.t2e.steps"),
            'n_echoes': getp(params, f"{active_qubit}.scripts.t2e.n_echoes"),

            # Single shot
            'steps': getp(params, f"{active_qubit}.scripts.single_shot.steps"),
        }

        return cfg
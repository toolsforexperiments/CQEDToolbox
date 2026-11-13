import logging

from qick.asm_v2 import AveragerProgramV2
from cqedtoolbox.instruments.qick.qick_sweep_v2 import QickBoardSweep, ComplexQICKData, PulseVariable, TimeVariable
from labcore.measurement import independent

'''
This format assumes some standard nomenclature that one must follow while using the QICK program.
You can refer to 'https://github.com/quantumcircuitsatuiuc/measurement/tree/main/doc/examples/qick_v2_example/my_experiment_setup.py' 
for a demo on this.
For an example file using these functions, refer to 
'https://github.com/quantumcircuitsatuiuc/measurement/tree/main/doc/examples/qick_v2_example/Qubit_tuneup.ipynb'.
'''

logger = logging.getLogger(__name__)

PI_PULSE_SHAPES = ("gaussian", "flattop")


def get_pi_pulse_shape(cfg):
    """Return the configured pi-pulse shape, 'gaussian' or 'flattop'.

    Read from ``cfg['q_pi_pulse_shape']`` (``{qubit}.pulses.pi.pulse_shape`` in the
    parameter manager). Falls back to 'flattop' with a warning when unset.
    """
    shape = cfg.get("q_pi_pulse_shape")
    if shape is None:
        logger.warning("pulses.pi.pulse_shape is not set; defaulting to 'flattop'.")
        return "flattop"
    shape = str(shape).strip().lower()
    if shape not in PI_PULSE_SHAPES:
        raise ValueError(
            f"Unknown pi pulse shape {shape!r}; expected one of {PI_PULSE_SHAPES}."
        )
    return shape


def add_pi_envelope(prog, cfg, ch, prefix, label, key_prefix="q_pi", require_gain=True):
    """Register the pi-pulse envelope on ``ch`` according to the configured shape.

    The shape always comes from ``cfg['q_pi_pulse_shape']``. Pulse settings are read
    from cfg keys built from ``key_prefix``:

    * gaussian: ``{key_prefix}_gaussian_sigma``, ``{key_prefix}_gaussian_nsigma``,
      ``{key_prefix}_gaussian_amp``
    * flattop: ``{key_prefix}_flattop_ramp``, ``{key_prefix}_flattop_length``,
      ``{key_prefix}_flattop_gain``

    The gain key is only required when ``require_gain`` is True (amplitude Rabi
    sweeps the gain instead). Which parameter-manager entries feed these keys is
    decided in QickConfig.

    :param prog: The AveragerProgramV2 being initialized.
    :param cfg: The program cfg dict.
    :param ch: Qubit DAC channel.
    :param prefix: Prefix for the envelope name registered on the program.
    :param label: Human-readable program name used in log and error messages.
    :param key_prefix: Prefix of the cfg keys holding the pulse settings.
    :param require_gain: Whether to require and return the calibrated gain.
    :return: ``(pulse_kwargs, gain)``. ``pulse_kwargs`` holds the shape-specific
        ``add_pulse`` arguments (``style``, ``envelope`` and, for flat-top, ``length``);
        ``gain`` is None when ``require_gain`` is False.
    """
    shape = get_pi_pulse_shape(cfg)

    if shape == "gaussian":
        keys = [f"{key_prefix}_gaussian_sigma", f"{key_prefix}_gaussian_nsigma"]
        gain_key = f"{key_prefix}_gaussian_amp"
    else:
        keys = [f"{key_prefix}_flattop_ramp", f"{key_prefix}_flattop_length"]
        gain_key = f"{key_prefix}_flattop_gain"
    if require_gain:
        keys.append(gain_key)

    missing = [k for k in keys if cfg.get(k) is None]
    if missing:
        raise ValueError(f"{label} ({shape} pi pulse) requires {', '.join(missing)}.")

    gain = cfg[gain_key] if require_gain else None
    gain_str = f"gain={gain}" if require_gain else "gain=swept"

    if shape == "gaussian":
        sigma = cfg[f"{key_prefix}_gaussian_sigma"]
        nsigma = cfg[f"{key_prefix}_gaussian_nsigma"]
        envelope = f"{prefix}_gauss"
        prog.add_gauss(ch=ch, name=envelope, sigma=sigma,
                       length=nsigma * sigma, even_length=True)
        logger.info(
            "%s: Gaussian pi pulse (sigma=%s us, nsigma=%s, total=%s us, %s).",
            label, sigma, nsigma, nsigma * sigma, gain_str,
        )
        return dict(style="arb", envelope=envelope), gain

    ramp = cfg[f"{key_prefix}_flattop_ramp"]
    length = cfg[f"{key_prefix}_flattop_length"]
    envelope = f"{prefix}_flattop_cosine"
    # QICK splits the cosine envelope into equal rising and falling ramps.
    prog.add_cosine(ch=ch, name=envelope, length=2 * ramp, even_length=True)
    logger.info(
        "%s: flat-top pi pulse (ramp=%s us per edge, flat=%s us, total=%s us, %s).",
        label, ramp, length, length + 2 * ramp, gain_str,
    )
    return dict(style="flat_top", envelope=envelope, length=length), gain


def log_time_rabi_drive(cfg, label, ramp, gain):
    """Log the flat-top drive used by the time Rabi programs."""
    logger.info(
        "%s: flat-top drive (ramp=%s us per edge, flat length swept, gain=%s).",
        label, ramp, gain,
    )
    shape = cfg.get("q_pi_pulse_shape")
    if shape is not None and str(shape).strip().lower() == "gaussian":
        logger.info(
            "%s: pulses.pi.pulse_shape='gaussian' is ignored; time Rabi always "
            "uses a flat-top drive.", label,
        )

class OffsetProgram(AveragerProgramV2):
    """
    To calibrate the adc trigger offset caused by electrical delays.
    This function is not supported by the QICK sweep wrapper.
    """
    def _initialize(self, cfg):

        if cfg["reps"] != 1:
            raise ValueError("reps should be 1!")

        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']

        self.declare_gen(ch=ro_dac_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_adc_ch, length=cfg['ro_len'])

        self.add_readoutconfig(ch=ro_adc_ch, name="readout", freq=cfg['ro_freq'], gen_ch=ro_dac_ch)

        self.add_pulse(ch=ro_dac_ch, name="probe_pulse", ro_ch=ro_adc_ch,
                       style="const", freq=cfg['ro_freq'], length=cfg['ro_len'],
                       phase=cfg['ro_phase'], gain=cfg['ro_gain'])

    def _body(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']

        self.send_readoutconfig(ch=ro_adc_ch, name="readout", t=0)

        self.pulse(ch=ro_dac_ch, name="probe_pulse", t=0)
        self.trigger(ros=[ro_adc_ch], pins=[0], t=cfg['trig_time'])

        
@QickBoardSweep(
    PulseVariable('freq', pulse_parameter="probe_pulse", sweep_parameter="freq", unit="MHz"),
    ComplexQICKData('signal', depends_on=['freq'])
)
class FreqSweepProgram(AveragerProgramV2):
    '''
    Performs single tone spectroscopy on the resonator.
    '''
    def _initialize(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_gen_ch = cfg['ro_dac_ch']
        
        self.declare_gen(ch=ro_gen_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_adc_ch, length=cfg['ro_len'])

        self.add_loop("ro_freqs_loop", self.cfg["ro_freq_steps"])

        self.add_readoutconfig(ch=ro_adc_ch, name="readout", freq=cfg['ro_freq_loop_var'], gen_ch=ro_gen_ch)  # Sweep variable

        self.add_pulse(ch=ro_gen_ch, name="probe_pulse", ro_ch=ro_adc_ch,
                       style="const",
                       freq=cfg['ro_freq_loop_var'],  # Sweep variable
                       length=cfg['ro_len'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ro_gain'],
                       )
        
    def _body(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_gen_ch = cfg['ro_dac_ch']

        self.send_readoutconfig(ch=ro_adc_ch, name="readout", t=0)

        self.pulse(ch=ro_gen_ch, name="probe_pulse", t=0)
        self.trigger(ros=[ro_adc_ch], pins=[0], t=cfg['trig_time'])


@QickBoardSweep(
    PulseVariable('freq', pulse_parameter="probe_pulse", sweep_parameter="freq", unit="MHz"),
    PulseVariable('gain', pulse_parameter="probe_pulse", sweep_parameter="gain"),
    ComplexQICKData('signal', depends_on=['freq', 'gain'])
)
class FreqGainSweepProgram(AveragerProgramV2):
    '''
    Performs a sweep over gain vs single tone spectroscopy on the resonator.
    '''

    def _initialize(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']
        
        self.declare_gen(ch=ro_dac_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_adc_ch, length=cfg['ro_len'])

        self.add_loop("ro_gains_loop", self.cfg["ro_gain_steps"])
        self.add_loop("ro_freqs_loop", self.cfg["ro_freq_steps"])

        self.add_readoutconfig(ch=ro_adc_ch, name="readout", freq=cfg['ro_freq_loop_var'], gen_ch=ro_dac_ch)  # Sweep variable

        self.add_pulse(ch=ro_dac_ch, name="probe_pulse", ro_ch=ro_adc_ch,
                       style="const",
                       freq=cfg['ro_freq_loop_var'],  # Sweep variable
                       length=cfg['ro_len'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ro_gain_loop_var'],  # Sweep variable
                       )
        
    def _body(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_gen_ch = cfg['ro_dac_ch']

        self.send_readoutconfig(ch=ro_adc_ch, name="readout", t=0)

        self.pulse(ch=ro_gen_ch, name="probe_pulse", t=0)
        self.trigger(ros=[ro_adc_ch], pins=[0], t=cfg['trig_time'])


@QickBoardSweep(
    PulseVariable('freq', pulse_parameter="pump_pulse", sweep_parameter="freq", unit="MHz"),
    ComplexQICKData('signal', depends_on=['freq'])
)
class PulseProbeSpectroscopy(AveragerProgramV2):
    '''
    Performs two-tone spectroscopy where the probe tone is probing the readout resonator
    while the pump tone excites the qubit.
    '''

    def _initialize(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']
        q_dac_ch = cfg['q_dac_ch']

        self.declare_gen(ch=ro_dac_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_adc_ch, length=cfg['ro_len'])

        self.declare_gen(ch=q_dac_ch, nqz=cfg['q_nqz'])

        # Pump pulse definition
        self.add_loop("sat_spec_freqs_loop", self.cfg["sat_spec_freq_steps"])
        self.add_pulse(ch=q_dac_ch, name='pump_pulse',
                       style='const',
                       freq=cfg['sat_spec_freq_loop_var'],  # Sweep variable
                       length=cfg['q_const_len'],
                       phase=cfg['q_const_phase'],
                       gain=cfg['q_const_gain'])

        # Probe pulse definition
        self.add_readoutconfig(ch=ro_adc_ch, name="readout", freq=cfg['ro_freq'], gen_ch=ro_dac_ch)

        self.add_pulse(ch=ro_dac_ch, name="read_pulse", ro_ch=ro_adc_ch,
                       style="const",
                       freq=cfg['ro_freq'],
                       length=cfg['ro_len'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ro_gain'],
                       )

    def _body(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']
        q_dac_ch = cfg['q_dac_ch']

        # if you delay the config by too long, you can see the readout get reconfigured in the middle of your pulse
        self.send_readoutconfig(ch=ro_adc_ch, name="readout", t=0)

        self.pulse(ch=q_dac_ch, name='pump_pulse', t=0)
        self.delay_auto(t=0.05, gens=True, ros=False, tag='wait_time')
        self.pulse(ch=ro_dac_ch, name="read_pulse", t=0)
        self.trigger(ros=[ro_adc_ch], pins=[0], t=cfg['trig_time'])


@QickBoardSweep(
    PulseVariable("time", pulse_parameter="time_rabi_pulse", sweep_parameter="length", unit="us"),
    ComplexQICKData("signal", depends_on=["time"]),
)
class TimeRabiProgram(AveragerProgramV2):
    '''
    Performs time Rabi on a qubit using a variable-length drive pulse.
    '''
    def _initialize(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']
        q_dac_ch = cfg['q_dac_ch']

        self.declare_gen(ch=ro_dac_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_adc_ch, length=cfg['ro_len'])

        self.declare_gen(ch=q_dac_ch, nqz=cfg['q_nqz'])

        
        # Pump pulse definition
        self.add_loop("timerabi_q_pulse_lens_loop", self.cfg["timerabi_q_pulse_len_steps"])
        flattop_ramp = cfg["timerabi_flattop_ramp"]
        if flattop_ramp is None:
            raise ValueError("Time Rabi requires timerabi_flattop_ramp.")
        log_time_rabi_drive(cfg, "Time Rabi", flattop_ramp, cfg['timerabi_q_gain'])
        self.add_cosine(ch=q_dac_ch, name="timerabi_flattop_cosine",
                        length=2 * flattop_ramp, even_length=True)
        self.add_pulse(ch=q_dac_ch, name="time_rabi_pulse",
                       style="flat_top",
                       envelope="timerabi_flattop_cosine",
                       length=cfg["timerabi_q_pulse_len_loop_var"], # sweep variable
                       freq=cfg['q_freq'],
                       phase=cfg['q_pi_phase'],
                       gain=cfg['timerabi_q_gain'],
                       )
        
        # Probe pulse definition
        self.add_readoutconfig(ch=ro_adc_ch, name="readout", freq=cfg['ro_freq'], gen_ch=ro_dac_ch)
        self.add_pulse(ch=ro_dac_ch, name="read_pulse", ro_ch=ro_adc_ch,
                       style="const",
                       freq=cfg['ro_freq'],
                       length=cfg['ro_len'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ro_gain'],
                       )
        
    def _body(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']
        q_dac_ch = cfg['q_dac_ch']

        self.send_readoutconfig(ch=ro_adc_ch, name="readout", t=0)

        self.pulse(ch=q_dac_ch, name='time_rabi_pulse', t=0)
        self.delay_auto(t=0.0, gens=True, ros=False)
        self.pulse(ch=ro_dac_ch, name="read_pulse", t=0)
        self.trigger(ros=[ro_adc_ch], pins=[0], t=cfg['trig_time'])


@QickBoardSweep(
    PulseVariable("time", pulse_parameter="time_rabi_pulse", sweep_parameter="length", unit="us"),
    PulseVariable("freq", pulse_parameter="time_rabi_pulse", sweep_parameter="freq", unit="MHz"),
    ComplexQICKData("signal", depends_on=["time", "freq"]),
)
class TimeRabiVsFreqProgram(AveragerProgramV2):
    """
    Performs time Rabi on a qubit with variable length and frequency.

    """

    def _initialize(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']
        q_dac_ch = cfg['q_dac_ch']

        self.declare_gen(ch=ro_dac_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_adc_ch, length=cfg['ro_len'])

        self.declare_gen(ch=q_dac_ch, nqz=cfg['q_nqz'])

        self.add_loop("timerabi_vs_freq_q_freq_loop", self.cfg["timerabi_vs_freq_q_freq_steps"])
        self.add_loop("timerabi_vs_freq_q_pulse_lens_loop", self.cfg["timerabi_vs_freq_q_pulse_len_steps"])

        flattop_ramp = cfg["timerabi_vs_freq_flattop_ramp"]
        if flattop_ramp is None:
            raise ValueError(
                "Time Rabi vs. frequency requires timerabi_vs_freq_flattop_ramp."
            )
        log_time_rabi_drive(cfg, "Time Rabi vs. frequency", flattop_ramp,
                            cfg['timerabi_vs_freq_q_gain'])
        self.add_cosine(ch=q_dac_ch, name="timerabi_vs_freq_flattop_cosine",
                        length=2 * flattop_ramp, even_length=True)
        self.add_pulse(ch=q_dac_ch, name="time_rabi_pulse",
                       style="flat_top",
                       envelope="timerabi_vs_freq_flattop_cosine",
                       length=cfg["timerabi_vs_freq_q_pulse_len_loop_var"], # sweep variable
                       freq=cfg["timerabi_vs_freq_q_freq_loop_var"], # sweep variable
                       phase=cfg['q_pi_phase'],
                       gain=cfg['timerabi_vs_freq_q_gain'],
                       )
        
        # Probe pulse definition
        self.add_readoutconfig(ch=ro_adc_ch, name="readout", freq=cfg['ro_freq'], gen_ch=ro_dac_ch)
        self.add_pulse(ch=ro_dac_ch, name="read_pulse", ro_ch=ro_adc_ch,
                       style="const",
                       freq=cfg['ro_freq'],
                       length=cfg['ro_len'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ro_gain'],
                       )
        
    def _body(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']
        q_dac_ch = cfg['q_dac_ch']

        self.send_readoutconfig(ch=ro_adc_ch, name="readout", t=0)

        self.pulse(ch=q_dac_ch, name='time_rabi_pulse', t=0)
        self.delay_auto(t=0.0, gens=True, ros=False)
        self.pulse(ch=ro_dac_ch, name="read_pulse", t=0)
        self.trigger(ros=[ro_adc_ch], pins=[0], t=cfg['trig_time'])


@QickBoardSweep(
    PulseVariable('freq', pulse_parameter="pump_pulse", sweep_parameter="freq", unit="MHz"),
    ComplexQICKData('signal', depends_on=['freq'])
)
class TwoToneSpectroscopy(AveragerProgramV2):
    '''
    Performs two-tone spectroscopy where the probe tone is probing the readout resonator
    while the pump tone excites the qubit.
    '''

    def _initialize(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']
        q_dac_ch = cfg['q_dac_ch']

        self.declare_gen(ch=ro_dac_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_adc_ch, length=cfg['two_tone_spec_ro_len'])

        self.declare_gen(ch=q_dac_ch, nqz=cfg['q_nqz'])



        # ====== Two tone pulses =======
        # Pump pulse definition
        self.add_loop("two_tone_spec_q_freq_loop", self.cfg["two_tone_spec_q_freq_steps"])
        self.add_pulse(ch=q_dac_ch, name='pump_pulse',
                       style='const',
                       freq=cfg['two_tone_spec_q_freq_var'],  # Sweep variable
                       length=cfg['two_tone_q_len'],
                       phase=cfg['two_tone_q_const_phase'],
                       gain=cfg['two_tone_q_const_gain'])

        # Probe pulse definition
        self.add_readoutconfig(ch=ro_adc_ch, name="readout", freq=cfg['ro_freq'], gen_ch=ro_dac_ch)

        self.add_pulse(ch=ro_dac_ch, name="read_pulse", ro_ch=ro_adc_ch,
                       style="const",
                       freq=cfg['ro_freq'],
                       length=cfg['two_tone_spec_ro_len'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ro_gain'],
                       )

    def _body(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']
        q_dac_ch = cfg['q_dac_ch']

        # if you delay the config by too long, you can see the readout get reconfigured in the middle of your pulse
        self.send_readoutconfig(ch=ro_adc_ch, name="readout", t=0)

        self.pulse(ch=q_dac_ch, name='pump_pulse', t=0.05)
        # self.delay_auto(t=0.05, gens=True, ros=False, tag='wait_time')
        self.pulse(ch=ro_dac_ch, name="read_pulse", t=0.05)
        self.trigger(ros=[ro_adc_ch], pins=[0], t=cfg['trig_time'])


@QickBoardSweep(
    PulseVariable("gain", pulse_parameter='pi_pulse', sweep_parameter="gain"),
    ComplexQICKData("signal", depends_on=["gain"])
)
class AmplitudeRabiProgram(AveragerProgramV2):
    '''
    Calibrates the power of a pi pulse by sweeping its gain.
    Note that the current version of the QICK firmware supports 16-bit signed gain.
    '''
    def _initialize(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']
        q_dac_ch = cfg['q_dac_ch']

        self.declare_gen(ch=ro_dac_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_adc_ch, length=cfg['ro_len'])

        self.declare_gen(ch=q_dac_ch, nqz=cfg['q_nqz'])

        
        # Pump pulse definition
        self.add_loop("q_gains_loop", self.cfg["q_gain_steps"])
        # Pulse timing comes from the amprabi_* cfg keys; QickConfig decides
        # whether those point at the pi pulse or at power-Rabi-specific settings.
        pi_kwargs, _ = add_pi_envelope(
            self, cfg, q_dac_ch, prefix="amprabi", label="Amplitude Rabi",
            key_prefix="amprabi", require_gain=False,
        )
        self.add_pulse(ch=q_dac_ch, name="pi_pulse",
                       **pi_kwargs,
                       freq=cfg['q_freq'],
                       phase=cfg['q_pi_phase'],
                       gain=cfg['q_gain_loop_var'],  # Sweep variable
                       )
        
        # Probe pulse definition
        self.add_readoutconfig(ch=ro_adc_ch, name="readout", freq=cfg['ro_freq'], gen_ch=ro_dac_ch)
        self.add_pulse(ch=ro_dac_ch, name="read_pulse", ro_ch=ro_adc_ch,
                       style="const",
                       freq=cfg['ro_freq'],
                       length=cfg['ro_len'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ro_gain'],
                       )
        
    def _body(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']
        q_dac_ch = cfg['q_dac_ch']

        self.send_readoutconfig(ch=ro_adc_ch, name="readout", t=0)

        self.pulse(ch=q_dac_ch, name='pi_pulse', t=0)
        self.delay_auto(t=0.0, gens=True, ros=False, tag='wait_time')
        self.pulse(ch=ro_dac_ch, name="read_pulse", t=0)
        self.trigger(ros=[ro_adc_ch], pins=[0], t=cfg['trig_time'])


@QickBoardSweep(
    PulseVariable("freq", pulse_parameter='pi_pulse', sweep_parameter="freq", unit="MHz"),
    ComplexQICKData("signal", depends_on=["freq"])
)
class PiSpecProgram(AveragerProgramV2):
    '''
    Runs pi-pulse spectroscopy to determine the qubit frequency.
    '''
    def _initialize(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']
        q_dac_ch = cfg['q_dac_ch']

        self.declare_gen(ch=q_dac_ch, nqz=cfg['q_nqz'])
        self.declare_gen(ch=ro_dac_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_adc_ch, length=cfg['ro_len'])

        # Pump pulse definition
        self.add_loop("pi_spec_freqs_loop", self.cfg["pi_spec_freq_steps"])
        pi_kwargs, pi_gain = add_pi_envelope(
            self, cfg, q_dac_ch, prefix="pi_spec", label="Pi spectroscopy")
        self.add_pulse(ch=q_dac_ch, name="pi_pulse",
                       **pi_kwargs,
                       freq=cfg['pi_spec_freq_loop_var'],  # Sweep variable
                       phase=cfg['q_pi_phase'],
                       gain=pi_gain,
                      )

        # Probe pulse definition
        self.add_readoutconfig(ch=ro_adc_ch, name="readout", freq=cfg['ro_freq'], gen_ch=ro_dac_ch)
        self.add_pulse(ch=ro_dac_ch, name="read_pulse", ro_ch=ro_adc_ch,
                       style="const",
                       freq=cfg['ro_freq'],
                       length=cfg['ro_len'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ro_gain'],
                      )

    def _body(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']
        q_dac_ch = cfg['q_dac_ch']

        self.send_readoutconfig(ch=ro_adc_ch, name="readout", t=0)

        self.pulse(ch=q_dac_ch, name='pi_pulse', t=0)
        self.delay_auto(t=0.0, gens=True, ros=False, tag='wait_time')
        self.pulse(ch=ro_dac_ch, name="read_pulse", t=0)
        self.trigger(ros=[ro_adc_ch], pins=[0], t=cfg['trig_time'])


@QickBoardSweep(
    PulseVariable("freq", pulse_parameter="probe_pulse", sweep_parameter="freq", unit="MHz"),
    ComplexQICKData("signal", depends_on=["freq"])
)
class ResProbeProgram(AveragerProgramV2):
    '''
    Performs single tone spectroscopy on RO resonator after exciting g->e transition of qubit.
    '''

    def _initialize(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']
        q_dac_ch = cfg['q_dac_ch']

        self.declare_gen(ch=q_dac_ch, nqz=cfg['q_nqz'])
        self.declare_gen(ch=ro_dac_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_adc_ch, length=cfg['ro_len'])

        self.add_loop("ro_freqs_loop", self.cfg["ro_freq_steps"])

        pi_kwargs, pi_gain = add_pi_envelope(
            self, cfg, q_dac_ch, prefix="resprobe", label="Resonator probe")
        self.add_pulse(ch=q_dac_ch, name="pi_pulse", **pi_kwargs,
                       freq=cfg['q_freq'], phase=cfg['q_pi_phase'], gain=pi_gain)
        self.add_readoutconfig(ch=ro_adc_ch, name="readout", freq=cfg['ro_freq_loop_var'], gen_ch=ro_dac_ch)

        self.add_pulse(ch=ro_dac_ch, name="probe_pulse", ro_ch=ro_adc_ch,
                       style="const",
                       freq=cfg['ro_freq_loop_var'],  # Sweep variable
                       length=cfg['ro_len'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ro_gain'],
                       )

    def _body(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']
        q_dac_ch = cfg['q_dac_ch']

        self.send_readoutconfig(ch=ro_adc_ch, name="readout", t=0)

        self.pulse(ch=q_dac_ch, name='pi_pulse', t=0)
        self.delay_auto(t=0.00, gens=True, ros=False)
        self.pulse(ch=ro_dac_ch, name="probe_pulse", t=0)
        self.trigger(ros=[ro_adc_ch], pins=[0], t=cfg['trig_time'])


@QickBoardSweep(
    TimeVariable("t", time_parameter='wait_time', unit="us"),
    ComplexQICKData("signal", depends_on=["t"])
)
class T1Program(AveragerProgramV2):
    '''
    Measures the T1 time of the qubit.
    '''
    def _initialize(self, cfg):
        if cfg["n_echoes"] != 0:
            raise ValueError("n_echoes should be 0 for T1 program!")
        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']
        q_dac_ch = cfg['q_dac_ch']

        self.declare_gen(ch=q_dac_ch, nqz=cfg['q_nqz'])
        self.declare_gen(ch=ro_dac_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_adc_ch, length=cfg['ro_len'])

        self.add_loop("t1_wait_time_loop", self.cfg["t1_steps"])

        # Pump pulse definition
        pi_kwargs, pi_gain = add_pi_envelope(
            self, cfg, q_dac_ch, prefix="t1", label="T1")
        self.add_pulse(ch=q_dac_ch, name="pi_pulse",
                       **pi_kwargs,
                       freq=cfg['q_freq'],
                       phase=cfg['q_pi_phase'],
                       gain=pi_gain,
                      )

        # Probe pulse definition
        self.add_readoutconfig(ch=ro_adc_ch, name="readout", freq=cfg['ro_freq'], gen_ch=ro_dac_ch)
        self.add_pulse(ch=ro_dac_ch, name="read_pulse", ro_ch=ro_adc_ch,
                       style="const",
                       freq=cfg['ro_freq'],
                       length=cfg['ro_len'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ro_gain'],
                      )

    def _body(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']
        q_dac_ch = cfg['q_dac_ch']

        self.send_readoutconfig(ch=ro_adc_ch, name="readout", t=0)

        self.pulse(ch=q_dac_ch, name='pi_pulse', t=0)
        self.delay_auto(t=0, gens=True, ros=False)
        self.delay_auto(t=cfg['t1_wait_time_loop_var'], gens=True, ros=False, tag='wait_time')
        self.pulse(ch=ro_dac_ch, name="read_pulse", t=0.0)
        self.trigger(ros=[ro_adc_ch], pins=[0], t=cfg['trig_time'])


@QickBoardSweep(
    TimeVariable("t", time_parameter='wait_time', unit="us"),
    ComplexQICKData("signal", depends_on=["t"])
)
class T2RProgram(AveragerProgramV2):
    '''
    Measures the T2 Ramsey time of the qubit by the Ramsey measurement.

    '''
    def _initialize(self, cfg):
        if cfg["n_echoes"] != 0:
            raise ValueError("n_echoes should be 0 for T2R program!")
        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']
        q_dac_ch = cfg['q_dac_ch']

        self.declare_gen(ch=q_dac_ch, nqz=cfg['q_nqz'])
        self.declare_gen(ch=ro_dac_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_adc_ch, length=cfg['ro_len'])

        self.add_loop("t2r_wait_time_loop", self.cfg["t2r_steps"])

        # Pump pulse definition
        pi_kwargs, pi_gain = add_pi_envelope(
            self, cfg, q_dac_ch, prefix="t2r", label="T2 Ramsey")
        self.add_pulse(ch=q_dac_ch, name="pi_2_pulse",
                       **pi_kwargs,
                       freq=cfg['q_freq'],
                       phase=cfg['q_pi_phase'],
                       gain=pi_gain/2,
                      )
        self.add_pulse(ch=q_dac_ch, name="pi_2_detuned_pulse",
                       **pi_kwargs,
                       freq=cfg['q_freq'],
                       phase=cfg['t2r_wait_time_loop_var']*cfg['q_detuning']*360,
                       gain=pi_gain/2,
                      )

        # Probe pulse
        self.add_readoutconfig(ch=ro_adc_ch, name="readout", freq=cfg['ro_freq'], gen_ch=ro_dac_ch)
        self.add_pulse(ch=ro_dac_ch, name="read_pulse", ro_ch=ro_adc_ch,
                       style="const",
                       freq=cfg['ro_freq'],
                       length=cfg['ro_len'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ro_gain'],
                      )

    def _body(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']
        q_dac_ch = cfg['q_dac_ch']

        self.send_readoutconfig(ch=ro_adc_ch, name="readout", t=0)

        # pi/2 pulse
        self.pulse(ch=q_dac_ch, name='pi_2_pulse', t=0)
        self.delay_auto(t=0.0, gens=True, ros=False)

        # wait
        self.delay_auto(t=cfg['t2r_wait_time_loop_var'], gens=True, ros=False, tag='wait_time')

        # detuned pi/2 pulse
        self.pulse(ch=q_dac_ch, name='pi_2_detuned_pulse', t=0)
        self.delay_auto(t=0.0, gens=True, ros=False)
        self.pulse(ch=ro_dac_ch, name="read_pulse", t=0)
        self.trigger(ros=[ro_adc_ch], pins=[0], t=cfg['trig_time'])


@QickBoardSweep(
    TimeVariable("t", time_parameter="wait_time", unit="us"),
    ComplexQICKData("signal", depends_on=["t"])
)
class T2nProgram(AveragerProgramV2):
    '''
    Measures the T2 echo time of the qubit using the Hahn echo experiment based on the number of pi pulses you give it.
    When executed without any artificial detuning set by phases, one can fine-tune
    the frequency of the qubit from the fit.

    '''
    def _initialize(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']
        q_dac_ch = cfg['q_dac_ch']

        self.declare_gen(ch=q_dac_ch, nqz=cfg['q_nqz'])
        self.declare_gen(ch=ro_dac_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_adc_ch, length=cfg['ro_len'])

        self.add_loop("t2e_wait_time_loop", self.cfg["t2e_steps"])

        # Pump pulse definition
        pi_kwargs, pi_gain = add_pi_envelope(
            self, cfg, q_dac_ch, prefix="t2n", label="T2 echo")
        self.add_pulse(ch=q_dac_ch, name="pi_2_pulse",
                       **pi_kwargs,
                       freq=cfg['q_freq'],
                       phase=cfg['q_pi_phase'],
                       gain=pi_gain/2,
                      )
        self.add_pulse(ch=q_dac_ch, name="pi_2_detuned_pulse",
                       **pi_kwargs,
                       freq=cfg['q_freq'],
                       phase=cfg['t2e_wait_time_loop_var']*cfg['q_detuning']*360,
                       gain=pi_gain/2,
                      )
        self.add_pulse(ch=q_dac_ch, name="pi_pulse",
                       **pi_kwargs,
                       freq=cfg['q_freq'],
                       phase=cfg['q_pi_phase'],
                       gain=pi_gain,
                      )

        self.add_readoutconfig(ch=ro_adc_ch, name="readout", freq=cfg['ro_freq'], gen_ch=ro_dac_ch)
        self.add_pulse(ch=ro_dac_ch, name="read_pulse", ro_ch=ro_adc_ch,
                       style="const",
                       freq=cfg['ro_freq'],
                       length=cfg['ro_len'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ro_gain'],
                      )

    def _body(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']
        q_dac_ch = cfg['q_dac_ch']

        self.send_readoutconfig(ch=ro_adc_ch, name="readout", t=0)

        # pi/2 pulse
        self.pulse(ch=q_dac_ch, name='pi_2_pulse', t=0)

        # wait & echoes
        self.delay_auto(t=cfg['t2e_wait_time_loop_var']/(cfg['n_echoes']+1), gens=True, ros=False, tag='wait_time')
        for i in range(cfg['n_echoes']):
            self.pulse(ch=q_dac_ch, name='pi_pulse', t=0)
            self.delay_auto(t=cfg['t2e_wait_time_loop_var']/(cfg['n_echoes']+1), gens=True, ros=False)

        # detuned pi/2 pulse
        self.pulse(ch=q_dac_ch, name='pi_2_detuned_pulse', t=0)
        self.delay_auto(t=0.00, gens=True, ros=False)
        self.pulse(ch=ro_dac_ch, name="read_pulse", t=0.00)
        self.trigger(ros=[ro_adc_ch], pins=[0], t=cfg['trig_time'])


@QickBoardSweep(
    independent('repetition'),
    ComplexQICKData("g", depends_on=['repetition']),
)
class SingleShotGroundProgram(AveragerProgramV2):
    '''
    Program for readout calibration on IQ plane. This simply takes shots of ground state.
    One has to set reps to 1 to avoid averaging of shots.
    '''

    def _initialize(self, cfg):
        if cfg["reps"] != 1:
            raise ValueError("reps should be 1!")

        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']
        q_dac_ch = cfg['q_dac_ch']

        self.declare_gen(ch=q_dac_ch, nqz=cfg['q_nqz'])
        self.declare_gen(ch=ro_dac_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_adc_ch, length=cfg['ro_len'])

        self.add_loop("shot_loop", cfg['steps'])

        self.add_readoutconfig(ch=ro_adc_ch, name="readout", freq=cfg['ro_freq'], gen_ch=ro_dac_ch)
        self.add_pulse(ch=ro_dac_ch, name="read_pulse", ro_ch=ro_adc_ch,
                       style="const",
                       freq=cfg['ro_freq'],
                       length=cfg['ro_len'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ro_gain'],
                       )

    def _body(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']

        self.send_readoutconfig(ch=ro_adc_ch, name="readout", t=0)

        # ground state measurement
        self.delay_auto(t=0, gens=True, ros=False)
        self.pulse(ch=ro_dac_ch, name="read_pulse", t=0)
        self.trigger(ros=[ro_adc_ch], pins=[0], t=cfg['trig_time'])


@QickBoardSweep(
    independent('repetition'),
    ComplexQICKData("e", depends_on=['repetition'])
)
class SingleShotExcitedProgram(AveragerProgramV2):
    '''
    Program for readout calibration on IQ plane. Takes singles shots in excited state.
    One has to set reps to 1 to avoid averaging of shots.
    '''

    def _initialize(self, cfg):
        if cfg["reps"] != 1:
            raise ValueError("reps should be 1!")

        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']
        q_dac_ch = cfg['q_dac_ch']

        self.declare_gen(ch=q_dac_ch, nqz=cfg['q_nqz'])
        self.declare_gen(ch=ro_dac_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_adc_ch, length=cfg['ro_len'])

        self.add_loop("shot_loop", cfg['steps'])
        pi_kwargs, pi_gain = add_pi_envelope(
            self, cfg, q_dac_ch, prefix="single_shot", label="Single-shot excited")
        self.add_pulse(ch=q_dac_ch, name="pi_pulse", **pi_kwargs,
                       freq=cfg['q_freq'], phase=cfg['q_pi_phase'], gain=pi_gain)

        self.add_readoutconfig(ch=ro_adc_ch, name="readout", freq=cfg['ro_freq'], gen_ch=ro_dac_ch)
        self.add_pulse(ch=ro_dac_ch, name="read_pulse", ro_ch=ro_adc_ch,
                       style="const",
                       freq=cfg['ro_freq'],
                       length=cfg['ro_len'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ro_gain'],
                       )

    def _body(self, cfg):
        ro_adc_ch = cfg['ro_adc_ch']
        ro_dac_ch = cfg['ro_dac_ch']
        q_dac_ch = cfg['q_dac_ch']

        self.send_readoutconfig(ch=ro_adc_ch, name="readout", t=0)

        # excited state measurement
        self.pulse(ch=q_dac_ch, name='pi_pulse', t=0)
        self.delay_auto(t=0, gens=True, ros=True)
        self.pulse(ch=ro_dac_ch, name="read_pulse", t=0)
        self.trigger(ros=[ro_adc_ch], pins=[0], t=cfg['trig_time'])

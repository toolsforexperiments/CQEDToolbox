# the main program class
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

class OffsetProgram(AveragerProgramV2):
    """
    To calibrate the adc trigger offset caused by electrical delays.
    This function is not supported by the QICK sweep wrapper.
    """
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        gen_ch = cfg['ro_gen_ch']
        
        self.declare_gen(ch=gen_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_ch, length=cfg['ro_len'])

        self.add_readoutconfig(ch=ro_ch, name="myro", freq=cfg['ro_freq'], gen_ch=gen_ch)

        self.add_pulse(ch=gen_ch, name="probe_pulse", ro_ch=ro_ch,
                        style="const", freq=cfg['ro_freq'], length=cfg['ro_len'],
                        phase=cfg['ro_phase'], gain=cfg['ro_gain'])

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)

        self.pulse(ch=cfg['ro_gen_ch'], name="probe_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])

        
@QickBoardSweep(
    PulseVariable('freq', pulse_parameter="probe_pulse", sweep_parameter="freq"),
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

        self.add_loop("ro_freqs_loop", self.cfg["ro_steps"])
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
    PulseVariable('freq', pulse_parameter="probe_pulse", sweep_parameter="freq"),
    PulseVariable('gain', pulse_parameter="probe_pulse", sweep_parameter="gain"),
    ComplexQICKData('signal', depends_on=['freq', 'gain'])
)
class FreqGainSweepProgram(AveragerProgramV2):
    '''
    Performs a sweep over gain vs single tone spectroscopy on the resonator.
    '''

    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        gen_ch = cfg['ro_gen_ch']
        
        self.declare_gen(ch=gen_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_ch, length=cfg['ro_len'])

        self.add_loop("ro_gain_loop", self.cfg["steps2"])
        self.add_loop("ro_freq_loop", self.cfg["steps"])

        self.add_readoutconfig(ch=ro_ch, name="myro", freq=cfg['ro_freqs'], gen_ch=gen_ch)  # Sweep variable

        self.add_pulse(ch=gen_ch, name="probe_pulse", ro_ch=ro_ch, 
                       style="const", 
                       freq=cfg['ro_freqs'],  # Sweep variable 
                       length=cfg['ro_len'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ro_gains'],  # Sweep variable 
                      )
        
    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)

        self.pulse(ch=cfg['ro_gen_ch'], name="probe_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


@QickBoardSweep(
    PulseVariable("freq", pulse_parameter="probe_pulse", sweep_parameter="freq"),
    ComplexQICKData("signal", depends_on=["freq"])
)
class ResProbeProgram(AveragerProgramV2):
    '''
    Performs single tone spectroscopy on RO resonator after exciting g->e transition of qubit.
    '''
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        gen_ch = cfg['ro_gen_ch']
        q_gen_ch = cfg['q_gen_ch']

        self.declare_gen(ch=q_gen_ch, nqz=cfg['q_nqz'])
        self.declare_gen(ch=gen_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_ch, length=cfg['ro_len'])

        self.add_loop("ro_freq_loop", self.cfg["steps"])

        self.add_gauss(ch=q_gen_ch, name="gauss", sigma=cfg['q_ge_sig'], length=4*cfg['q_ge_sig'], even_length=True)        
        self.add_pulse(ch=q_gen_ch, name="pi_pulse",  
                       style="arb", 
                       envelope="gauss", 
                       freq=cfg['q_ge'], 
                       phase=cfg['q_ge_phase'],
                       gain=cfg['q_ge_gain'], 
                      )
        self.add_readoutconfig(ch=ro_ch, name="myro", freq=cfg['ro_freqs'], gen_ch=gen_ch)

        self.add_pulse(ch=gen_ch, name="probe_pulse", ro_ch=ro_ch, 
                       style="const", 
                       freq=cfg['ro_freqs'],  # Sweep variable
                       length=cfg['ro_len'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ro_gain'], 
                      )
        
    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)

        self.pulse(ch=cfg['q_gen_ch'], name='pi_pulse', t=0)
        self.delay_auto(t=0.00, gens=True, ros=False)  
        self.pulse(ch=cfg['ro_gen_ch'], name="probe_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])
        
        
@QickBoardSweep(
    PulseVariable('freq', pulse_parameter="pump_pulse", sweep_parameter="freq"),
    ComplexQICKData('signal', depends_on=['freq'])
)
class PulseProbeSpectroscopy(AveragerProgramV2):
    '''
    Performs two-tone spectroscopy where the probe tone is probing the readout resonator
    while the pump tone excites the qubit.
    '''
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        ro_gen_ch = cfg['ro_gen_ch']
        q_gen_ch = cfg['q_gen_ch']

        self.declare_gen(ch=q_gen_ch, nqz=cfg['q_nqz'])
        self.declare_gen(ch=ro_gen_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_ch, length=cfg['ro_len'])

        # Pump pulse defintiion
        self.add_loop("q_ge_freq_loop", self.cfg["steps"])
        self.add_pulse(ch=q_gen_ch, name='pump_pulse',
                        style='const',
                        freq=cfg['q_ges'],  # Sweep variable
                        length=cfg['q_flat_len'],
                        phase=cfg['q_ge_phase'],
                        gain=cfg['q_ge_gain'])
        
        # Probe pulse definition
        self.add_readoutconfig(ch=ro_ch, name="myro", freq=cfg['ro_freq'], gen_ch=ro_gen_ch)
        self.add_pulse(ch=ro_gen_ch, name="read_pulse", ro_ch=ro_ch, 
                       style="const", 
                       freq=cfg['ro_freq'], 
                       length=cfg['ro_len'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ro_gain'], 
                      )

    def _body(self, cfg):
        # if you delay the config by too long, you can see the readout get reconfigured in the middle of your pulse
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)

        self.pulse(ch=cfg['q_gen_ch'], name='pump_pulse', t=0)
        self.delay_auto(t=0.05, gens=True, ros=False, tag='wait_time')
        self.pulse(ch=cfg['ro_gen_ch'], name="read_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


@QickBoardSweep(
    PulseVariable("gain", pulse_parameter='pi_pulse', sweep_parameter="gain"),
    ComplexQICKData("signal", depends_on=["gain"])
)
class AmplitudeRabiProgram(AveragerProgramV2):
    '''
    Calibrates the power of the pi pulse where the pulse has the Gaussian pulse shape.
    Note that the current version of the QICK firmware supports 16-bit signed gain.
    '''
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        ro_gen_ch = cfg['ro_gen_ch']
        q_gen_ch = cfg['q_gen_ch']

        self.declare_gen(ch=q_gen_ch, nqz=cfg['q_nqz'])
        self.declare_gen(ch=ro_gen_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_ch, length=cfg['ro_len'])
        
        # Pump pulse definition
        self.add_loop("q_ge_gain_loop", self.cfg["steps"])
        self.add_gauss(ch=q_gen_ch, name="gauss", sigma=cfg['q_ge_sig'], length=4*cfg['q_ge_sig'], even_length=True)        
        self.add_pulse(ch=q_gen_ch, name="pi_pulse",  
                       style="arb", 
                       envelope="gauss", 
                       freq=cfg['q_ge'], 
                       phase=cfg['q_ge_phase'],
                       gain=cfg['q_ge_gains'],  # Sweep variable 
                      )
        
        # Probe pulse definition
        self.add_readoutconfig(ch=ro_ch, name="myro", freq=cfg['ro_freq'], gen_ch=ro_gen_ch)
        self.add_pulse(ch=ro_gen_ch, name="read_pulse", ro_ch=ro_ch, 
                       style="const", 
                       freq=cfg['ro_freq'], 
                       length=cfg['ro_len'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ro_gain'], 
                      )
        
    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)

        self.pulse(ch=cfg['q_gen_ch'], name='pi_pulse', t=0)
        self.delay_auto(t=0.0, gens=True, ros=False, tag='wait_time')
        self.pulse(ch=cfg['ro_gen_ch'], name="read_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])



@QickBoardSweep(
    PulseVariable("freq", pulse_parameter='pi_pulse', sweep_parameter="freq"),
    ComplexQICKData("signal", depends_on=["freq"])
)
class PiSpecProgram(AveragerProgramV2):
    '''
    This runs a pi pulse spectroscopy in order to nail down the qubit frequency by fitting it to a gaussian pulse.
    '''
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        ro_gen_ch = cfg['ro_gen_ch']
        q_gen_ch = cfg['q_gen_ch']

        self.declare_gen(ch=q_gen_ch, nqz=cfg['q_nqz'])
        self.declare_gen(ch=ro_gen_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_ch, length=cfg['ro_len'])

        # Pump pulse definition
        self.add_loop("q_ge_freq_loop", self.cfg["steps"])
        self.add_gauss(ch=q_gen_ch, name="gauss", sigma=cfg['q_ge_sig'], length=4*cfg['q_ge_sig'], even_length=True)        
        self.add_pulse(ch=q_gen_ch, name="pi_pulse",  
                       style="arb", 
                       envelope="gauss", 
                       freq=cfg['q_ges'],  # Sweep variable
                       phase=cfg['q_ge_phase'],
                       gain=cfg['q_ge_gain'], 
                      )
        
        # Probe pulse definition
        self.add_readoutconfig(ch=ro_ch, name="myro", freq=cfg['ro_freq'], gen_ch=ro_gen_ch)
        self.add_pulse(ch=ro_gen_ch, name="read_pulse", ro_ch=ro_ch, 
                       style="const", 
                       freq=cfg['ro_freq'], 
                       length=cfg['ro_len'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ro_gain'], 
                      )
        
    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)

        self.pulse(ch=cfg['q_gen_ch'], name='pi_pulse', t=0)
        self.delay_auto(t=0.0, gens=True, ros=False, tag='wait_time')
        self.pulse(ch=cfg['ro_gen_ch'], name="read_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


@QickBoardSweep(
    TimeVariable("t", time_parameter='wait_time'),
    ComplexQICKData("signal", depends_on=["t"])
)
class T1Program(AveragerProgramV2):
    '''
    Measures the T1 time of the qubit by the Rabi measurement.
    This programs wants users to set n_echoes to 0.
    '''
    def _initialize(self, cfg):
        if cfg["n_echoes"] != 0:
            raise ValueError("n_echoes should be 0!")

        ro_ch = cfg['ro_ch']
        ro_gen_ch = cfg['ro_gen_ch']
        q_gen_ch = cfg['q_gen_ch']

        self.declare_gen(ch=q_gen_ch, nqz=cfg['q_nqz'])
        self.declare_gen(ch=ro_gen_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_ch, length=cfg['ro_len'])
        
        self.add_loop("T1_wait_time_loop", self.cfg["steps"])

        # Pump pulse definition
        self.add_gauss(ch=q_gen_ch, name="gauss", sigma=cfg['q_ge_sig'], length=4*cfg['q_ge_sig'], even_length=True)        
        self.add_pulse(ch=q_gen_ch, name="pi_pulse",  
                       style="arb", 
                       envelope="gauss", 
                       freq=cfg['q_ge'], 
                       phase=cfg['q_ge_phase'],
                       gain=cfg['q_ge_gain'], 
                      )
        
        # Probe pulse definition
        self.add_readoutconfig(ch=ro_ch, name="myro", freq=cfg['ro_freq'], gen_ch=ro_gen_ch)
        self.add_pulse(ch=ro_gen_ch, name="read_pulse", ro_ch=ro_ch, 
                       style="const",
                       freq=cfg['ro_freq'], 
                       length=cfg['ro_len'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ro_gain'], 
                      )
        
    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)

        self.pulse(ch=cfg['q_gen_ch'], name='pi_pulse', t=0)
        self.delay_auto(t=0, gens=True, ros=False)
        self.delay_auto(t=cfg['wait_time_T1'], gens=True, ros=False, tag='wait_time')
        self.pulse(ch=cfg['ro_gen_ch'], name="read_pulse", t=0.0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


@QickBoardSweep(
    TimeVariable("t", time_parameter='wait_time'),
    ComplexQICKData("signal", depends_on=["t"])
)
class T2RProgram(AveragerProgramV2):
    '''
    Measures the T2 Ramsey time of the qubit by the Ramsey measurement.
    This programs wants users to set n_echoes to 0.
    '''
    def _initialize(self, cfg):
        if cfg["n_echoes"] != 0:
            raise ValueError("n_echos should be 0!")
    
        ro_ch = cfg['ro_ch']
        ro_gen_ch = cfg['ro_gen_ch']
        q_gen_ch = cfg['q_gen_ch']

        self.declare_gen(ch=q_gen_ch, nqz=cfg['q_nqz'])
        self.declare_gen(ch=ro_gen_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_ch, length=cfg['ro_len'])

        self.add_loop("T2R_wait_time_loop", self.cfg["steps"])
        
        # Pump pulse definition
        self.add_gauss(ch=q_gen_ch, name="gauss", sigma=cfg['q_ge_sig'], length=4*cfg['q_ge_sig'], even_length=True)
        ## pi/2 pulse        
        self.add_pulse(ch=q_gen_ch, name="pi_2_pulse",  
                       style="arb", 
                       envelope="gauss", 
                       freq=cfg['q_ge'], 
                       phase=cfg['q_ge_phase'],
                       gain=cfg['q_ge_gain']/2, 
                      )
                
        self.add_pulse(ch=q_gen_ch, name="pi_2_detuned_pulse",  
                       style="arb", 
                       envelope="gauss", 
                       freq=cfg['q_ge'], 
                       phase=cfg['wait_time_T2R']*cfg['q_ge_detuning']*360,  # Artificial detuning for observation of Ramsey oscillation
                       gain=cfg['q_ge_gain']/2, 
                      )
        
        # Probe pulse
        self.add_readoutconfig(ch=ro_ch, name="myro", freq=cfg['ro_freq'], gen_ch=ro_gen_ch)
        self.add_pulse(ch=ro_gen_ch, name="read_pulse", ro_ch=ro_ch, 
                       style="const", 
                       freq=cfg['ro_freq'], 
                       length=cfg['ro_len'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ro_gain'], 
                      )
        
    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)

        # pi/2 pulse
        self.pulse(ch=cfg['q_gen_ch'], name='pi_2_pulse', t=0)
        self.delay_auto(t=0.0, gens=True, ros=False)

        # wait
        self.delay_auto(t=cfg['wait_time_T2R'], gens=True, ros=False, tag='wait_time')
        
        # detuned pi/2 pulse
        self.pulse(ch=cfg['q_gen_ch'], name='pi_2_detuned_pulse', t=0)
        self.delay_auto(t=0.0, gens=True, ros=False)
        self.pulse(ch=cfg['ro_gen_ch'], name="read_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


@QickBoardSweep(
    TimeVariable("t", time_parameter="wait_time"),
    ComplexQICKData("signal", depends_on=["t"])
)
class T2nProgram(AveragerProgramV2):
    '''
    Measures the T2 n time of the qubit using the Ramsey experiment based on the number of pi pulses you give it.
    When executed without any artificial detuning set by phases, one can fine-tune
    the frequency of the qubit from the fit.
    '''
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        ro_gen_ch = cfg['ro_gen_ch']
        q_gen_ch = cfg['q_gen_ch']

        self.declare_gen(ch=q_gen_ch, nqz=cfg['q_nqz'])
        self.declare_gen(ch=ro_gen_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_ch, length=cfg['ro_len'])

        self.add_loop("T2E_wait_time_loop", self.cfg["steps"])

        # Pump pulse definition
        self.add_gauss(ch=q_gen_ch, name="gauss", sigma=cfg['q_ge_sig'], length=4*cfg['q_ge_sig'], even_length=True)        
        ## pi/2 pulse
        self.add_pulse(ch=q_gen_ch, name="pi_2_pulse",  
                       style="arb", 
                       envelope="gauss", 
                       freq=cfg['q_ge'], 
                       phase=cfg['q_ge_phase'],
                       gain=cfg['q_ge_gain']/2, 
                      )
        
        self.add_pulse(ch=q_gen_ch, name="pi_2_detuned_pulse",  
                       style="arb", 
                       envelope="gauss", 
                       freq=cfg['q_ge'], 
                       phase=cfg['wait_time_T2E']*cfg['q_ge_detuning']*360,
                       gain=cfg['q_ge_gain']/2, 
                      )
        ## pi pulse
        self.add_pulse(ch=q_gen_ch, name="pi_pulse",  
                       style="arb", 
                       envelope="gauss", 
                       freq=cfg['q_ge'], 
                       phase=cfg['q_ge_phase'],
                       gain=cfg['q_ge_gain'], 
                      )
        
        self.add_readoutconfig(ch=ro_ch, name="myro", freq=cfg['ro_freq'], gen_ch=ro_gen_ch)
        self.add_pulse(ch=ro_gen_ch, name="read_pulse", ro_ch=ro_ch, 
                       style="const", 
                       freq=cfg['ro_freq'], 
                       length=cfg['ro_len'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ro_gain'], 
                      )
        
    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)

        # pi/2 pulse
        self.pulse(ch=cfg['q_gen_ch'], name='pi_2_pulse', t=0)

        # wait & echoes
        self.delay_auto(t=cfg['wait_time_T2E']/(cfg['n_echoes']+1), gens=True, ros=False, tag='wait_time')
        for i in range(cfg['n_echoes']):
            self.pulse(ch=cfg['q_gen_ch'], name='pi_pulse', t=0)
            self.delay_auto(t=cfg['wait_time_T2E']/(cfg['n_echoes']+1), gens=True, ros=False)

        # detuned pi/2 pulse
        self.pulse(ch=cfg['q_gen_ch'], name='pi_2_detuned_pulse', t=0)
        self.delay_auto(t=0.00, gens=True, ros=False)
        self.pulse(ch=cfg['ro_gen_ch'], name="read_pulse", t=0.00)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


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

        ro_ch = cfg['ro_ch']
        ro_gen_ch = cfg['ro_gen_ch']
        q_gen_ch = cfg['q_gen_ch']

        self.declare_gen(ch=q_gen_ch, nqz=cfg['q_nqz'])
        self.declare_gen(ch=ro_gen_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_ch, length=cfg['ro_len'])

        self.add_loop("shot_loop", cfg['steps'])
        self.add_gauss(ch=q_gen_ch, name="gauss", sigma=cfg['q_ge_sig'], length=4 * cfg['q_ge_sig'], even_length=True)
        self.add_pulse(ch=q_gen_ch, name="pi_pulse",
                       style="arb",
                       envelope="gauss",
                       freq=cfg['q_ge'],
                       phase=cfg['q_ge_phase'],
                       gain=cfg['q_ge_gain'],
                       )

        self.add_readoutconfig(ch=ro_ch, name="myro", freq=cfg['ro_freq'], gen_ch=ro_gen_ch)
        self.add_pulse(ch=ro_gen_ch, name="read_pulse", ro_ch=ro_ch,
                       style="const",
                       freq=cfg['ro_freq'],
                       length=cfg['ro_len'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ro_gain'],
                       )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)

        # # ground state measurement
        self.delay_auto(t=0, gens=True, ros=False)
        self.pulse(ch=cfg['ro_gen_ch'], name="read_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


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

        ro_ch = cfg['ro_ch']
        ro_gen_ch = cfg['ro_gen_ch']
        q_gen_ch = cfg['q_gen_ch']

        self.declare_gen(ch=q_gen_ch, nqz=cfg['q_nqz'])
        self.declare_gen(ch=ro_gen_ch, nqz=cfg['ro_nqz'])
        self.declare_readout(ch=ro_ch, length=cfg['ro_len'])

        self.add_loop("shot_loop", cfg['steps'])
        self.add_gauss(ch=q_gen_ch, name="gauss", sigma=cfg['q_ge_sig'], length=4 * cfg['q_ge_sig'], even_length=True)
        self.add_pulse(ch=q_gen_ch, name="pi_pulse",
                       style="arb",
                       envelope="gauss",
                       freq=cfg['q_ge'],
                       phase=cfg['q_ge_phase'],
                       gain=cfg['q_ge_gain'],
                       )

        self.add_readoutconfig(ch=ro_ch, name="myro", freq=cfg['ro_freq'], gen_ch=ro_gen_ch)
        self.add_pulse(ch=ro_gen_ch, name="read_pulse", ro_ch=ro_ch,
                       style="const",
                       freq=cfg['ro_freq'],
                       length=cfg['ro_len'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ro_gain'],
                       )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)

        # # excited state measurement
        self.pulse(ch=cfg['q_gen_ch'], name='pi_pulse', t=0)
        self.delay_auto(t=0, gens=True, ros=True)
        self.pulse(ch=cfg['ro_gen_ch'], name="read_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])
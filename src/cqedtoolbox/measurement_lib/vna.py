"""Tools for VNA measurements."""
from typing import Optional
from time import sleep

from labcore.measurement import recording, independent, dependent, indep, dep

from cqedtoolbox.instruments.qcodes_drivers.SignalCore.SignalCore_sc5511a import SignalCore_SC5511A
from cqedtoolbox.measurement_lib.vna_type import VnaType

#: VNA

vna: Optional[VnaType] = None

#: qubit generator -- used for twotone spec
qubit_generator: Optional[SignalCore_SC5511A] = None

# Make sure this is called before referencing the functions below
def set_vna(instrument: VnaType):
    global vna
    vna = instrument

@recording(
    independent('frequency', unit='Hz'),
    dependent('trace', depends_on=['frequency'])
)
def current_vna_trace():
    """Get the 1st trace from the vna.
    Return frequency and trace data.

    VNA must be set on the module level.
    """
    frq = vna.trace_1.frequency()
    trace = vna.trace_1.data()
    return frq, trace


@recording(
    indep('frequency', unit='Hz'),
    dep('signal', depends_on=['frequency']),
)
def twotone_qubit_spec(qubit_frequencies, naverages=1, dwell_time=50e-3):
    """
    measure qubit two-tone spec with a vna and generator in list-mode.

    Set the VNA (``VNA``) and generator (``qubit_generator``)
    on the module level before using.

    things that need to be set manually:
        - generator power
        - vna center frequency and power

    Parameters
    ----------
    qubit_frequencies : np.ndarray
        list of qubit probe frequencies. must be equidistant.
    naverages : int, optional
        number of averages on the vna. The default is 1.
    dwell_time : float, optional
        wait time per frequencies for the generator. The default is 50e-3.

    Returns
    -------
    qubit_frqs : np.ndarray
        list of qubit frequencies.
    data : np.ndarray
        vna trace data.

    """
    configure_vna_for_twotone_spec(naverages)
    configure_qubit_generator_for_twotone_spec(qubit_frequencies, naverages, dwell_time)

    vna.num_points(qubit_frequencies.size)  # number of point on the VNA x axis
    vna.clear_averages()  # clear the average in the VNA and wait for trigger

    # set the generator mode to sweep/list mode
    # here just to stop any other sweep that is still running
    # first sweep to single fixed tone then to sweep/list mode
    qubit_generator.rf1_mode(0)
    qubit_generator.rf1_mode(1)

    # turn on the generator
    qubit_generator.output_status(1)

    # trigger generator to start the sweep
    qubit_generator.soft_trigger()

    print('Generator is done')

    sleep(qubit_frequencies.size * dwell_time * naverages * 1.1)

    # turn off the qubit generator and set it back to normal fixed tone mode
    qubit_generator.output_status(0)
    qubit_generator.rf1_mode(0)

    data = vna.trace_1.data()
    return qubit_frequencies, data


def configure_vna_for_twotone_spec(naverages):
    vna.fspan(0)
    vna.trigger_source('EXT')  # set the trigger
    vna.sweep_mode('CONT')  # sweep mode set to continous
    vna.trigger_mode('POIN')  # trigger mode set to point
    vna.avg_num(naverages)  # set how many times of average
    vna.averaging(1)


def configure_qubit_generator_for_twotone_spec(frequencies, naverages, dwell_time):
    start, stop, step = frequencies[0], frequencies[-1], frequencies[1] - frequencies[0]
    qubit_generator.sweep_start_frequency(start)
    qubit_generator.sweep_stop_frequency(stop)
    qubit_generator.sweep_step_frequency(step)

    # qubit_generator.power(qubit_drive_power)
    # set the dwell time
    qubit_generator.sweep_dwell_time(int(dwell_time * 1e3 / 0.5))
    # set the cycle number
    qubit_generator.sweep_cycles(naverages)

    # set the generator mode to 1, means sweep mode
    qubit_generator.sss_mode(1)
    # enable generator to set output trigger
    qubit_generator.trig_out_enable(1)
    # send out trigger on every frequency point
    qubit_generator.trig_out_on_cycle(0)
    # make that the generator can be trigger by a software trigger
    qubit_generator.step_on_hw_trig(0)
    # do return to start at the end of the sweep
    qubit_generator.return_to_start(0)
    # set to softwar trigger
    qubit_generator.hw_trigger(0)
    # set the generator so that it do not sweep reverse the direction of the sweep
    qubit_generator.tri_waveform(0)
    # set the sweep direction to go from low to high
    qubit_generator.sweep_dir(0)

# Non-labcore sweep
def s_parameter_vs_freq(start_frequency, stop_frequency, num_points=200, s_parameter='S21', naverages=None, settling_time=1):
    """
    Acquire one complex S-parameter trace over a frequency range.

    Configures "trace_1" for the selected S-parameter, sets the VNA
    center frequency, span, and number of sweep points, enables internal
    continuous sweeping, clears the averaging buffer, waits for acquisition,
    and returns the trace frequency axis and complex data.

    This function is compatible with the Keysight and M5180 VNA drivers.
    The M5180 uses ``settling_time`` to allow the sweep and averaging to
    complete; the Keysight driver waits internally when trace data is read.

    Note: To use in a labcore sweep, add the @recording decorator with appropriate independent and dependent variables.
    Ex: 
        recording(
            independent('frequency', unit='Hz'),
            dependent('trace', depends_on=['frequency'])
                ) (s_parameter_vs_freq)

    Parameters
    ----------
    start_frequency : float
        Sweep start frequency in Hz.
    stop_frequency : float
        Sweep stop frequency in Hz. Must exceed ``start_frequency``.
    num_points : int, optional
        Number of frequency points in the VNA sweep. Default is 200.
    s_parameter : {'S11', 'S12', 'S21', 'S22'}, optional
        S-parameter assigned to ``trace_1``. Default is ``'S21'``.
    naverages : int or None, optional
        VNA average count. If None, retains the currently configured value;
        this is useful when an outer LabCore sweep sets ``avg_num``.
    settling_time : float, optional
        Delay in seconds before reading the trace. For the M5180, choose a
        value long enough for the configured IF bandwidth, point count, and
        average count. Default is 1 second.

    Returns
    -------
    frequency : numpy.ndarray
        Frequency values in Hz.
    trace : numpy.ndarray or str
        Complex S-parameter data from ``trace_1``. Older InstrumentServer
        versions may return its serialized representation as a string.
    """
    vna.fcenter((start_frequency + stop_frequency) / 2)
    vna.fspan(stop_frequency - start_frequency)
    vna.num_points(num_points)
    vna.trace_1.s_parameter(s_parameter)

    vna.trigger_source("IMM")
    vna.trigger_mode("SWE")
    vna.sweep_mode("CONT")
    if naverages is not None:
        vna.avg_num(naverages)
    vna.averaging(1)
    vna.clear_averages()

    sleep(settling_time)

    freqs = vna.trace_1.frequency()
    trace = vna.trace_1.data()

    return freqs, trace

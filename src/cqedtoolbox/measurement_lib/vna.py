"""Tools for VNA measurements."""
from typing import Optional
from time import sleep

from labcore.measurement import recording, independent, dependent, indep, dep

from cqedtoolbox.instruments.qcodes_drivers.Keysight.Keysight_P937A import Keysight_P9374A_SingleChannel
from cqedtoolbox.instruments.qcodes_drivers.SignalCore.SignalCore_sc5511a import SignalCore_SC5511A

#: VNA
vna: Optional[Keysight_P9374A_SingleChannel] = None

#: qubit generator -- used for twotone spec
qubit_generator: Optional[SignalCore_SC5511A] = None


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

from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

import numpy as np

from qm.qua import (
    program,
    declare_stream,
    declare,
    fixed,
    for_,
    save,
    measure,
    demod,
    wait,
    stream_processing,
    update_frequency,
    play,
    align,
    amp,
    assign,
    Cast,
    frame_rotation_2pi,
    if_,
)

from labcore.measurement import independent
from labcore.instruments.opx.sweep import (
    RecordOPXdata,
    ComplexOPXData,
    TimedOPXData,
)

# FIXME: This is temporary while testing the new labcore update
try:
    from labcore.instruments.opx.sweep import RecordPrecompiledOPXdata
except:
    pass

"""
The following measurements are meant to be performed on a single transmon qubit.

Most measurements require specific elements and pulses to be present in the config dict to perform the measurement.
When this is the case, the docstring of each measurement specifies what it needs. Custom ways of preparing the qubit
between repetitions and custom measuring fucntions are supported. For the module measurements to use custom preparing or
measuring functions, the Callables need to be passed to the correct variables. Custom measurement functions should 
accept 2 parameters called I and Q, which represent the I and Q values being collected from the OPX.

All integration weights in the config dictionary should end with the suffixes '_cos' and '_sin' respectively.
The suffixes will be added automatically in the functions when necessary. The shuold not be passed in the argument.

All general measurement related parameters (like the readout_element or readout_pulse) should be inside the
dataclass Options. Multiple different instances of Options can be used to switch the parameters by changing the module
label varaible options.

To get the current values of the parameters inside of options, we recommend using the dataclasses.asdict() method
to get the items in dictionary form.

Specific documentation of what each measurement does coming in the future :).

Parameters
----------
options: Options
    Holds the current instance of Options. This is the instance the measurements in this module look for general 
    parameter values. To change them you can either create a new instance of Options or change an individual value of 
    options.
prepare: Callable
    Holds the function that prepares the qubit between repetitions. Default prepare_by_wait. 
measure_qubit: Callable
    Holds the function that measures the qubit. Default, measure_full_integration.
"""


@dataclass
class Options:
    """
    Holds universal parameters used throughout this module

    Parameters
    ----------
    readout_element:
        The element in the config dict that performs the readout.
    readout_pulse:
        Pulse used by the readout element during readout.
    readout_integration_weight:
        The integration weights should exists in the config dictionary.
        There should be both a sin and cos integration weights with the first part of their name the same followed
        by '_sin' and '_cos'. The function will add the suffixes _sin and _cos.
        e.g. if our readout_integration_weight = 'short_readout', in the config dictionary we have 2 different
        integration weights, 'short_readout_cos' and 'short_readout_sin'. If no integration_weight is passed,
        it is a assumed that the readout_integration_weight is the same as the readout_pulse.
    repetition_delay:
        The desired delay between the repetitions of the measurments in ns.
    readout_pulse_length:
        Only used for sliced integration. The total length of the readout pulse.
    chunksize_ns:
        Only used for sliced integration. The chunksize of the slicded demodulation in ns.
    """

    # what qubit element to run an experiment on
    qubit_element: str = "qubit"

    # The element in the config dict that performs the readout.
    readout_element: str = ""

    # Pulse used by the readout element during readout.
    readout_pulse: str = ""

    # Optional specific integration weights used in measurements.
    readout_integration_weight: Optional[str] = None

    # Universal delay between measurements in ns.
    repetition_delay: Optional[int] = 0  # in ns.

    # Length of the readout pulse in ns. Default, 4000 ns
    readout_pulse_length: Optional[int] = 4000

    # Size of chunks for sliced demodulation in ns
    chunksize_ns: Optional[int] = 40

    # checks whether you are using an octave and the corresponding API
    analog_output: bool = False


options = Options()


def prep_by_wait():
    wait(options.repetition_delay // 4, options.readout_element)
    align()


prepare = prep_by_wait


def measure_full_integration(I, Q):
    integration_weights = _check_readout_integration_weight()
    if options.analog_output:
        measure(
            options.readout_pulse,
            options.readout_element,
            None,
            demod.full(integration_weights + "_cos", I, "out1"),
            demod.full(integration_weights + "_sin", Q, "out1"),
        )

    else:
        measure(
            options.readout_pulse,
            options.readout_element,
            None,
            demod.full(integration_weights + "_cos", I),
            demod.full(integration_weights + "_sin", Q),
        )


def measure_sliced_integration(I, Q, raw_stream=None):
    integration_weights = _check_readout_integration_weight()

    _chunksize = int(options.chunksize_ns // 4)

    if options.analog_output:
        measure(
            options.readout_pulse,
            options.readout_element,
            raw_stream,
            demod.sliced(integration_weights + "_cos", I, _chunksize, "out1"),
            demod.sliced(integration_weights + "_sin", Q, _chunksize, "out1"),
        )

    else:
        measure(
            options.readout_pulse,
            options.readout_element,
            raw_stream,
            demod.sliced(integration_weights + "_cos", I, _chunksize),
            demod.sliced(integration_weights + "_sin", Q, _chunksize),
        )


measure_qubit = measure_full_integration


def _check_readout_integration_weight() -> str:
    """
    Checks wether to use the options.readout_pulse as readout_integration_weight.
    Should be called at the beginning of each measurement, with return saved in a variable that indicates the
    integration weights.
    """
    if options.readout_integration_weight is None:
        weight = options.readout_pulse
    else:
        weight = options.readout_integration_weight
    return weight


# ORIGINALLY WITH LONG READOUT
@RecordOPXdata(
    independent("repetition"),
    independent("ssb_frequency", unit='Hz'),
    ComplexOPXData(
        "signal",
        depends_on=["repetition", "ssb_frequency"],
        i_data_stream="I",
        q_data_stream="Q",
    ),
)
def pulsed_resonator_spec(start, stop, step, n_reps):
    n_frqs = abs(round((stop - start) / step))

    with program() as qua_measurement:
        i_stream = declare_stream()
        q_stream = declare_stream()
        ssb_f_stream = declare_stream()
        rep_stream = declare_stream()

        I = declare(fixed)
        Q = declare(fixed)
        i = declare(int)
        ssb_f = declare(int)

        with for_(i, 0, i < n_reps, i + 1):
            with for_(ssb_f, start, ssb_f < stop, ssb_f + step):
                prepare()

                update_frequency(options.readout_element, ssb_f)

                measure_qubit(I, Q)

                save(I, i_stream)
                save(Q, q_stream)
                save(ssb_f, ssb_f_stream)
                save(i, rep_stream)

        with stream_processing():
            rep_stream.buffer(n_frqs).save_all("repetition")
            ssb_f_stream.buffer(n_frqs).save_all("ssb_frequency")
            i_stream.buffer(n_frqs).save_all("I")
            q_stream.buffer(n_frqs).save_all("Q")

    return qua_measurement

# plays a measures the readout resonator with and without a pi pulse played on the qubit
@RecordOPXdata(
    independent("repetition"),
    independent("setting"),
    independent("ssb_frequency", unit='Hz'),
    ComplexOPXData(
        "signal",
        depends_on=["repetition", "setting", "ssb_frequency"],
        i_data_stream="I",
        q_data_stream="Q",
    ),
)
def pulsed_resonator_spec_after_pi_pulse(start, stop, step, n_reps):
    n_frqs = abs(round((stop - start) / step))

    with program() as qua_measurement:
        i_stream = declare_stream()
        q_stream = declare_stream()
        ssb_f_stream = declare_stream()
        rep_stream = declare_stream()

        I = declare(fixed)
        Q = declare(fixed)
        i = declare(int)
        ssb_f = declare(int)

        k_stream = declare_stream()
        k = declare(int)

        assign(k, 1)

        with for_(i, 0, i < n_reps, i + 1):
            with for_(ssb_f, start, ssb_f < stop, ssb_f + step):
                prepare()

                update_frequency(options.readout_element, ssb_f)
                align()
                measure_qubit(I, Q)

                save(I, i_stream)
                save(Q, q_stream)
                save(ssb_f, ssb_f_stream)
                save(i, rep_stream)
                save(k, k_stream)

        assign(k, 2)

        with for_(i, 0, i < n_reps, i + 1):
            with for_(ssb_f, start, ssb_f < stop, ssb_f + step):
                prepare()

                update_frequency(options.readout_element, ssb_f)

                play(f'{options.qubit_element}_pi_pulse', options.qubit_element)
                align()
                measure_qubit(I, Q)

                save(I, i_stream)
                save(Q, q_stream)
                save(ssb_f, ssb_f_stream)
                save(i, rep_stream)
                save(k, k_stream)

        with stream_processing():
            rep_stream.buffer(2, n_frqs).save_all("repetition")
            k_stream.buffer(2, n_frqs).save_all("setting")
            ssb_f_stream.buffer(2, n_frqs).save_all("ssb_frequency")
            i_stream.buffer(2, n_frqs).save_all("I")
            q_stream.buffer(2, n_frqs).save_all("Q")

    return qua_measurement


@RecordOPXdata(
    independent("repetition"),
    independent("ssb_frequency"),
    ComplexOPXData(
        "signal",
        depends_on=["repetition", "ssb_frequency"],
        i_data_stream="I",
        q_data_stream="Q",
    ),
)
def qubit_ssb_spec_saturation(start, stop, step, n_reps):
    """
    To perform this measurment the element qubit must have an operation called: 'long_drive'.
    """
    fvals = np.arange(start, stop, step)
    n_frqs = fvals.size

    with program() as qua_measurement:
        i_stream = declare_stream()
        q_stream = declare_stream()
        ssb_f_stream = declare_stream()
        rep_stream = declare_stream()

        I = declare(fixed)
        Q = declare(fixed)
        i = declare(int)
        ssb_f = declare(int)

        with for_(i, 0, i < n_reps, i + 1):
            with for_(ssb_f, start, ssb_f < stop, ssb_f + step):
                prepare()

                update_frequency(options.qubit_element, ssb_f)
                play(f"{options.qubit_element}_long_drive", options.qubit_element)
                align(options.qubit_element, options.readout_element)

                measure_qubit(I, Q)

                save(I, i_stream)
                save(Q, q_stream)
                save(ssb_f, ssb_f_stream)
                save(i, rep_stream)

        with stream_processing():
            rep_stream.buffer(n_frqs).save_all("repetition")
            ssb_f_stream.buffer(n_frqs).save_all("ssb_frequency")
            i_stream.buffer(n_frqs).save_all("I")
            q_stream.buffer(n_frqs).save_all("Q")

    return qua_measurement


@RecordOPXdata(
    independent("repetition"),
    independent("ssb_frequency"),
    ComplexOPXData(
        "signal",
        depends_on=["repetition", "ssb_frequency"],
        i_data_stream="I",
        q_data_stream="Q",
    ),
)
def qubit_ssb_spec_pi(start, stop, step, n_reps, duration=None, amplitude=None):
    """qubit pi pulse spectroscopy program.
    See qua documentation for some details.

    Parameters
    ----------
    start
        start ssb frequency [Hz]
    stop
        stop ssb frequency [Hz]
    step
        step size [Hz]
    n_reps
        number of repetitions
    duration
        length of the pipulse to stretch the default pipulse [4 ns].
        if not given, default pipulse is used.
    amplitude
        amplitude scaling factor.

    Returns
    -------
    The qua program.
    """

    n_frqs = abs(round((stop - start) / step))

    with program() as qua_measurement:
        i_stream = declare_stream()
        q_stream = declare_stream()
        ssb_f_stream = declare_stream()
        rep_stream = declare_stream()

        I = declare(fixed)
        Q = declare(fixed)
        i = declare(int)
        ssb_f = declare(int)

        with for_(i, 0, i < n_reps, i + 1):
            with for_(ssb_f, start, ssb_f < stop, ssb_f + step):
                prepare()

                update_frequency(options.qubit_element, ssb_f)
                if amplitude is None:
                    play(
                        f"{options.qubit_element}_pi_pulse",
                        options.qubit_element,
                        duration=duration,
                    )
                else:
                    play(
                        f"{options.qubit_element}_pi_pulse" * amp(amplitude),
                        options.qubit_element,
                        duration=duration,
                    )
                align(options.qubit_element, options.readout_element)

                measure_qubit(I, Q)

                save(I, i_stream)
                save(Q, q_stream)
                save(ssb_f, ssb_f_stream)
                save(i, rep_stream)

        with stream_processing():
            rep_stream.buffer(n_frqs).save_all("repetition")
            ssb_f_stream.buffer(n_frqs).save_all("ssb_frequency")
            i_stream.buffer(n_frqs).save_all("I")
            q_stream.buffer(n_frqs).save_all("Q")

    return qua_measurement


@RecordOPXdata(
    independent("repetition"),
    independent("amplitude"),
    ComplexOPXData(
        "signal",
        depends_on=["repetition", "amplitude"],
        i_data_stream="I",
        q_data_stream="Q",
    ),
)
def qubit_power_rabi(start, stop, step, n_reps):
    """
    To perform this measurement the qubit element must have an operation 'pi_pulse'.
    """
    n_amps = abs(round((stop - start) / step))

    with program() as qua_measurement:
        i_stream = declare_stream()
        q_stream = declare_stream()
        amp_stream = declare_stream()
        rep_stream = declare_stream()

        I = declare(fixed)
        Q = declare(fixed)
        i = declare(int)
        pulse_amp = declare(fixed)

        with for_(i, 0, i < n_reps, i + 1):
            with for_(pulse_amp, start, pulse_amp < stop, pulse_amp + step):
                prepare()

                play(
                    f"{options.qubit_element}_pi_pulse" * amp(pulse_amp),
                    options.qubit_element,
                )
                align(options.qubit_element, options.readout_element)

                measure_qubit(I, Q)

                save(I, i_stream)
                save(Q, q_stream)
                save(pulse_amp, amp_stream)
                save(i, rep_stream)

        with stream_processing():
            rep_stream.buffer(n_amps).save_all("repetition")
            amp_stream.buffer(n_amps).save_all("amplitude")
            i_stream.buffer(n_amps).save_all("I")
            q_stream.buffer(n_amps).save_all("Q")

    return qua_measurement

@RecordOPXdata(
    independent("repetition"),
    independent("qubit_IF"),
    independent("pulse_len"),
    ComplexOPXData(
        "signal",
        depends_on=["repetition", "qubit_IF", "pulse_len"],
        i_data_stream="I",
        q_data_stream="Q",
    ),
)
def qubit_time_rabi(t0, t1, nt, ssb0, ssb1, nssb, n_reps):
    """
    To perform this measurement the qubit element must have an operation 'pi_pulse'.
    """
    # start = start//4
    # step = (stop-start*4)/(npts-1)//4

    t0 = t0//4
    t1 = t1//4

    dt = (t1 - t0)//(nt - 1)

    if nssb == 1:
        dssb=0
    else:
        dssb = (ssb1 - ssb0)//(nssb-1)


    with program() as qua_measurement:
        i_stream = declare_stream()
        q_stream = declare_stream()
        ssb_stream = declare_stream()
        t_stream = declare_stream()
        rep_stream = declare_stream()

        I = declare(fixed)
        Q = declare(fixed)
        i = declare(int)
        
        i_t = declare(int)
        t = declare(int)
        t_ns = declare(int)
        
                
        i_ssb=declare(int)
        ssb = declare(int)


        with for_(i, 0, i < n_reps, i + 1):
                
            assign(ssb, ssb0-dssb)
            with for_(i_ssb, 0, i_ssb<nssb, i_ssb+1):
                assign(ssb, ssb+dssb)

                assign(t, t0-dt)
                with for_(i_t, 0, i_t < nt, i_t+1):
                    assign(t, t+dt)
                    assign(t_ns, t*4)  
                    
                    prepare()

                    with if_(nssb > 1):
                        update_frequency(f"{options.qubit_element}", ssb)

                    # play(
                    #     f"{options.qubit_element}_pi_pulse",
                    #     options.qubit_element, duration=t
                    # )

                    play(
                        f"{options.qubit_element}_square_pi",
                        options.qubit_element, duration=t
                    )

                    align(options.qubit_element, options.readout_element)

                    measure_qubit(I, Q)

                    save(I, i_stream)
                    save(Q, q_stream)
                    save(ssb, ssb_stream)
                    save(t_ns, t_stream)
                    save(i, rep_stream)

        with stream_processing():
            rep_stream.buffer(nssb, nt).save_all("repetition")
            ssb_stream.buffer(nssb, nt).save_all("qubit_IF")
            t_stream.buffer(nssb, nt).save_all("pulse_len")
            i_stream.buffer(nssb, nt).save_all("I")
            q_stream.buffer(nssb, nt).save_all("Q")

    return qua_measurement


@RecordOPXdata(
    independent("repetition"),
    independent("delay"),
    ComplexOPXData(
        "signal",
        depends_on=["repetition", "delay"],
        i_data_stream="I",
        q_data_stream="Q",
    ),
)
def qubit_T1(start, stop, npts, n_reps):
    """
    To perform this measurement the qubit element must have an operation 'pi_pulse'.
    """
    start = start//4
    step = (stop-start*4)/(npts-1)//4

    with program() as qua_measurement:
        i_stream = declare_stream()
        q_stream = declare_stream()
        delay_stream = declare_stream()
        rep_stream = declare_stream()

        I = declare(fixed)
        Q = declare(fixed)
        i = declare(int)
        j = declare(int)
        delay_ns = declare(int)

        with for_(i, 0, i < n_reps, i + 1):
            with for_(j, 0, j < npts, j + 1):
                prepare()

                play(f"{options.qubit_element}_pi_pulse", options.qubit_element)
                wait(start+j*step, options.qubit_element)
                align(options.qubit_element, options.readout_element)

                measure_qubit(I, Q)

                assign(delay_ns, start*4.0 + j*step*4.0)

                save(I, i_stream)
                save(Q, q_stream)
                save(delay_ns, delay_stream)
                save(i, rep_stream)

        with stream_processing():
            rep_stream.buffer(npts).save_all("repetition")
            delay_stream.buffer(npts).save_all("delay")
            i_stream.buffer(npts).save_all("I")
            q_stream.buffer(npts).save_all("Q")

    return qua_measurement


@RecordOPXdata(
    independent("repetition"),
    independent("delay"),
    ComplexOPXData(
        "signal",
        depends_on=["repetition", "delay"],
        i_data_stream="I",
        q_data_stream="Q",
    ),
)
def qubit_T2(
    start,
    stop,
    npts,
    n_reps,
    detuning_MHz=0,
    n_echos=0,
):
    """
    To perform this measurement the qubit element must have an operation 'pi_pulse'.
    """
    start = start//4
    step = (stop-start*4)/(npts-1)/(n_echos+1)//4

    with program() as qua_measurement:
        i_stream = declare_stream()
        q_stream = declare_stream()
        delay_stream = declare_stream()
        rep_stream = declare_stream()

        I = declare(fixed)
        Q = declare(fixed)
        i = declare(int)
        j = declare(int)
        delay_ns = declare(int)
        phase = declare(fixed, value=0.0)

        with for_(i, 0, i < n_reps, i + 1):
            with for_(j, 0, j < npts, j+1):
                prepare()

                play(
                    f"{options.qubit_element}_pi_pulse" * amp(0.5),
                    options.qubit_element,
                )
                wait(start+j*step, options.qubit_element)

                for k in range(n_echos):
                    play(f"{options.qubit_element}_pi_pulse", options.qubit_element)
                    wait(start+j*step, options.qubit_element)

                assign(phase, Cast.mul_fixed_by_int(detuning_MHz * 4.0 * 1e-3, j*step*(n_echos+1)))
                frame_rotation_2pi(phase, options.qubit_element)
                play(
                    f"{options.qubit_element}_pi_pulse" * amp(-0.5),
                    options.qubit_element,
                )

                align(options.qubit_element, options.readout_element)

                measure_qubit(I, Q)

                assign(delay_ns, start + j*step*4.0 * (n_echos+1))

                save(I, i_stream)
                save(Q, q_stream)
                save(delay_ns, delay_stream)
                save(i, rep_stream)

        with stream_processing():
            rep_stream.buffer(npts).save_all("repetition")
            delay_stream.buffer(npts).save_all("delay")
            i_stream.buffer(npts).save_all("I")
            q_stream.buffer(npts).save_all("Q")

    return qua_measurement

# readout calibration - measures e0, g0, and will be used to convert later measurement data into probabilities
@RecordOPXdata(
    independent("repetition"),
    independent("setting"),
    ComplexOPXData(
        "signal",
        depends_on=[
            "repetition",
            "setting",
        ],
        i_data_stream="i",
        q_data_stream="q",
    ),
)
def readout_calibration(n_reps):
    """This measurement is to prepare different states: e0 and g0 then measure.
    g1 is prepared by qubit-pi, then swap.

    We need this measurement to see how a photon in the bus affects the readout.
    """


    with program() as qua_measurement:
        # readout results
        i_stream = declare_stream()
        q_stream = declare_stream()
        i = declare(fixed)
        q = declare(fixed)

        # repetitions
        rep_stream = declare_stream()
        j = declare(int)

        # indexing the different settings
        # 1: e0; 2: g0;
        k_stream = declare_stream()
        k = declare(int)

        with for_(j, 0, j < n_reps, j + 1):

            # e0
            assign(k, 1)
            prepare()
            play(f"{options.qubit_element}_pi_pulse", options.qubit_element)
            wait(40//4)
            measure_qubit(i, q)

            save(i, i_stream)
            save(q, q_stream)
            save(j, rep_stream)
            save(k, k_stream)

            # g0
            assign(k, 2)
            prepare()
            wait(40//4)
            measure_qubit(i, q)

            save(i, i_stream)
            save(q, q_stream)
            save(j, rep_stream)
            save(k, k_stream)

        with stream_processing():
            rep_stream.buffer(2).save_all("repetition")
            k_stream.buffer(2).save_all("setting")
            i_stream.buffer(2).save_all("i")
            q_stream.buffer(2).save_all("q")

    return qua_measurement



# @RecordOPXdata(
#     independent("repetition"),
#     TimedOPXData("I", depends_on=["repetition"]),
#     TimedOPXData("Q", depends_on=["repetition"]),
# )
# def IQ_trace(n_reps=1000, pulse_amp=0.0, trigger=0):
#     """
#     To perform this measurement the qubit element must have an operation 'pi_pulse'.

#     This function saves an IQ trace vs time. A pulse_amp is applied after the repetition number indicated by trigger.

#     Parameters
#     ----------
#     n_reps:
#         The amount of times the measurement is going to be performed.
#     pulse_amp:
#         The amplitude of the pulse applied after the repetition indicated by trigger. In units of pi.
#     trigger:
#         Indicates what repetition to start applying pulse_amp. If all the repetitions should have the pulse_amp applied,
#         trigger should have its default value of 0.
#     """
#     _chunksize = int(options.chunksize_ns // 4)
#     _n_chunks = options.readout_pulse_length // (4 * _chunksize)
#     with program() as qua_measurement:
#         i_stream = declare_stream()
#         q_stream = declare_stream()
#         rep_stream = declare_stream()

#         I = declare(fixed, size=_n_chunks)
#         Q = declare(fixed, size=_n_chunks)
#         i = declare(int)
#         j = declare(int)

#         with for_(i, 0, i < n_reps, i + 1):
#             prepare()

#             with if_(i > trigger):
#                 play("pi_pulse" * amp(pulse_amp), "qubit")
#                 align("qubit", options.readout_element)

#             measure_qubit(I, Q)

#             with for_(j, 0, j < _n_chunks, j + 1):
#                 save(I[j], i_stream)
#                 save(Q[j], q_stream)

#             save(i, rep_stream)

#         with stream_processing():
#             rep_stream.save_all("repetition")

#             i_stream.buffer(_n_chunks).save_all("I")
#             q_stream.buffer(_n_chunks).save_all("Q")

#     return qua_measurement

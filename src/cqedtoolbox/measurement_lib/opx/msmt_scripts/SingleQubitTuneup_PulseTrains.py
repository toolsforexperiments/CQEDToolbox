#TODO: Work in progress - Michael Mollenhauer, Abdullah Irfan

from qm.qua import *

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

from labcore.setup_measurements import run_measurement, getp, param_from_name
from labcore.measurement.sweep import sweep_parameter 
from labcore.analysis.mpl import fit_and_plot_1d
from labcore.data.datadict_storage import load_as_xr
from labcore.analysis.fitfuncs.generic import Cosine
from cqedtoolbox.measurement_lib.opx.single_transmon import options

### adding this:

single_transmon_options = options


@RecordOPXdata(
    independent("repetition"),
    ComplexOPXData(
        "signal",
        depends_on=[
            "repetition"
        ],
        i_data_stream="i",
        q_data_stream="q",
    ),
)
def pi_pulse_train(train_len, n_reps):

    with program() as qua_measurement:
        # readout results
        i_stream = declare_stream()
        q_stream = declare_stream()
        i = declare(fixed)
        q = declare(fixed)

        # repetitions
        rep_stream = declare_stream()
        j = declare(int)

        k = declare(int)

        with for_(j, 0, j < n_reps, j + 1):

            prepare()

            with for_(k, 0, k<train_len, k+1):
                
                play(
                    f"{single_transmon_options.qubit_element}_pi_pulse",
                    single_transmon_options.qubit_element,)
            
                
            align(single_transmon_options.qubit_element, single_transmon_options.readout_element)
            measure_qubit(i, q)

            save(i, i_stream)
            save(q, q_stream)
            save(j, rep_stream)



        with stream_processing():
            rep_stream.buffer(1).save_all("repetition")
            i_stream.buffer(1).save_all("i")
            q_stream.buffer(1).save_all("q")

    return qua_measurement



@RecordOPXdata(
    independent("repetition"),
    ComplexOPXData(
        "signal",
        depends_on=[
            "repetition"
        ],
        i_data_stream="i",
        q_data_stream="q",
    ),
)
def pibytwo_pulse_train(train_len, n_reps):

    with program() as qua_measurement:
        # readout results
        i_stream = declare_stream()
        q_stream = declare_stream()
        i = declare(fixed)
        q = declare(fixed)

        # repetitions
        rep_stream = declare_stream()
        j = declare(int)

        k = declare(int)

        with for_(j, 0, j < n_reps, j + 1):

            prepare()

            with for_(k, 0, k<train_len, k+1):
                
                play(
                    f"{single_transmon_options.qubit_element}_pibytwo_pulse",
                    single_transmon_options.qubit_element,)
            
                
            align(single_transmon_options.qubit_element, single_transmon_options.readout_element)
            measure_qubit(i, q)

            save(i, i_stream)
            save(q, q_stream)
            save(j, rep_stream)



        with stream_processing():
            rep_stream.buffer(1).save_all("repetition")
            i_stream.buffer(1).save_all("i")
            q_stream.buffer(1).save_all("q")

    return qua_measurement



# def measure_after_pulse_train(qubit_name, train_len):
#     N = 10_000
#     msmt = pulse_train(train_len=train_len, n_reps=N, collector_options=dict(batchsize=N))
#     data_loc, _ = run_measurement(sweep=msmt, name=f"{single_transmon_options.qubit_element}_pulse_train")
#     return data_loc



def measure_after_pi_pulse_train_vs_amp(train_len, amp_range, amp_points):

    amp = param_from_name(f"{single_transmon_options.qubit_element}.pulses.pi.amp")

    N = 10_000
    msmt = pi_pulse_train(train_len=train_len, n_reps=N, collector_options=dict(batchsize=N))


    swp = sweep_parameter(amp, np.linspace(amp_range[0], amp_range[1], amp_points)) \
    @ msmt

    data_loc, _ = run_measurement(
        sweep=swp,
        name=f"{single_transmon_options.qubit_element}_pi_pulse_train_vs_amp"
    )

    return data_loc


def measure_after_pibytwo_pulse_train_vs_amp(train_len, amp_range, amp_points):

    amp = param_from_name(f"{single_transmon_options.qubit_element}.pulses.pi.pibytwo_amp")

    N = 10_000
    msmt = pibytwo_pulse_train(train_len=train_len, n_reps=N, collector_options=dict(batchsize=N))


    swp = sweep_parameter(amp, np.linspace(amp_range[0], amp_range[1], amp_points)) \
    @ msmt

    data_loc, _ = run_measurement(
        sweep=swp,
        name=f"{single_transmon_options.qubit_element}_pibytwo_pulse_train_vs_amp"
    )

    return data_loc

    

# if __name__ == "__main__":

#     qubit_name = 'qA'



#     #train_len = 8
#     #measure_after_pulse_train_vs_amp(train_len, amp_range, amp_points)


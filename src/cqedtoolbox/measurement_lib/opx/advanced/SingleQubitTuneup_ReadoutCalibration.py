#TODO: Work in progress - Michael Mollenhauer, Abdullah Irfan

import numpy as np

from qm.qua import (
    program,
    declare_stream,
    declare,
    fixed,
    for_,
    save,
    wait,
    stream_processing,
    play,
    align,
    amp,
    assign,
)

from labcore.measurement import independent
from labcore.instruments.opx.sweep import (
    RecordOPXdata,
    ComplexOPXData,
    TimedOPXData,
)

from cqedtoolbox.setup_measurements import run_measurement, getp, param_from_name
from labcore.measurement.sweep import sweep_parameter 
from labcore.analysis.mpl import fit_and_plot_1d, plot_fit_1d
from labcore.data.datadict_storage import load_as_xr
from cqedtoolbox.measurement_lib.opx.single_transmon import options as single_transmon_options, measure_qubit, prepare
from labcore.analysis.fitfuncs.generic import Cosine


delta_t_RO = 20

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
def readout_ground_and_excited(n_reps):
    """This measurement is to readout the qubit with and without a pi-pulse.

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
        # 1: e0; 2: g0; 3: g1
        k_stream = declare_stream()
        k = declare(int)

        with for_(j, 0, j < n_reps, j + 1):
            # excited
            assign(k, 1)
            prepare()
            play(
                f"{single_transmon_options.qubit_element}_pi_pulse",
                single_transmon_options.qubit_element,)
            wait(delta_t_RO)
            align()
            measure_qubit(i, q)

            save(i, i_stream)
            save(q, q_stream)
            save(j, rep_stream)
            save(k, k_stream)

            # ground
            assign(k, 2)
            prepare()
            wait(delta_t_RO)
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

# FIXME: There should be a way of adding this function instead of importing it from my_experiment_setup
# def measure_readout_ground_and_excited(qubit_name):
#     setup_qubit_measurement_defaults(repetition_delay=500_000, qubit_name=qubit_name)
#     N = 5000
#     msmt = readout_ground_and_excited(n_reps=N, collector_options=dict(batchsize=N))
#     data_loc, _ = run_measurement(sweep=msmt, name=f"{qubit_name}_ROcal")
#     return data_loc



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
def readout_superposition(n_reps):

    with program() as qua_measurement:
        # readout results
        i_stream = declare_stream()
        q_stream = declare_stream()
        i = declare(fixed)
        q = declare(fixed)

        # repetitions
        rep_stream = declare_stream()
        j = declare(int)


        with for_(j, 0, j < n_reps, j + 1):

            prepare()
            play(
                f"{single_transmon_options.qubit_element}_pi_pulse" * amp(0.5),
                single_transmon_options.qubit_element,)
            wait(delta_t_RO)
            align()
            measure_qubit(i, q)

            save(i, i_stream)
            save(q, q_stream)
            save(j, rep_stream)



        with stream_processing():
            rep_stream.buffer(1).save_all("repetition")
            i_stream.buffer(1).save_all("i")
            q_stream.buffer(1).save_all("q")

    return qua_measurement


def sweeping_RO_params(qubit_name):

    RO_len = param_from_name(f"{qubit_name}.readout.short.len")
    RO_amp = param_from_name(f"{qubit_name}.readout.short.amp")
    RO_IF = param_from_name(f"{qubit_name}.readout.IF")


    #setup_qubit_measurement_defaults(repetition_delay=800_000, qubit_name=qubit_name)
    N = 1000
    msmt = readout_ground_and_excited(n_reps=N, collector_options=dict(batchsize=N))

    # swp = sweep_parameter(RO_len, np.linspace(1800, 2800, 11)) \
    #     @ sweep_parameter(RO_amp, np.linspace(0.01, 0.04, 11)) \
    #     @ msmt

    swp = sweep_parameter(RO_IF, np.linspace(48e6, 52e6, 6)) \
    @ sweep_parameter(RO_amp, np.linspace(0.05, 0.12, 6)) \
    @ msmt

    data_loc, _ = run_measurement(
        sweep=swp,
        name=f"{qubit_name}_ROcal_sweeping_RO_params"
    )

    return data_loc

def measure_readout_superposition(qubit_name):
    #setup_qubit_measurement_defaults(repetition_delay=500_000, qubit_name=qubit_name)
    N = 10_000
    msmt = readout_superposition(n_reps=N, collector_options=dict(batchsize=N))
    data_loc, _ = run_measurement(sweep=msmt, name=f"{qubit_name}_RO_superposition")
    return data_loc

def measure_readout_ground_and_excited(qubit_name):
    #setup_qubit_measurement_defaults(repetition_delay=500_000, qubit_name=qubit_name)
    N = 10_000
    msmt = readout_ground_and_excited(n_reps=N, collector_options=dict(batchsize=N))
    data_loc, _ = run_measurement(sweep=msmt, name=f"{qubit_name}_RO_ground_and_excited")
    return data_loc

# TODO: Clean this up.
# if __name__ == "__main__":
#
#     qubit = getp('active.qubit')
#     # setup_single_qubit_measurement_defaults()
#
#
#     measure_readout_ground_and_excited(qubit)
#
#     loc = sweeping_RO_params(qubit)




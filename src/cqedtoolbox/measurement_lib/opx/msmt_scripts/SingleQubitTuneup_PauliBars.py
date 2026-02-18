
#TODO: Work in progress - Michael Mollenhauer, Abdullah Irfan

from qm.qua import *

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
from labcore.analysis.fitfuncs.generic import Cosine
from labcore.data.datadict_storage import load_as_xr
from cqedtoolbox.measurement_lib.opx.single_transmon import options


### adding this:

single_transmon_options = options


#pibytwo_amp = 0.500891301762813
pibytwo_amp = 0.5

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
def readoutXYZ(n_reps):

    with program() as qua_measurement:
        i_stream = declare_stream()
        q_stream = declare_stream()
        i = declare(fixed)
        q = declare(fixed)
        rep_stream = declare_stream()
        j = declare(int)
        k_stream = declare_stream()
        k = declare(int)

        with for_(j, 0, j < n_reps, j + 1):
            # measure Z
            assign(k, 1)
            prepare()
            
            reset_frame(f"{single_transmon_options.qubit_element}")

            # prepare_state
            #frame_rotation_2pi(0.25, f"{single_transmon_options.qubit_element}")
            play(f'{single_transmon_options.qubit_element}_pi_pulse' * amp(pibytwo_amp), f"{single_transmon_options.qubit_element}")
            #frame_rotation_2pi(-0.25, f"{single_transmon_options.qubit_element}")
            wait(32)

            align()
            measure_qubit(i, q)

            save(i, i_stream)
            save(q, q_stream)
            save(j, rep_stream)
            save(k, k_stream)

            # measure X
            assign(k, 2)
            prepare()
            reset_frame(f"{single_transmon_options.qubit_element}")
            #prepare_state
            #frame_rotation_2pi(0.25, f"{single_transmon_options.qubit_element}")
            play(f'{single_transmon_options.qubit_element}_pi_pulse' * amp(pibytwo_amp), f"{single_transmon_options.qubit_element}")
            #frame_rotation_2pi(-0.25, f"{single_transmon_options.qubit_element}")
            wait(32)


            play(f'{single_transmon_options.qubit_element}_pi_pulse' * amp(pibytwo_amp), f"{single_transmon_options.qubit_element}")
            align()
            measure_qubit(i, q)

            save(i, i_stream)
            save(q, q_stream)
            save(j, rep_stream)
            save(k, k_stream)

            # measure Y
            assign(k, 3)
            prepare()
            reset_frame(f"{single_transmon_options.qubit_element}")

            #prepare_state
            #frame_rotation_2pi(0.25, f"{single_transmon_options.qubit_element}")
            play(f'{single_transmon_options.qubit_element}_pi_pulse' * amp(pibytwo_amp), f"{single_transmon_options.qubit_element}")
            #frame_rotation_2pi(-0.25, f"{single_transmon_options.qubit_element}")
            wait(32)

            frame_rotation_2pi(0.25, f"{single_transmon_options.qubit_element}")
            play(f'{single_transmon_options.qubit_element}_pi_pulse' * amp(pibytwo_amp), f"{single_transmon_options.qubit_element}")
            frame_rotation_2pi(-0.25, f"{single_transmon_options.qubit_element}")
            align()
            measure_qubit(i, q)


            save(i, i_stream)
            save(q, q_stream)
            save(j, rep_stream)
            save(k, k_stream)


        with stream_processing():
            rep_stream.buffer(3).save_all("repetition")
            k_stream.buffer(3).save_all("setting")
            i_stream.buffer(3).save_all("i")
            q_stream.buffer(3).save_all("q")

    return qua_measurement



def measure_readoutXYZ(n_reps):

    msmt = readoutXYZ(n_reps=n_reps, collector_options=dict(batchsize=n_reps))
    data_loc, _ = run_measurement(sweep=msmt, name=f"{single_transmon_options.qubit_element}_measureXYZ")
    return data_loc

    

if __name__ == "__main__":

    qubit_name = 'qA'
    n_reps = 10_000

    measure_readoutXYZ(n_reps)




    
  
    




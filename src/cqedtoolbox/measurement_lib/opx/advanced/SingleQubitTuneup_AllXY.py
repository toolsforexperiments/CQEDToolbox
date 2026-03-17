
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
    wait,
    stream_processing,
    play,
    align,
    amp,
    assign,
    frame_rotation_2pi,
    if_,
)

from labcore.measurement import independent
from labcore.measurement.sweep import sweep_parameter

from cqedtoolbox.instruments.opx.sweep import (
    RecordOPXdata,
    ComplexOPXData,
)

from cqedtoolbox.setup_measurements import run_measurement, getp, param_from_name
from cqedtoolbox.measurement_lib.opx.single_transmon import options as single_transmon_options, measure_qubit, prepare

angles1 = [0.0, 0.0, 0.25, 0.0, 0.25,
           0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.0, 0.25, 0.25,
           0.0, 0.25, 0.0, 0.25]
angles2 = [0.0, 0.0, 0.25, 0.25, 0.0,
           0.0, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.0, 0.0, 0.25, 0.25,
           0.0, 0.0, 0.0, 0.25]
amps1 = [0.0, 1.0, 1.0, 1.0, 1.0,
         0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 0.5, 1.0, 0.5, 1.0,
         1.0, 1.0, 0.5, 0.5]
amps2 = [0.0, 1.0, 1.0, 1.0, 1.0,
         0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 1.0, 0.5,
         0.0, 0.0, 0.5, 0.5]





angles1 = [v for p in zip(angles1, angles1) for v in p]
angles2 = [v for p in zip(angles2, angles2) for v in p]
amps1 = [v for p in zip(amps1, amps1) for v in p]
amps2 = [v for p in zip(amps2, amps2) for v in p]

allxy_seq = [
    'II', 'XX', 'YY', 'XY', 'YX', 
    'xI', 'yI', 'xy', 'yx', 'xY', 'yX', 'Xy', 'Yx', 'xX', 'Xx', 'yY', 'Yy',
    'XI', 'YI', 'xx', 'yy',
]
allxy_seq = [v for p in zip(allxy_seq, ['']*21) for v in p]



@RecordOPXdata(
    independent('repetition'),
    independent('seq'),
    ComplexOPXData('signal',
                   depends_on=['repetition', 'seq'],
                   i_data_stream='I', q_data_stream='Q')
)
def allXY(n_reps):
    n_pts = len(angles1)

    with program() as qua_measurement:
        i_stream = declare_stream()
        q_stream = declare_stream()
        rep_stream = declare_stream()
        seq_stream = declare_stream()

        I = declare(fixed)
        Q = declare(fixed)
        i = declare(int)
        seq_num = declare(int, value=0)
        angle1 = declare(fixed)
        angle2 = declare(fixed)
        amp1 = declare(fixed)
        amp2 = declare(fixed)

        with for_(i, 0, i < n_reps, i + 1):
            prepare()

            assign(seq_num, 0)
            with for_each_(
                    (angle1, angle2, amp1, amp2),
                    (angles1, angles2, amps1, amps2),
            ):
                prepare()
                
                reset_frame(f"{single_transmon_options.qubit_element}")
                align(f"{single_transmon_options.qubit_element}", single_transmon_options.readout_element)
                frame_rotation_2pi(angle1, f"{single_transmon_options.qubit_element}")

                with if_(amp1==1):
                    play(f'{single_transmon_options.qubit_element}_pi_pulse', f"{single_transmon_options.qubit_element}")

                with elif_(amp1==0.5):
                    play(f'{single_transmon_options.qubit_element}_pibytwo_pulse', f"{single_transmon_options.qubit_element}")
                
                with elif_(amp1==0):
                    play(f'{single_transmon_options.qubit_element}_pi_pulse' * amp(0), f"{single_transmon_options.qubit_element}")

                wait(4)
                frame_rotation_2pi(-angle1 + angle2, f"{single_transmon_options.qubit_element}")

                with if_(amp2==1):
                    play(f'{single_transmon_options.qubit_element}_pi_pulse', f"{single_transmon_options.qubit_element}")

                with elif_(amp2==0.5):
                    play(f'{single_transmon_options.qubit_element}_pibytwo_pulse', f"{single_transmon_options.qubit_element}")
                
                with elif_(amp2==0):
                    play(f'{single_transmon_options.qubit_element}_pi_pulse' * amp(0), f"{single_transmon_options.qubit_element}")



                wait(4)
                frame_rotation_2pi(-angle2, f"{single_transmon_options.qubit_element}")
                align(f"{single_transmon_options.qubit_element}", single_transmon_options.readout_element)
                wait(40)
                measure_qubit(I, Q)

                save(I, i_stream)
                save(Q, q_stream)
                save(seq_num, seq_stream)
                save(i, rep_stream)
                
                assign(seq_num, seq_num + 1)

        with stream_processing():
            rep_stream.buffer(n_pts).save_all('repetition')
            seq_stream.buffer(n_pts).save_all('seq')
            i_stream.buffer(n_pts).save_all('I')
            q_stream.buffer(n_pts).save_all('Q')

    return qua_measurement


@RecordOPXdata(
    independent('repetition'),
    independent('seq'),
    ComplexOPXData('signal',
                   depends_on=['repetition', 'seq'],
                   i_data_stream='I', q_data_stream='Q')
)
def not_allXY(n_reps, range):
    n_pts = len(angles1[range[0]:range[1]])

    with program() as qua_measurement:
        i_stream = declare_stream()
        q_stream = declare_stream()
        rep_stream = declare_stream()
        seq_stream = declare_stream()

        I = declare(fixed)
        Q = declare(fixed)
        i = declare(int)
        seq_num = declare(int, value=0)
        angle1 = declare(fixed)
        angle2 = declare(fixed)
        amp1 = declare(fixed)
        amp2 = declare(fixed)

        with for_(i, 0, i < n_reps, i + 1):
            prepare()

            assign(seq_num, 0)
            with for_each_(
                    (angle1, angle2, amp1, amp2),
                    (angles1[range[0]:range[1]], angles2[range[0]:range[1]], amps1[range[0]:range[1]], amps2[range[0]:range[1]]),
            ):
                prepare()
                
                reset_frame(f"{single_transmon_options.qubit_element}")
                align(f"{single_transmon_options.qubit_element}", single_transmon_options.readout_element)
                frame_rotation_2pi(angle1, f"{single_transmon_options.qubit_element}")
                play(f'{single_transmon_options.qubit_element}_pi_pulse' * amp(amp1), f"{single_transmon_options.qubit_element}")
                frame_rotation_2pi(-angle1 + angle2, f"{single_transmon_options.qubit_element}")
                play(f'{single_transmon_options.qubit_element}_pi_pulse' * amp(amp2), f"{single_transmon_options.qubit_element}")
                frame_rotation_2pi(-angle2, f"{single_transmon_options.qubit_element}")
                align(f"{single_transmon_options.qubit_element}", single_transmon_options.readout_element)

                measure_qubit(I, Q)

                save(I, i_stream)
                save(Q, q_stream)
                save(seq_num, seq_stream)
                save(i, rep_stream)
                
                assign(seq_num, seq_num + 1)

        with stream_processing():
            rep_stream.buffer(n_pts).save_all('repetition')
            seq_stream.buffer(n_pts).save_all('seq')
            i_stream.buffer(n_pts).save_all('I')
            q_stream.buffer(n_pts).save_all('Q')

    return qua_measurement



def measure_allXY(qubit_name, n_reps):

    measurement = allXY(
        n_reps=n_reps,
        collector_options=dict(batchsize=1000)
    )

    data_loc, _ = run_measurement(sweep=measurement, name=f'AllXY')
    return data_loc

def measure_not_allXY(qubit_name, n_reps, range):

    measurement = not_allXY(
        n_reps=n_reps,
        range=range,
        collector_options=dict(batchsize=100)
    )

    data_loc, _ = run_measurement(sweep=measurement, name=f'not_AllXY')
    return data_loc


def measure_allXY_vs_detuning(qubit_name, n_reps, f_range=0.01e6, nf=11):

    detuning = param_from_name(f"{qubit_name}.IF")
    fc = param_from_name(f"{qubit_name}.IF")()

    measurement = allXY(
        n_reps=n_reps,
        collector_options=dict(batchsize=10_000)
    )

    swp = sweep_parameter(detuning, np.linspace(fc - f_range, fc + f_range, nf)) \
        @ measurement

    data_loc, _ = run_measurement(sweep=swp, name=f'AllXY_vs_det')
    return data_loc


def measure_allXY_vs_amp(qubit_name, n_reps, a_start, a_stop, n_a):

    pi_amp = param_from_name(f"{single_transmon_options.qubit_element}.pulses.pi.amp")
    a = param_from_name(f"{single_transmon_options.qubit_element}.pulses.pi.amp")()

    measurement = allXY(
        n_reps=n_reps,
        collector_options=dict(batchsize=n_reps)
    )

    swp = sweep_parameter(pi_amp, np.linspace(a_start, a_stop, n_a)) \
        @ measurement

    data_loc, _ = run_measurement(sweep=swp, name=f'AllXY_vs_amp')
    return data_loc



def measure_allXY_vs_DRAG(qubit_name, n_reps, drag_start=-2, drag_stop=2, drag_points=11):

    DRAG_mult = param_from_name(f"{single_transmon_options.qubit_element}.pulses.pi.drag_multiplier")

    measurement = allXY(
        n_reps=n_reps,
        collector_options=dict(batchsize=n_reps)
    )

    swp = sweep_parameter(DRAG_mult, np.linspace(drag_start, drag_stop, drag_points)) \
        @ measurement

    data_loc, _ = run_measurement(sweep=swp, name=f'AllXY_vs_DRAG')
    return data_loc


def measure_not_allXY_vs_DRAG(qubit_name, n_reps, range, drag_range, drag_points):

    DRAG_mult = param_from_name(f"{single_transmon_options.qubit_element}.pulses.pi.drag_multiplier")

    measurement = not_allXY(
        n_reps=n_reps,
        range=range,
        collector_options=dict(batchsize=20_000)
    )

    swp = sweep_parameter(DRAG_mult, np.linspace(drag_range[0], drag_range[1], drag_points)) \
        @ measurement

    data_loc, _ = run_measurement(sweep=swp, name=f'not_AllXY_vs_DRAG')
    return data_loc


def measure_not_allXY_vs_detuning_and_amp(qubit_name, n_reps, range):

    detuning = param_from_name(f"{single_transmon_options.qubit_element}.IF")
    fc = getp(f"{single_transmon_options.qubit_element}.IF")

    pi_amp = param_from_name(f"{single_transmon_options.qubit_element}.pulses.pi.amp")
    a = getp(f"{single_transmon_options.qubit_element}.pulses.pi.amp")

    measurement = not_allXY(
        n_reps=n_reps,
        range=range,
        collector_options=dict(batchsize=5_000)
    )

    swp = sweep_parameter(detuning, np.linspace(fc-0.08e6, fc+0.08e6, 4)) \
        @sweep_parameter(pi_amp, np.linspace(a-0.004, a+0.004, 4)) \
        @ measurement

    data_loc, _ = run_measurement(sweep=swp, name=f'not_AllXY_vs_detuning_and_amp')
    return data_loc


# TODO: Clean this up
# if __name__ == "__main__":
#
#     qubit_name = 'qC'
#     n_reps = 5_000
#
#     #measure_not_allXY(qubit_name=qubit_name, n_reps=n_reps, range=[18, 19])
#
#     #measure_allXY(qubit_name=qubit_name, n_reps=n_reps)
#
#     #measure_not_allXY_vs_DRAG(qubit_name=qubit_name, n_reps=n_reps, range=[19, 21])
#
#     #measure_allXY_vs_detuning(qubit_name=qubit_name, n_reps=n_reps)
#
#     #measure_allXY_vs_amp(qubit_name=qubit_name, n_reps=n_reps)
#
#     #measure_allXY_vs_DRAG(qubit_name=qubit_name, n_reps=n_reps)
#
#     measure_not_allXY_vs_detuning_and_amp(qubit_name=qubit_name, n_reps=n_reps, range=[10, 33])
    
    
  
    





#TODO: Work in progress - Michael Mollenhauer, Abdullah Irfan

import numpy as np
from qm.qua import *
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table

from labcore.measurement import independent
from cqedtoolbox.instruments.opx.sweep import (
    RecordOPXdata,
    ComplexOPXData,
)

from cqedtoolbox.setup_measurements import run_measurement
from cqedtoolbox.measurement_lib.opx import single_transmon

single_transmon_options = single_transmon.options

def get_interleaved_gate(gate_index):

    # index of the gate to interleave from the play_sequence() function defined below
    # Correspondence table:
    #  0: identity |  1: x180 |  2: y180
    # 12: x90      | 13: -x90 | 14: y90 | 15: -y90 |

    match gate_index:
        case 0:
            return 0
        case 1:
            return 1
        case 2:
            return 4
        case 12:
            return 2
        case 13:
            return 3
        case 14:
            return 5
        case 15:
            return 6
        


def power_law(power, a, b, p):
    return a * (p**power) + b


# generates the sequence for performing randomized benchmarking
# TODO: understand the math
def generate_sequence(interleaved_gate_index, seed, inv_gates, max_circuit_depth):
    cayley = declare(int, value=c1_table.flatten().tolist())
    inv_list = declare(int, value=inv_gates)
    current_state = declare(int)
    step = declare(int)
    sequence = declare(int, size=2 * max_circuit_depth + 1)
    inv_gate = declare(int, size=2 * max_circuit_depth + 1)
    i = declare(int)
    rand = Random(seed=seed)

    assign(current_state, 0)
    with for_(i, 0, i < 2 * max_circuit_depth, i + 2):
        assign(step, rand.rand_int(24))
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i], step)
        assign(inv_gate[i], inv_list[current_state])
        # interleaved gate
        assign(step, interleaved_gate_index)
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i + 1], step)
        assign(inv_gate[i + 1], inv_list[current_state])

    return sequence, inv_gate


# given a list of indexes, play the corresponding pulses linked
# to those indicies
# TODO: make an element for each of these pulses?
def play_sequence(qubit_name, sequence_list, depth):
    i = declare(int)
    with for_(i, 0, i <= depth, i + 1):
        with switch_(sequence_list[i], unsafe=True):
            with case_(0):
                wait((64//4), f"{qubit_name}")
            with case_(1):
                #play_pulse("x180", qubit_name)
                play(f'{qubit_name}_pi_pulse', f"{qubit_name}")
            with case_(2):
                #play_pulse("y180", qubit_name)
                frame_rotation_2pi(0.25, f"{qubit_name}")
                play(f'{qubit_name}_pi_pulse', f"{qubit_name}")
                frame_rotation_2pi(-0.25, f"{qubit_name}")
            with case_(3):
                #play_pulse("y180", qubit_name)
                frame_rotation_2pi(0.25, f"{qubit_name}")
                play(f'{qubit_name}_pi_pulse', f"{qubit_name}")
                frame_rotation_2pi(-0.25, f"{qubit_name}")
                #play_pulse("x180", qubit_name)
                play(f'{qubit_name}_pi_pulse', f"{qubit_name}")
            with case_(4):
                #play_pulse("x90", qubit_name)
                play(f'{qubit_name}_pibytwo_pulse', f"{qubit_name}")
                #play_pulse("y90", qubit_name)
                frame_rotation_2pi(0.25, f"{qubit_name}")
                play(f'{qubit_name}_pibytwo_pulse', f"{qubit_name}")
                frame_rotation_2pi(-0.25, f"{qubit_name}")
            with case_(5):
                #play_pulse("x90", qubit_name)
                play(f'{qubit_name}_pibytwo_pulse', f"{qubit_name}")
                #play_pulse("-y90", qubit_name)
                frame_rotation_2pi(0.25, f"{qubit_name}")
                play(f'{qubit_name}_pibytwo_pulse' * amp(-1), f"{qubit_name}")
                frame_rotation_2pi(-0.25, f"{qubit_name}")
            with case_(6):
                #play_pulse("-x90", qubit_name)
                play(f'{qubit_name}_pibytwo_pulse' * amp(-1), f"{qubit_name}")
                #play_pulse("y90", qubit_name)
                frame_rotation_2pi(0.25, f"{qubit_name}")
                play(f'{qubit_name}_pibytwo_pulse', f"{qubit_name}")
                frame_rotation_2pi(-0.25, f"{qubit_name}")
            with case_(7):
                #play_pulse("-x90", qubit_name)
                play(f'{qubit_name}_pibytwo_pulse' * amp(-1), f"{qubit_name}")
                #play_pulse("-y90", qubit_name)
                frame_rotation_2pi(0.25, f"{qubit_name}")
                play(f'{qubit_name}_pibytwo_pulse' * amp(-1), f"{qubit_name}")
                frame_rotation_2pi(-0.25, f"{qubit_name}")
            with case_(8):
                #play_pulse("y90", qubit_name)
                frame_rotation_2pi(0.25, f"{qubit_name}")
                play(f'{qubit_name}_pibytwo_pulse', f"{qubit_name}")
                frame_rotation_2pi(-0.25, f"{qubit_name}")
                #play_pulse("x90", qubit_name)
                play(f'{qubit_name}_pibytwo_pulse', f"{qubit_name}")
            with case_(9):
                #play_pulse("y90", qubit_name)
                frame_rotation_2pi(0.25, f"{qubit_name}")
                play(f'{qubit_name}_pibytwo_pulse', f"{qubit_name}")
                frame_rotation_2pi(-0.25, f"{qubit_name}")
                #play_pulse("-x90", qubit_name)
                play(f'{qubit_name}_pibytwo_pulse' * amp(-1), f"{qubit_name}")
            with case_(10):
                #play_pulse("-y90", qubit_name)
                frame_rotation_2pi(0.25, f"{qubit_name}")
                play(f'{qubit_name}_pibytwo_pulse' * amp(-1), f"{qubit_name}")
                frame_rotation_2pi(-0.25, f"{qubit_name}")
                #play_pulse("x90", qubit_name)
                play(f'{qubit_name}_pibytwo_pulse', f"{qubit_name}")
            with case_(11):
                #play_pulse("-y90", qubit_name)
                frame_rotation_2pi(0.25, f"{qubit_name}")
                play(f'{qubit_name}_pibytwo_pulse' * amp(-1), f"{qubit_name}")
                frame_rotation_2pi(-0.25, f"{qubit_name}")
                #play_pulse("-x90", qubit_name)
                play(f'{qubit_name}_pibytwo_pulse' * amp(-1), f"{qubit_name}")
            with case_(12):
                #play_pulse("x90", qubit_name)
                play(f'{qubit_name}_pibytwo_pulse', f"{qubit_name}")
            with case_(13):
                #play_pulse("-x90", qubit_name)
                play(f'{qubit_name}_pibytwo_pulse' * amp(-1), f"{qubit_name}")
            with case_(14):
                #play_pulse("y90", qubit_name)
                frame_rotation_2pi(0.25, f"{qubit_name}")
                play(f'{qubit_name}_pibytwo_pulse', f"{qubit_name}")
                frame_rotation_2pi(-0.25, f"{qubit_name}")
            with case_(15):
                #play_pulse("-y90", qubit_name)
                frame_rotation_2pi(0.25, f"{qubit_name}")
                play(f'{qubit_name}_pibytwo_pulse' * amp(-1), f"{qubit_name}")
                frame_rotation_2pi(-0.25, f"{qubit_name}")
            with case_(16):
                #play_pulse("-x90", qubit_name)
                play(f'{qubit_name}_pibytwo_pulse' * amp(-1), f"{qubit_name}")
                #play_pulse("y90", qubit_name)
                frame_rotation_2pi(0.25, f"{qubit_name}")
                play(f'{qubit_name}_pibytwo_pulse', f"{qubit_name}")
                frame_rotation_2pi(-0.25, f"{qubit_name}")
                #play_pulse("x90", qubit_name)
                play(f'{qubit_name}_pibytwo_pulse', f"{qubit_name}")
            with case_(17):
                #play_pulse("-x90", qubit_name)
                play(f'{qubit_name}_pibytwo_pulse' * amp(-1), f"{qubit_name}")
                #play_pulse("-y90", qubit_name)
                frame_rotation_2pi(0.25, f"{qubit_name}")
                play(f'{qubit_name}_pibytwo_pulse' * amp(-1), f"{qubit_name}")
                frame_rotation_2pi(-0.25, f"{qubit_name}")
                #play_pulse("x90", qubit_name)
                play(f'{qubit_name}_pibytwo_pulse', f"{qubit_name}")
            with case_(18):
                #play_pulse("x180", qubit_name)
                play(f'{qubit_name}_pi_pulse', f"{qubit_name}")
                #play_pulse("y90", qubit_name)
                frame_rotation_2pi(0.25, f"{qubit_name}")
                play(f'{qubit_name}_pibytwo_pulse', f"{qubit_name}")
                frame_rotation_2pi(-0.25, f"{qubit_name}")
            with case_(19):
                #play_pulse("x180", qubit_name)
                play(f'{qubit_name}_pi_pulse', f"{qubit_name}")
                #play_pulse("-y90", qubit_name)
                frame_rotation_2pi(0.25, f"{qubit_name}")
                play(f'{qubit_name}_pibytwo_pulse' * amp(-1), f"{qubit_name}")
                frame_rotation_2pi(-0.25, f"{qubit_name}")
            with case_(20):
                #play_pulse("y180", qubit_name)
                frame_rotation_2pi(0.25, f"{qubit_name}")
                play(f'{qubit_name}_pi_pulse', f"{qubit_name}")
                frame_rotation_2pi(-0.25, f"{qubit_name}")
                #play_pulse("x90", qubit_name)
                play(f'{qubit_name}_pibytwo_pulse', f"{qubit_name}")
            with case_(21):
                #play_pulse("y180", qubit_name)
                frame_rotation_2pi(0.25, f"{qubit_name}")
                play(f'{qubit_name}_pi_pulse', f"{qubit_name}")
                frame_rotation_2pi(-0.25, f"{qubit_name}")
                #play_pulse("-x90", qubit_name)
                play(f'{qubit_name}_pibytwo_pulse' * amp(-1), f"{qubit_name}")
            with case_(22):
                #play_pulse("x90", qubit_name)
                play(f'{qubit_name}_pibytwo_pulse', f"{qubit_name}")
                #play_pulse("y90", qubit_name)
                frame_rotation_2pi(0.25, f"{qubit_name}")
                play(f'{qubit_name}_pibytwo_pulse', f"{qubit_name}")
                frame_rotation_2pi(-0.25, f"{qubit_name}")
                #play_pulse("x90", qubit_name)
                play(f'{qubit_name}_pibytwo_pulse', f"{qubit_name}")
            with case_(23):
                #play_pulse("-x90", qubit_name)
                play(f'{qubit_name}_pibytwo_pulse' * amp(-1), f"{qubit_name}")
                #play_pulse("y90", qubit_name)
                frame_rotation_2pi(0.25, f"{qubit_name}")
                play(f'{qubit_name}_pibytwo_pulse', f"{qubit_name}")
                frame_rotation_2pi(-0.25, f"{qubit_name}")
                #play_pulse("-x90", qubit_name)
                play(f'{qubit_name}_pibytwo_pulse' * amp(-1), f"{qubit_name}")



# performs the randomized benchmark protocol
@RecordOPXdata(
    independent('repetition'),
    independent('iteration'),
    independent('depth'),
    independent('sequence_average'),
    ComplexOPXData(
        "signal",
        depends_on=['repetition', 'iteration','depth', 'sequence_average'],
        i_data_stream="I",
        q_data_stream="Q",
    ),
)            
def randomized_benchmarking(qubit_name, num_of_sequences, n_avg, max_circuit_depth, delta_clifford, 
                            seed, inv_gates, interleaved_gate_index, n_reps=1
):

    assert (max_circuit_depth / delta_clifford).is_integer(), "max_circuit_depth / delta_clifford must be an integer."
    n_depth = max_circuit_depth / delta_clifford + 1

    with program() as qua_measurement:

        depth = declare(int)  # QUA variable for the varying depth
        depth_target = declare(int)  # QUA variable for the current depth (changes in steps of delta_clifford)
        # QUA variable to store the last Clifford gate of the current sequence which is replaced by the recovery gate
        saved_gate = declare(int)
        j = declare(int)
        m = declare(int)  # QUA variable for the loop over random sequences
        n = declare(int)  # QUA variable for the averaging loop
        I = declare(fixed)  # QUA variable for the 'I' quadrature
        Q = declare(fixed)  # QUA variable for the 'Q' quadrature
        state = declare(bool)  # QUA variable for state discrimination
        # The relevant streams
        rep_stream = declare_stream()
        iter_stream = declare_stream()
        depth_stream = declare_stream()
        in_avg_stream = declare_stream()
        i_stream = declare_stream()
        q_stream = declare_stream()


        with for_(j, 0, j < n_reps, j+1):

            with for_(m, 0, m < num_of_sequences, m + 1):  # QUA for_ loop over the random sequences
                sequence_list, inv_gate_list = generate_sequence(interleaved_gate_index=interleaved_gate_index, 
                                                                seed=seed, inv_gates=inv_gates, max_circuit_depth=max_circuit_depth)  # Generate the random sequence of length max_circuit_depth

                assign(depth_target, 0) 
                with for_(depth, 1, depth <= 2 * max_circuit_depth, depth + 1): 

                    assign(saved_gate, sequence_list[depth])
                    assign(sequence_list[depth], inv_gate_list[depth - 1])
        
                    with if_((depth == 2) | (depth == depth_target)):
                        with for_(n, 0, n < n_avg, n + 1):  
        
                            single_transmon.prepare()
                            # with strict_timing_():
                            play_sequence(qubit_name, sequence_list, depth)
                            align()
                            single_transmon.measure_qubit(I, Q)

                            save(I, i_stream)
                            save(Q, q_stream)
                            save(m, iter_stream)
                            save(depth, depth_stream)
                            save(n, in_avg_stream)
                            save(j, rep_stream)

                        assign(depth_target, depth_target + 2 * delta_clifford)
                    assign(sequence_list[depth], saved_gate)
            

        with stream_processing():

            rep_stream.buffer(num_of_sequences,n_depth, n_avg).save_all("repetition")
            iter_stream.buffer(num_of_sequences,n_depth, n_avg).save_all("iteration")
            depth_stream.buffer(num_of_sequences,n_depth, n_avg).save_all("depth")
            in_avg_stream.buffer(num_of_sequences,n_depth, n_avg).save_all("sequence_average")
            i_stream.buffer(num_of_sequences,n_depth, n_avg).save_all("I")
            q_stream.buffer(num_of_sequences,n_depth, n_avg).save_all("Q")

    
    return qua_measurement


def measure_randomized_benchmarking(qubit_name='qA', num_seq = 20, inner_avg = 100, gate_index=0):

    num_of_sequences = num_seq #20  # Number of random sequences
    n_avg = inner_avg #100  # Number of averaging loops for each random sequence
    max_circuit_depth = 400 #400  # Maximum circuit depth
    delta_clifford = 10  #  Play each sequence with a depth step equals to 'delta_clifford - Must be > 1
    assert (max_circuit_depth / delta_clifford).is_integer(), "max_circuit_depth / delta_clifford must be an integer."
    seed = 345324  # Pseudo-random number generator seed
    # Flag to enable state discrimination if the readout has been calibrated (rotated blobs and threshold)
    state_discrimination = False
    # List of recovery gates from the lookup table
    inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]

    # index of the gate to interleave from the play_sequence() function defined below
    # Correspondence table:
    #  0: identity |  1: x180 |  2: y180
    # 12: x90      | 13: -x90 | 14: y90 | 15: -y90 |
    interleaved_gate_index = gate_index
    interleave_gate = get_interleaved_gate(interleaved_gate_index)

    msmt = randomized_benchmarking(
        qubit_name=qubit_name, 
        num_of_sequences=num_of_sequences, 
        n_avg=n_avg, 
        max_circuit_depth=max_circuit_depth, 
        delta_clifford=delta_clifford, 
        seed=seed, 
        inv_gates=inv_gates, 
        interleaved_gate_index=interleaved_gate_index,
        n_reps=1,
        collector_options=dict(batchsize=1),

    )

    data_loc, _ = run_measurement(sweep=msmt, name=f"{qubit_name}_randomized_benchmarking_interleave_{interleave_gate}")

    return data_loc

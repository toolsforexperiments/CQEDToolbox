"""Doc goes here."""

import numpy as np

from labcore.measurement.sweep import sweep_parameter
from labcore.analysis.mpl import fit_and_plot_1d
from labcore.data.datadict_storage import load_as_xr
from labcore.analysis.mpl import plot_fit_1d, fit_and_plot_1d
from labcore.analysis import DatasetAnalysis

from labcore.fitfuncs.generic import (
    Cosine,
    Gaussian,
    ExponentialDecay,
    ExponentiallyDecayingSine,
)
from cqedtoolbox.readout.qubit_readout import rotate_complex_qubit_data
from cqedtoolbox.measurement_lib.opx import single_transmon
from cqedtoolbox.setup_measurements import run_measurement, getp, param_from_name


# constants
SCRIPT = "qubit_tuneup"


#### RESONATOR TUNE UP ###########################################################################

def measure_pulse_resonator_spec():
    # things we infer automatically
    qn = getp("active.qubit", None)
    roif = getp(f"{qn}.readout.IF")
    bw = int(getp(f"{qn}.readout.bandwidth"))
    step = int(bw / 8)
    range = bw * int(getp(f"scripts.{SCRIPT}.resonator_spec_range"))
    start = roif - range // 2
    stop = roif + range // 2

    delay = 10_000
    single_transmon.options.repetition_delay = delay

    nreps = int(getp("opx.default_reps"))
    batchsize = int(getp("opx.default_batch"))

    measurement = single_transmon.pulsed_resonator_spec(
        start=start,
        stop=stop,
        step=step,
        n_reps=nreps,
        collector_options=dict(batchsize=batchsize),
    )
    data_loc, _ = run_measurement(sweep=measurement, name=f"{qn}_pulsed_resonator_spec")

    return data_loc


def measure_pulse_resonator_spec_vs_readout_amp():
    # things we infer automatically
    qn = getp("active.qubit", None)
    roif = getp(f"{qn}.readout.IF")
    bw = int(getp(f"{qn}.readout.bandwidth"))
    step = int(bw / 8)
    range = bw * int(getp(f"scripts.{SCRIPT}.resonator_spec_range"))
    start = roif - range // 2
    stop = roif + range // 2

    delay = 10_000
    single_transmon.options.repetition_delay = delay

    nreps = int(getp("opx.default_reps"))
    batchsize = int(getp("opx.default_batch"))


    ro_amp = param_from_name(f"{qn}.readout.short.amp")
    a0 = getp(f"scripts.{SCRIPT}.resonator_spec_vs_amp_a0")
    a1 = getp(f"scripts.{SCRIPT}.resonator_spec_vs_amp_a1")
    na = getp(f"scripts.{SCRIPT}.resonator_spec_vs_amp_na")



    measurement = single_transmon.pulsed_resonator_spec(
        start=start,
        stop=stop,
        step=step,
        n_reps=nreps,
        collector_options=dict(batchsize=nreps),
    )

    sweep = sweep_parameter(ro_amp, np.linspace(a0, a1, na)) @ measurement

    data_loc, _ = run_measurement(
        sweep=sweep, name=f"{qn}_pulsed_resonator_spec_vs_pwr"
    )

    return data_loc

def measure_pulse_resonator_spec_after_pi_pulse():
    # things we infer automatically
    qn = getp("active.qubit", None)
    roif = getp(f"{qn}.readout.IF")
    bw = int(getp(f"{qn}.readout.bandwidth"))
    step = int(bw / 8)
    range = bw * int(getp(f"scripts.{SCRIPT}.resonator_spec_range"))
    start = roif - range // 2
    stop = roif + range // 2

    nreps = int(getp("opx.default_reps"))
    batchsize = int(getp("opx.default_batch"))

    measurement = single_transmon.pulsed_resonator_spec_after_pi_pulse(
        start=start,
        stop=stop,
        step=step,
        n_reps=nreps,
        collector_options=dict(batchsize=batchsize),
    )
    data_loc, _ = run_measurement(sweep=measurement, name=f"{qn}_pulsed_resonator_spec_after_pulse")

    return data_loc


#### QUBIT TUNE UP ###########################################################################


def measure_qubit_ssb_spec_saturation():
    qn = getp("active.qubit", None)

    qnif = getp(f"{qn}.IF")
    step = int(getp(f"scripts.{SCRIPT}.saturation_spec_step"))
    range = int(getp(f"scripts.{SCRIPT}.saturation_spec_range"))
    start = qnif - range // 2
    stop = qnif + range // 2


    nreps = int(getp("opx.default_reps"))
    batchsize = int(getp("opx.default_batch"))

    measurement = single_transmon.qubit_ssb_spec_saturation(
        start=start,
        stop=stop,
        step=step,
        n_reps=nreps,
        collector_options=dict(batchsize=batchsize),
    )
    data_loc, _ = run_measurement(sweep=measurement, name=f"{qn}_ssb_saturation_spec")

    return data_loc


def measure_power_rabi():
    da = getp(f"scripts.{SCRIPT}.rabi_step", 0.05)
    rng = getp(f"scripts.{SCRIPT}.rabi_range", 2)
    nreps = int(getp("opx.default_reps"))
    batchsize = int(getp("opx.default_batch"))

    measurement = single_transmon.qubit_power_rabi(
        start=-rng,
        stop=rng,
        step=da,
        n_reps=nreps,
        collector_options=dict(batchsize=batchsize),
    )
    data_loc, _ = run_measurement(
        sweep=measurement, name=f"{single_transmon.options.qubit_element}_power_rabi"
    )
    return data_loc

def measure_time_rabi():
    qn = getp("active.qubit", None)

    time_rabi_stop = getp(f"scripts.{SCRIPT}.time_rabi_stop")
    time_rabi_points = getp(f"scripts.{SCRIPT}.time_rabi_time_points")


    nreps = int(getp("opx.default_reps"))
    batchsize = int(getp("opx.default_batch"))

    IF = getp(f"{qn}.IF")
    ssb_range = getp(f"scripts.{SCRIPT}.time_rabi_ssb_range")
    n_ssb = getp(f"scripts.{SCRIPT}.time_rabi_ssb_points")



    measurement = single_transmon.qubit_time_rabi(
        t0=20, 
        t1=20 + time_rabi_stop, 
        nt=time_rabi_points, 
        ssb0=IF - ssb_range,
        ssb1=IF + ssb_range,
        nssb=n_ssb,
        n_reps=nreps,
        collector_options=dict(batchsize=batchsize),
    )
    data_loc, _ = run_measurement(
        sweep=measurement, name=f"{single_transmon.options.qubit_element}_time_rabi"
    )
    return data_loc


def measure_tuneup_time_rabi(msmt_id=None):
    qn = getp("active.qubit", None)

    time_rabi_stop = getp(f"tuneup_gd.time_rabi_stop")
    time_rabi_points = getp(f"tuneup_gd.time_rabi_time_points")

    nreps = int(getp("opx.default_reps"))
    batchsize = int(getp("opx.default_batch"))

    measurement = single_transmon.qubit_tuneup_time_rabi(
        t0=20, 
        t1=20 + time_rabi_stop, 
        nt=time_rabi_points, 
        n_reps=nreps,
        collector_options=dict(batchsize=batchsize),
    )
    if msmt_id is None:
        data_loc, _ = run_measurement(
            sweep=measurement, name=f"{single_transmon.options.qubit_element}_tuneup_time_rabi"
        )
    else:
        data_loc, _ = run_measurement(
            sweep=measurement, name=f"{single_transmon.options.qubit_element}_tuneup_time_rabi_run_{msmt_id}"
        )
    return data_loc


def measure_line_search_time_rabi():
    """Checks the cost function at different multiples of the proposed update step to determine the optimal step size.
    
    Assumes that there is a square pi pulse defined for the qubit, and the parameter manager has the fields:
    - tuneup_gd.qssb0
    - tuneup_gd.qssb1
    - tuneup_gd.qamp_mid
    - tuneup_gd.rssb0
    - tuneup_gd.rssb1
    - tuneup_gd.ramp_mid
    - tuneup_gd.time_rabi_stop
    - tuneup_gd.time_rabi_time_points
    - tuneup_gd.line_search.proposed_step
    - tuneup_gd.line_search.m0
    - tuneup_gd.line_search.m1
    - tuneup_gd.line_search.nm
    """
    qn = getp("active.qubit", None)
    time_rabi_stop = getp(f"tuneup_gd.time_rabi_stop")
    time_rabi_points = getp(f"tuneup_gd.time_rabi_time_points")

    nreps = int(getp("opx.default_reps"))
    batchsize = int(getp("opx.default_batch"))

    proposed_step = getp("tuneup_gd.line_search.proposed_step")
    if isinstance(proposed_step, str):
        proposed_step = proposed_step.strip('[]').split()
        proposed_step = [float(x) for x in proposed_step]
    proposed_step = np.array(proposed_step)
    m0 = getp(f"tuneup_gd.line_search.m0")
    m1 = getp(f"tuneup_gd.line_search.m1")
    nm = getp(f"tuneup_gd.line_search.nm")

    qssb = int(np.mean([getp(f"tuneup_gd.qssb0"), getp(f"tuneup_gd.qssb1")]))
    qamp_center = getp(f"tuneup_gd.qamp_mid")
    rssb = int(np.mean([getp(f"tuneup_gd.rssb0"), getp(f"tuneup_gd.rssb1")]))
    ramp_center = getp(f"tuneup_gd.ramp_mid")

    pm_qamp = getp(f"{qn}.pulses.pi.square_amp")
    pm_ramp = getp(f"{qn}.readout.short.amp")

    param_from_name(f"{qn}.pulses.pi.square_amp")(qamp_center)
    param_from_name(f"{qn}.readout.short.amp")(ramp_center)

    measurement = single_transmon.qubit_line_search_time_rabi(
        t0=20,
        t1=20 + time_rabi_stop,
        nt = time_rabi_points,
        m0=m0,
        m1=m1,
        nm=nm,
        qssb=qssb,
        qamp=qamp_center,
        rssb=rssb,
        ramp=ramp_center,
        proposed_step=proposed_step,
        n_reps=nreps,
        collector_options=dict(batchsize=batchsize)
    )
    data_loc, _ = run_measurement(
        sweep=measurement, name=f"{single_transmon.options.qubit_element}_line_search_time_rabi"
    )

    param_from_name(f"{qn}.pulses.pi.square_amp")(pm_qamp)
    param_from_name(f"{qn}.readout.short.amp")(pm_ramp)

    return data_loc


def measure_gradient_time_rabi():
    """Precompiles the measurement for calculating gradient of time rabi msmt with respect to
    drive IF, qubit square pulse amplitude, readout IF, and readout amplitude.
    
    Assumes that qmconfig has a square pi pulse defined, and the parameter manager has fields:
    - tuneup_gd.qssb0
    - tuneup_gd.qssb1
    - tuneup_gd.qamp0
    - tuneup_gd.qamp1
    - tuneup_gd.qamp_mid
    - tuneup_gd.rssb0
    - tuneup_gd.rssb1
    - tuneup_gd.ramp0
    - tuneup_gd.ramp1
    - tuneup_gd.ramp_mid
    - tuneup_gd.time_rabi_stop
    - tuneup_gd.time_rabi_time_points
    """
    qn = getp("active.qubit", None)
    time_rabi_stop = getp(f"tuneup_gd.time_rabi_stop")
    time_rabi_points = getp(f"tuneup_gd.time_rabi_time_points")

    nreps = int(getp("opx.default_reps"))
    batchsize = int(getp("opx.default_batch"))

    qssb0, qssb1 = int(getp(f"tuneup_gd.qssb0")), int(getp(f"tuneup_gd.qssb1"))

    qamp0, qamp1 = getp(f"tuneup_gd.qamp0")/getp(f"tuneup_gd.qamp_mid"), getp(f"tuneup_gd.qamp1")/getp(f"tuneup_gd.qamp_mid")
    qamp_center = np.mean([getp(f"tuneup_gd.qamp0"), getp(f"tuneup_gd.qamp1")])

    rssb0, rssb1 = int(getp(f"tuneup_gd.rssb0")), int(getp(f"tuneup_gd.rssb1"))

    ramp0, ramp1 = getp(f"tuneup_gd.ramp0")/getp(f"tuneup_gd.ramp_mid"), getp(f"tuneup_gd.ramp1")/getp(f"tuneup_gd.ramp_mid")
    ramp_center = np.mean([getp(f"tuneup_gd.ramp0"), getp(f"tuneup_gd.ramp1")])

    pm_qamp = getp(f"{qn}.pulses.pi.square_amp")
    pm_ramp = getp(f"{qn}.readout.short.amp")

    param_from_name(f"{qn}.pulses.pi.square_amp")(qamp_center)
    param_from_name(f"{qn}.readout.short.amp")(ramp_center)

    measurement = single_transmon.qubit_gradient_time_rabi(
        t0=20, 
        t1=20 + time_rabi_stop, 
        nt=time_rabi_points, 
        qssb0=qssb0,
        qssb1=qssb1,
        qamp0=qamp0,
        qamp1=qamp1,
        qamp_center=qamp_center,
        rssb0=rssb0,
        rssb1=rssb1,
        ramp0=ramp0,
        ramp1=ramp1,
        ramp_center=ramp_center,
        n_reps=nreps,
        collector_options=dict(batchsize=batchsize),
    )
    data_loc, _ = run_measurement(
        sweep=measurement, name=f"{single_transmon.options.qubit_element}_gradient_time_rabi"
    )

    param_from_name(f"{qn}.pulses.pi.square_amp")(pm_qamp)
    param_from_name(f"{qn}.readout.short.amp")(pm_ramp)

    return data_loc


def measure_pi_spec():
    qn = getp("active.qubit", None)
    duration = getp(f"scripts.{SCRIPT}.pispec_pulselen")
    fc = getp(f"{qn}.IF")
    frange = getp(f"scripts.{SCRIPT}.qubit_spec_range")

    # dynamically make a weaker pipulse to narrow the line
    weaken_by = duration / (
        getp(f"{qn}.pulses.pi.sigma") * getp(f"{qn}.pulses.pi.nsigma")
    )
    amplitude = 1.0 / weaken_by

    # crudely estimate the linewidth; then roughly 5 pts per linewidth
    w = int(1.5 / (duration) * 1e9)
    fstep = int(w / 5)

    nreps = int(getp("opx.default_reps"))
    batchsize = int(getp("opx.default_batch"))
    measurement = single_transmon.qubit_ssb_spec_pi(
        start=int(fc - frange / 2),
        stop=int(fc + frange / 2),
        step=fstep,
        amplitude=amplitude,
        duration=duration // 4,
        n_reps=nreps,
        collector_options=dict(batchsize=batchsize),
    )

    data_loc, _ = run_measurement(sweep=measurement, name=f"{qn}_ssb_spec_pi")
    return data_loc


def analyze_pi_spec(loc, update=True):

    with DatasetAnalysis(loc, "pispec_fit") as da:
        data, _ = rotate_complex_qubit_data(load_as_xr(loc).mean("repetition"))
        qn = da.load_saved_parameter("active.qubit")

        data, result, fig = fit_and_plot_1d(
            ds=data, name="signal", fit_class=Gaussian, run_kwargs={}
        )

        da.add(
            plot=fig,
            fitresult=result,
            data=data,
        )
        da.to_table(name=f"{qn}_pi_spec", data=dict(x0=result.params["x0"].value))

    if update:
        param = f"{qn}.IF"
        param_from_name(param)(np.round(result.params["x0"].value, 0))

    return da


def measure_t1():
    qn = getp("active.qubit", None)
    start = 16
    stop = getp(f"{qn}.T1") * 5
    npts = 5 * getp(f"scripts.{SCRIPT}.pts_per_t1") + 1

    nreps = int(getp("opx.default_reps"))
    batchsize = int(getp("opx.default_batch"))
    measurement = single_transmon.qubit_T1(
        start=start,
        stop=stop,
        npts=npts,
        n_reps=nreps,
        collector_options=dict(batchsize=batchsize),
    )
    data_loc, _ = run_measurement(sweep=measurement, name=f"{qn}_T1")
    return data_loc


def analyze_t1(loc, update=True, fit_kwargs={}):

    with DatasetAnalysis(loc, "t1_fit") as da:
        qn = da.load_saved_parameter("active.qubit")

        data, _ = rotate_complex_qubit_data(load_as_xr(loc).mean("repetition"))
        data.coords['delay'] = data['delay'] * 1e-3; data.delay.attrs['units'] = 'us' # convert to us

        data, result, fig = fit_and_plot_1d(
            ds=data,
            name="signal",
            fit_class=ExponentialDecay,
            run_kwargs=fit_kwargs,
        )

        da.add(
            plot=fig,
            fitresult=result,
            data=data,
        )
        da.to_table(
            name=f"{qn}_t1", 
            data={
                "T1 (us)": result.params["tau"].value,
            }
        )

    if update:
        param = f"{qn}.T1"
        param_from_name(param)(
            np.round(result.lmfit_result.params["tau"].value, 0) * 1e3
        )

    return da

def measure_t2(n_echos, nppp=8):
    qn = getp("active.qubit", None)
    nppp = getp(f"scripts.{SCRIPT}.t2_nppp")
    if n_echos == 0:
        max_delay = getp(f"{qn}.T2R") * 3
        oscillations = getp(f"scripts.{SCRIPT}.oscillations_per_t2r")
        t2 = getp(f"{qn}.T2R")
    else:
        max_delay = getp(f"{qn}.T2E") * 3
        oscillations = getp(f"scripts.{SCRIPT}.oscillations_per_t2e")
        t2 = getp(f"{qn}.T2E")
    
    period = int(t2 / oscillations)
    step = period // nppp
    npts = max_delay // step
    detuning = 1.0 / period * 1e3

    nreps = int(getp("opx.default_reps"))
    batchsize = int(getp("opx.default_batch"))
    measurement = single_transmon.qubit_T2(
        start=16,
        stop=max_delay,
        npts=npts,
        n_reps=nreps,
        detuning_MHz=detuning,
        n_echos=n_echos,
        collector_options=dict(batchsize=batchsize),
    )
    data_loc, _ = run_measurement(
        sweep=measurement, name=f"{qn}_T2-{n_echos}_Echo_{detuning:.3f}MHz_detuned"
    )
    return data_loc


def analyze_t2r(loc, update=True, fit_kwargs={}):

    with DatasetAnalysis(loc, 't2r_fit') as da:
        qn = da.load_saved_parameter("active.qubit")

        data, _ = rotate_complex_qubit_data(load_as_xr(loc).mean('repetition'))
        data.coords['delay'] = data['delay'] * 1e-3; data.delay.attrs['units'] = 'us' # convert to us

        data, result, fig = fit_and_plot_1d(
            ds=data,
            name='signal',
            fit_class=ExponentiallyDecayingSine,
            run_kwargs=fit_kwargs,
        )

        da.add(
            plot=fig,
            fitresult=result,
            data=data,
        )

        da.to_table(
            name=f"{qn}_t2r", 
            data={
                "T2R (us)": result.params["tau"].value,
                "detuning (MHz)": result.params["f"].value,
            }
        )

    ## update parameter manager:
    if update:
        param = f"{qn}.T2R"
        param_from_name(param)(np.round(result.lmfit_result.params['tau'].value,0) * 1e3)

    return da


def analyze_t2e(loc, update=True, fit_kwargs={}):

    with DatasetAnalysis(loc, 't2e_fit') as da:
        qn = da.load_saved_parameter("active.qubit")

        data, _ = rotate_complex_qubit_data(load_as_xr(loc).mean('repetition'))
        data.coords['delay'] = data['delay'] * 1e-3; data.delay.attrs['units'] = 'us' # convert to us

        data, result, fig = fit_and_plot_1d(
            ds=data,
            name='signal',
            fit_class=ExponentiallyDecayingSine,
            run_kwargs=fit_kwargs,
        )

        da.add(
            plot=fig,
            fitresult=result,
            data=data,
        )

        da.to_table(
            name=f"{qn}_t2e", 
            data={
                "T2E (us)": result.params["tau"].value,
                "detuning (MHz)": result.params["f"].value,
            }
        )

    ## update parameter manager:
    if update:
        param = f"{qn}.T2E"
        param_from_name(param)(np.round(result.lmfit_result.params['tau'].value,0) * 1e3)

    return da


# calls readout calibration measurement
def measure_readout_calibration(qubit_name, n_reps):
    """
    This calls the readout_calibration measurement.
    We pass in the qubit we want to calibrate and the number of reps we want
    """

    msmt = single_transmon.readout_calibration(n_reps=n_reps, collector_options=dict(batchsize=n_reps))
    data_loc, _ = run_measurement(sweep=msmt, name=f"{qubit_name}_ROcal")
    return data_loc


import logging
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

from labcore.analysis import DatasetAnalysis, FitResult
from labcore.measurement.storage import run_and_save_sweep
from labcore.data.datadict_storage import datadict_from_hdf5, load_as_xr
from labcore.measurement import sweep_parameter, record_as

from labcore.protocols.base import (ProtocolOperation, OperationStatus, serialize_fit_params,
                                    ParamImprovement, CorrectionParameter, CheckResult, Correction, PlatformTypes)
from cqedtoolbox.protocols.operations import ResonatorGeometry
from cqedtoolbox.protocols.parameters import (Repetition,
                                              ResonatorSpecSteps, ReadoutGain, ReadoutLength, StartReadoutFrequency,
                                              EndReadoutFrequency, ReadoutFrequency, nestedAttributeFromString)
from cqedtoolbox.measurement_lib.opx.advanced.qubit_tuneup import measure_pulse_resonator_spec
from cqedtoolbox.measurement_lib.qick.single_transmon_v2 import FreqSweepProgram

from cqedtoolbox.fitfuncs.resonators import HangerResponseBruno, ReflectionResponse, TransmissionResponse


logger = logging.getLogger(__name__)


@dataclass
class UnwindAndFitRet:
    signal_unwind: ArrayLike
    magnitude: ArrayLike
    phase: ArrayLike
    fit_curve: ArrayLike
    fit_result: FitResult
    residuals: ArrayLike
    residual_std: float
    snr: float
    unwind_sign: int
    fig: plt.Figure
    ax: plt.Axes

@dataclass
class SyntheticHangerResonatorData:
    f0: float
    Qc: float
    Qi: float
    A: float
    phi: float
    noise_amp: float
    
    def generate(self, frequencies: ArrayLike) -> ArrayLike:
        Q_l = 1./(1./self.Qc + 1./self.Qi)
        Q_e_complex = self.Qc * np.exp(-1j * self.phi)
        response = self.A * (1 - (Q_l / Q_e_complex) / (1 + 2j * Q_l * (frequencies - self.f0) / self.f0))
        return response + self.noise_amp * (np.random.randn() + 1j * np.random.randn())


@dataclass
class SNRThreshold(CorrectionParameter):
    name: str = field(default="resonator_spec_SNR_threshold", init=False)
    description: str = field(default="SNR threshold", init=False)

    def _qick_getter(self):
        return self.params.corrections.res_spec.snr()

    def _qick_setter(self, value):
        self.params.corrections.res_spec.snr(value)

    def _opx_getter(self):
        return self.params.corrections.res_spec.snr()

    def _opx_setter(self, value):
        self.params.corrections.res_spec.snr(value)

@dataclass
class MaxWindowShifts(CorrectionParameter):
    name: str = field(default="res_spec_max_window_shifts", init=False)
    description: str = field(default="Number of ±n window shifts to try", init=False)

    def _qick_getter(self):
        return int(self.params.corrections.res_spec.max_window_shifts())

    def _qick_setter(self, value):
        self.params.corrections.res_spec.max_window_shifts(value)

    def _opx_getter(self):
        return int(self.params.corrections.res_spec.max_window_shifts())

    def _opx_setter(self, value):
        self.params.corrections.res_spec.max_window_shifts(value)

@dataclass
class SamplingIncreaseFactor(CorrectionParameter):
    name: str = field(default="res_spec_sampling_increase_factor", init=False)
    description: str = field(default="Factor by which to increase frequency steps", init=False)

    def _qick_getter(self):
        return self.params.corrections.res_spec.sampling_factor()

    def _qick_setter(self, value):
        self.params.corrections.res_spec.sampling_factor(value)

    def _opx_getter(self):
        return self.params.corrections.res_spec.sampling_factor()

    def _opx_setter(self, value):
        self.params.corrections.res_spec.sampling_factor(value)

@dataclass
class MaxSamplingIncreases(CorrectionParameter):
    name: str = field(default="res_spec_max_sampling_increases", init=False)
    description: str = field(default="Maximum number of sampling rate increases to try", init=False)

    def _qick_getter(self):
        return int(self.params.corrections.res_spec.max_sampling_increases())

    def _qick_setter(self, value):
        self.params.corrections.res_spec.max_sampling_increases(value)

    def _opx_getter(self):
        return int(self.params.corrections.res_spec.max_sampling_increases())

    def _opx_setter(self, value):
        self.params.corrections.res_spec.max_sampling_increases(value)

@dataclass
class AveragingIncreaseFactor(CorrectionParameter):
    name: str = field(default="res_spec_averaging_increase_factor", init=False)
    description: str = field(default="Factor by which to increase repetitions", init=False)

    def _qick_getter(self):
        return self.params.corrections.res_spec.averaging_factor()

    def _qick_setter(self, value):
        self.params.corrections.res_spec.averaging_factor(value)

    def _opx_getter(self):
        return self.params.corrections.res_spec.averaging_factor()

    def _opx_setter(self, value):
        self.params.corrections.res_spec.averaging_factor(value)

@dataclass
class MaxAveragingIncreases(CorrectionParameter):
    name: str = field(default="res_spec_max_averaging_increases", init=False)
    description: str = field(default="Maximum number of averaging increases to try", init=False)

    def _qick_getter(self):
        return int(self.params.corrections.res_spec.max_averaging_increases())

    def _qick_setter(self, value):
        self.params.corrections.res_spec.max_averaging_increases(value)

    def _opx_getter(self):
        return int(self.params.corrections.res_spec.max_averaging_increases())

    def _opx_setter(self, value):
        self.params.corrections.res_spec.max_averaging_increases(value)

@dataclass
class MaxFitParamError(CorrectionParameter):
    name: str = field(default="res_spec_max_fit_param_error", init=False)
    description: str = field(default="Maximum allowed fractional fit parameter error (e.g. 1.0 = 100%)", init=False)

    def _qick_getter(self):
        return self.params.corrections.res_spec.max_fit_param_error()

    def _qick_setter(self, value):
        self.params.corrections.res_spec.max_fit_param_error(value)

    def _opx_getter(self):
        return self.params.corrections.res_spec.max_fit_param_error()

    def _opx_setter(self, value):
        self.params.corrections.res_spec.max_fit_param_error(value)

class WindowShiftCorrection(Correction):
    name = "window_shift"
    description = "Shift measurement window by multiples of the original window span"
    triggered_by = "snr_check"

    def __init__(self, start_param, end_param, max_shifts_param):
        self.start_param = start_param
        self.end_param = end_param
        self.max_shifts_param = max_shifts_param
        self._original_start: float | None = None
        self._original_end: float | None = None
        self._idx = 0
        self._last_new_start: float | None = None
        self._last_new_end: float | None = None

    @staticmethod
    def _shift_multiplier(idx: int) -> int:
        """idx 0 → +1, 1 → -1, 2 → +2, 3 → -2, ..."""
        n = idx // 2 + 1
        return n if idx % 2 == 0 else -n

    def can_apply(self) -> bool:
        return self._idx < int(self.max_shifts_param()) * 2

    def apply(self) -> None:
        if self._original_start is None:
            self._original_start = self.start_param()
            self._original_end = self.end_param()
        span = self._original_end - self._original_start
        shift = self._shift_multiplier(self._idx) * span
        self._last_new_start = self._original_start + shift
        self._last_new_end = self._original_end + shift
        self.start_param(self._last_new_start)
        self.end_param(self._last_new_end)
        self._idx += 1

    def report_output(self) -> str:
        if self._last_new_start is None:
            return ""
        return (f"[{self._last_new_start:.4f}, {self._last_new_end :.4f}] MHz"
                f" (shift={(self._last_new_start - self._original_start):+.1f} MHz)")

    def reset(self) -> None:
        """Restore original window and reset index. Called by higher-level corrections."""
        if self._original_start is not None:
            self.start_param(self._original_start)
            self.end_param(self._original_end)
        self._idx = 0


class IncreaseSamplingRateCorrection(Correction):
    name = "increase_sampling_rate"
    description = "Increase frequency step count and reset window shift"
    triggered_by = "snr_check"

    def __init__(self, steps_param, window_correction: WindowShiftCorrection,
                 factor_param, max_increases_param):
        self.steps_param = steps_param
        self.window_correction = window_correction
        self.factor_param = factor_param
        self.max_increases_param = max_increases_param
        self._original_steps: int | None = None
        self._count = 0
        self._last_change: str = ""

    def can_apply(self) -> bool:
        return self._count < int(self.max_increases_param())

    def apply(self) -> None:
        if self._original_steps is None:
            self._original_steps = int(self.steps_param())
        factor = self.factor_param()
        old = int(self.steps_param())
        new = int(self._original_steps * (factor ** (self._count + 1)))
        self.steps_param(new)
        self._count += 1
        self._last_change = f"steps: {old} → {new}"
        self.window_correction.reset()

    def report_output(self) -> str:
        return self._last_change


class IncreaseAveragingCorrection(Correction):
    name = "increase_averaging"
    description = "Increase repetitions and reset window shift"
    triggered_by = "snr_check"

    def __init__(self, reps_param, window_correction: WindowShiftCorrection,
                 factor_param, max_increases_param):
        self.reps_param = reps_param
        self.window_correction = window_correction
        self.factor_param = factor_param
        self.max_increases_param = max_increases_param
        self._original_reps: int | None = None
        self._count = 0
        self._last_change: str = ""

    def can_apply(self) -> bool:
        return self._count < int(self.max_increases_param())

    def apply(self) -> None:
        if self._original_reps is None:
            self._original_reps = int(self.reps_param())
        factor = self.factor_param()
        old = int(self.reps_param())
        new = int(self._original_reps * (factor ** (self._count + 1)))
        self.reps_param(new)
        self._count += 1
        self._last_change = f"reps: {old} → {new}"
        self.window_correction.reset()

    def report_output(self) -> str:
        return self._last_change
    

def fit_sine(t, data):
    """Fit a sine wave to the given data."""
    t = np.asarray(t, dtype=float)
    data = np.asarray(data, dtype=float)

    def sine_function(x, amplitude, frequency, phase, offset):
        return amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset

    y_offset = np.mean(data)
    amplitude = (np.max(data) - np.min(data)) / 2

    fft = np.fft.fft(data - y_offset)
    freqs = np.fft.fftfreq(len(data), d=t[1] - t[0])
    positive = np.argmax(np.abs(fft[1:len(fft) // 2 + 1])) + 1 if len(fft) > 2 else 0
    frequency = np.abs(freqs[positive])
    phase = np.angle(fft[positive]) if positive else 0.0

    initial_guess = (amplitude, frequency, phase, y_offset)
    params, covariance = curve_fit(sine_function, t, data, p0=initial_guess, maxfev=5000)
    return params, covariance

def background_filter(x, y):
    window_size = 8

    def moving_window_variance(data):
        half_window = window_size // 2
        padded_data = np.pad(data, (half_window, half_window), mode="reflect")
        return np.array([np.var(padded_data[i:i + window_size]) for i in range(len(data))])

    def filtered_variance(x, y):
        smooth = savgol_filter(y, window_size, 1)
        spline_derivative = CubicSpline(x, smooth)(y, 1)
        return savgol_filter(moving_window_variance(spline_derivative), window_size, 1)

    s = filtered_variance(x, y.real) + 1j * filtered_variance(x, y.imag)
    sabs = np.abs(s)
    s0 = np.median(sabs)
    sm, sp = s0 - 0.5 * sabs.std(), s0 + 0.5 * sabs.std()
    return (sabs > sm) & (sabs < sp)

def unwind_signal(x, y, f=None, sign=1):
    """Fit and remove the dominant sinusoidal loading from the complex signal."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)

    if f is None:
        bg = background_filter(x, y)
        if np.count_nonzero(bg) < 5:
            bg = np.ones(len(x), dtype=bool)

        xbg, ybg = x[bg], y[bg]
        ixbg = np.linspace(x[0], x[-1], x.size)
        iybg = np.interp(ixbg, xbg, ybg)

        pr, _ = fit_sine(ixbg, iybg.real)
        pi, _ = fit_sine(ixbg, iybg.imag)
        f = np.mean([pr[1], pi[1]])

    unwound = y * np.exp(1j * sign * 2 * np.pi * f * x)
    return unwound.real, unwound.imag, f


class ResonatorSpectroscopy(ProtocolOperation):
    _SIM_F0 = 7e9
    _SIM_QI = 20e3
    _SIM_QC = 20e3
    _SIM_A = 4.0
    _SIM_PHI = 0.0
    _SIM_NOISE_AMP = 0.05
    
    def __init__(self, params, geometry: ResonatorGeometry | str):
        super().__init__()
        self.params = params

        if isinstance(geometry, str):
            try:
                geometry = ResonatorGeometry(geometry.lower())
            except ValueError as err:
                valid = ", ".join(g.value for g in ResonatorGeometry)
                raise ValueError(
                    f"Unsupported resonator geometry '{geometry}'. Expected one of: {valid}"
                ) from err

        self.geometry = geometry
        self._fit_cls = geometry.fit_cls

        self._register_inputs(
            repetitions=Repetition(params),
            steps=ResonatorSpecSteps(params),
            gain=ReadoutGain(params),
            length=ReadoutLength(params),
            start_frequency=StartReadoutFrequency(params),
            end_frequency=EndReadoutFrequency(params),
        )
        self._register_outputs(
            readout_freq=ReadoutFrequency(params),
        )

        self._register_correction_params(
            snr_threshold=SNRThreshold(params),
            max_window_shifts=MaxWindowShifts(params),
            sampling_increase_factor=SamplingIncreaseFactor(params),
            max_sampling_increases=MaxSamplingIncreases(params),
            averaging_increase_factor=AveragingIncreaseFactor(params),
            max_averaging_increases=MaxAveragingIncreases(params),
            max_fit_param_error=MaxFitParamError(params),
        )

        self.params = params

        self._window_shift = WindowShiftCorrection(
            self.start_frequency,
            self.end_frequency,
            self.max_window_shifts,
        )
        self._increase_sampling = IncreaseSamplingRateCorrection(
            self.steps,
            self._window_shift,
            self.sampling_increase_factor,
            self.max_sampling_increases,
        )
        self._increase_averaging = IncreaseAveragingCorrection(
            self.repetitions,
            self._window_shift,
            self.averaging_increase_factor,
            self.max_averaging_increases,
        )

        self._register_check(
            "quality_check",
            self._check_quality,
            [self._window_shift, self._increase_sampling, self._increase_averaging],
        )
        self._register_check(
            "fit_in_range",
            self._check_fit_in_range,
        )

        self._register_success_update(
            self.readout_freq,
            lambda: self.fit_result.params["f_0"].value,
        )

        self._register_success_update(
            self.start_frequency,
            lambda: self.fit_result.params["f_0"].value - 5 if self.platform_type == PlatformTypes.QICK else self.fit_result.params["f_0"].value - 5e6,
        )

        self._register_success_update(
            self.end_frequency,
            lambda: self.fit_result.params["f_0"].value + 5 if self.platform_type == PlatformTypes.QICK else self.fit_result.params["f_0"].value + 5e6,
        )

        self.condition = f"Success if the SNR of the measurement is bigger than the current threshold of " # {self.SNR_THRESHOLD}"

        self.independents = {"frequencies": []}
        self.dependents = {"signal": []}
        self.unwind_signal = None
        self.magnitude = None
        self.phase = None
        self.snr = None
        self.fit_result = None
        self.improvements = None

    def _measure_qick(self) -> Path:
        logger.info("Starting qick resonator spectroscopy measurement")

        sweep = FreqSweepProgram()
        logger.debug("Sweep created, running measurement")
        loc, da = run_and_save_sweep(sweep, "data", self.name)
        logger.info("Measurement complete")

        return loc

    def _measure_opx(self) -> Path:
        logger.info("Starting opx resonator spectroscopy measurement")
        loc = measure_pulse_resonator_spec()
        logger.info("Measurement complete")
        return loc
    
    def _measure_dummy(self):
        logger.info("Starting dummy resonator spectroscopy measurement")
        frequencies = np.linspace(self.start_frequency(), self.end_frequency(), int(self.steps()))
        generator = SyntheticHangerResonatorData(
            f0 = self._SIM_F0,
            Qi = self._SIM_QI,
            Qc = self._SIM_QC,
            A = self._SIM_A,
            phi = self._SIM_PHI,
            noise_amp = self._SIM_NOISE_AMP
        )

        sweep = sweep_parameter("frequencies", frequencies + self.readout_lo(), record_as(generator.generate, "signal"))
        loc, _ = run_and_save_sweep(sweep, "data", self.name)

        logger.info("Dummy measurement complete")
        return loc

    def _load_data_qick(self):
        path = self.data_loc/"data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)

        self.independents["frequencies"] = data["freq"]["values"]
        self.dependents["signal"] = data["signal"]["values"]

    def _load_data_opx(self):
        data = load_as_xr(self.data_loc).mean("repetition")
        q = nestedAttributeFromString(self.params, "active.qubit")()
        lo = nestedAttributeFromString(self.params, f"{q}.readout.LO")()
        self.independents["frequencies"] = data["ssb_frequency"].values + lo
        self.dependents["signal"] = data["signal_Re"].values + 1j * data["signal_Im"].values

    def _load_data_dummy(self):
        path = self.data_loc/"data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)

        self.independents["frequencies"] = data["frequencies"]["values"]
        self.dependents["signal"] = data["signal"]["values"]

    @staticmethod
    def add_mag_and_unwind_and_fit(frequencies, signal_raw, platform_type, fit_cls, fig_title="") -> UnwindAndFitRet:
        frequencies = np.asarray(frequencies, dtype=float)
        signal_raw = np.asarray(signal_raw)

        magnitude = np.abs(signal_raw)
        del platform_type

        def fit_candidate(sign: int):
            unwound_real, unwound_imag, _ = unwind_signal(
                frequencies, signal_raw, sign=sign
            )
            signal_unwind = unwound_real + 1j * unwound_imag
            fit = fit_cls(frequencies, signal_unwind)
            fit_result = fit.run(fit)
            fit_curve = fit_result.eval()
            residuals = signal_unwind - fit_curve
            residual_std = float(np.std(residuals))
            amp = fit_result.params["A"].value
            snr = np.abs(amp / (4 * residual_std)) if residual_std > 0 else float("inf")
            return signal_unwind, fit_result, fit_curve, residuals, residual_std, snr

        candidates = [fit_candidate(sign) for sign in (1, -1)]
        best_idx = min(range(len(candidates)), key=lambda i: candidates[i][4])
        signal_unwind, fit_result, fit_curve, residuals, residual_std, snr = candidates[best_idx]
        unwind_sign = 1 if best_idx == 0 else -1
        phase = np.angle(signal_unwind)

        fig, ax = plt.subplots()
        ax.set_title(fig_title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude Signal (A.U)")
        ax.plot(frequencies, magnitude, label="Data")
        ax.plot(frequencies, np.abs(fit_curve), label="Fit")
        ax.legend()

        ret = UnwindAndFitRet(
            signal_unwind=signal_unwind,
            magnitude=magnitude,
            phase=phase,
            fit_curve=fit_curve,
            fit_result=fit_result,
            residuals=residuals,
            residual_std=residual_std,
            snr=snr,
            unwind_sign=unwind_sign,
            fig=fig,
            ax=ax,
        )

        return ret

    def analyze(self):
        with DatasetAnalysis(self.data_loc, self.name) as ds:
            ret = self.add_mag_and_unwind_and_fit(self.independents["frequencies"],
                                                  self.dependents["signal"],
                                                  self.platform_type,
                                                  self._fit_cls,
                                                  "Resonator Spectroscopy")

            self.unwind_signal = ret.signal_unwind
            self.magnitude = ret.magnitude
            self.phase = ret.phase
            self.snr = ret.snr
            self.fit_result = ret.fit_result

            ds.add(fit_curve=ret.fit_curve,
                   fit_result=ret.fit_result,
                   params=serialize_fit_params(ret.fit_result.params),
                   snr=float(ret.snr))
            ds.add_figure(self.name, fig=ret.fig)

            image_path = ds._new_file_path(ds.savefolders[1], self.name, suffix="png")
            self.figure_paths.append(image_path)

    def _check_quality(self) -> CheckResult:
        if self.snr is None or self.fit_result is None:
            raise RuntimeError("SNR and fit result must be set before checking quality")

        threshold = self.snr_threshold()
        snr_passed = self.snr >= threshold

        max_error = self.max_fit_param_error()
        bad_params = []
        for pname, param in self.fit_result.params.items():
            if pname in ["transmission_slope", "phase_slope", "phase_offset"]:
                continue
            if param.stderr is None:
                bad_params.append(f"{pname}(no stderr)")
            elif param.value == 0 or abs(param.stderr / param.value) > max_error:
                pct = abs(param.stderr / param.value) * 100 if param.value != 0 else float("inf")
                bad_params.append(f"{pname}({pct:.0f}%)")

        fit_passed = len(bad_params) == 0
        passed = snr_passed and fit_passed

        parts = [f"SNR={self.snr:.3f} (threshold={threshold:.3f})"]
        if bad_params:
            parts.append(f"high-error params: {', '.join(bad_params)}")

        return CheckResult("quality_check", passed, "; ".join(parts))

    def _check_fit_in_range(self) -> CheckResult:
        if self.fit_result is None:
            raise RuntimeError("Fit result must be set before checking fit range")

        freqs = np.asarray(self.independents["frequencies"], dtype=float)
        f0 = self.fit_result.params["f_0"].value
        fmin = float(np.min(freqs))
        fmax = float(np.max(freqs))
        passed = fmin <= f0 <= fmax

        return CheckResult(
            "fit_in_range",
            passed,
            f"f_0={f0:.6g}, sweep=[{fmin:.6g}, {fmax:.6g}]",
        )

import logging
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from scipy.constants import h

plt.switch_backend("agg")

from labcore.analysis import DatasetAnalysis, Fit
from labcore.analysis.fitfuncs.generic import Lorentzian
from labcore.measurement.storage import run_and_save_sweep
from labcore.data.datadict_storage import datadict_from_hdf5, load_as_xr
from labcore.measurement.sweep import sweep_parameter
from labcore.measurement.record import record_as

from labcore.protocols.base import (ProtocolOperation, OperationStatus, serialize_fit_params,
                                    ParamImprovement, CorrectionParameter, CheckResult, Correction,
                                    EvaluateResult)
from cqedtoolbox.protocols.parameters import (
    Repetition,
    SaturationSpecSteps,
    StartSaturationSpecFrequency, EndSaturationSpecFrequency, QubitFrequency,
    SaturationSpecDriveGain,
)
from cqedtoolbox.measurement_lib.opx.advanced.qubit_tuneup import measure_qubit_ssb_spec_saturation
from cqedtoolbox.measurement_lib.qick.single_transmon_v2 import PulseProbeSpectroscopy
from cqedtoolbox.readout.qubit_readout import rotate_complex_qubit_data


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Correction parameters
# ---------------------------------------------------------------------------

@dataclass
class SNRThreshold(CorrectionParameter):
    name: str = field(default="sat_spec_snr_threshold", init=False)
    description: str = field(default="SNR threshold for saturation spectroscopy fit quality", init=False)

    def _qick_getter(self):
        return self.params.corrections.sat_spec.snr()

    def _qick_setter(self, value):
        self.params.corrections.sat_spec.snr(value)

    def _opx_getter(self):
        return self.params.corrections.sat_spec.snr()

    def _opx_setter(self, value):
        self.params.corrections.sat_spec.snr(value)


@dataclass
class MaxFitParamError(CorrectionParameter):
    name: str = field(default="sat_spec_max_fit_param_error", init=False)
    description: str = field(default="Maximum allowed fractional fit parameter error (e.g. 1.0 = 100%)", init=False)

    def _qick_getter(self):
        return self.params.corrections.sat_spec.max_fit_param_error()

    def _qick_setter(self, value):
        self.params.corrections.sat_spec.max_fit_param_error(value)

    def _opx_getter(self):
        return self.params.corrections.sat_spec.max_fit_param_error()

    def _opx_setter(self, value):
        self.params.corrections.sat_spec.max_fit_param_error(value)


@dataclass
class MaxWindowShifts(CorrectionParameter):
    name: str = field(default="sat_spec_max_window_shifts", init=False)
    description: str = field(default="Number of ±n window shifts to try", init=False)

    def _qick_getter(self):
        return int(self.params.corrections.sat_spec.max_window_shifts())

    def _qick_setter(self, value):
        self.params.corrections.sat_spec.max_window_shifts(value)

    def _opx_getter(self):
        return int(self.params.corrections.sat_spec.max_window_shifts())

    def _opx_setter(self, value):
        self.params.corrections.sat_spec.max_window_shifts(value)


@dataclass
class AveragingIncreaseFactor(CorrectionParameter):
    name: str = field(default="sat_spec_averaging_increase_factor", init=False)
    description: str = field(default="Factor by which to increase repetitions", init=False)

    def _qick_getter(self):
        return self.params.corrections.sat_spec.averaging_factor()

    def _qick_setter(self, value):
        self.params.corrections.sat_spec.averaging_factor(value)

    def _opx_getter(self):
        return self.params.corrections.sat_spec.averaging_factor()

    def _opx_setter(self, value):
        self.params.corrections.sat_spec.averaging_factor(value)


@dataclass
class MaxAveragingIncreases(CorrectionParameter):
    name: str = field(default="sat_spec_max_averaging_increases", init=False)
    description: str = field(default="Maximum number of averaging increases to try", init=False)

    def _qick_getter(self):
        return int(self.params.corrections.sat_spec.max_averaging_increases())

    def _qick_setter(self, value):
        self.params.corrections.sat_spec.max_averaging_increases(value)

    def _opx_getter(self):
        return int(self.params.corrections.sat_spec.max_averaging_increases())

    def _opx_setter(self, value):
        self.params.corrections.sat_spec.max_averaging_increases(value)


@dataclass
class SamplingIncreaseFactor(CorrectionParameter):
    name: str = field(default="sat_spec_sampling_increase_factor", init=False)
    description: str = field(default="Factor by which to increase frequency steps", init=False)

    def _qick_getter(self):
        return self.params.corrections.sat_spec.sampling_factor()

    def _qick_setter(self, value):
        self.params.corrections.sat_spec.sampling_factor(value)

    def _opx_getter(self):
        return self.params.corrections.sat_spec.sampling_factor()

    def _opx_setter(self, value):
        self.params.corrections.sat_spec.sampling_factor(value)


@dataclass
class MaxSamplingIncreases(CorrectionParameter):
    name: str = field(default="sat_spec_max_sampling_increases", init=False)
    description: str = field(default="Maximum number of sampling rate increases to try", init=False)

    def _qick_getter(self):
        return int(self.params.corrections.sat_spec.max_sampling_increases())

    def _qick_setter(self, value):
        self.params.corrections.sat_spec.max_sampling_increases(value)

    def _opx_getter(self):
        return int(self.params.corrections.sat_spec.max_sampling_increases())

    def _opx_setter(self, value):
        self.params.corrections.sat_spec.max_sampling_increases(value)


@dataclass
class MaxPowerIncreases(CorrectionParameter):
    name: str = field(default="sat_spec_max_power_increases", init=False)
    description: str = field(default="Maximum number of drive power increases to try", init=False)

    def _qick_getter(self):
        return int(self.params.corrections.sat_spec.max_power_increases())

    def _qick_setter(self, value):
        self.params.corrections.sat_spec.max_power_increases(value)

    def _opx_getter(self):
        return int(self.params.corrections.sat_spec.max_power_increases())

    def _opx_setter(self, value):
        self.params.corrections.sat_spec.max_power_increases(value)


@dataclass
class PowerIncreaseFactor(CorrectionParameter):
    name: str = field(default="sat_spec_power_increase_factor", init=False)
    description: str = field(default="Multiplicative factor for increasing drive gain (e.g. 1.1 = +10%)", init=False)

    def _qick_getter(self):
        return self.params.corrections.sat_spec.power_increase_factor()

    def _qick_setter(self, value):
        self.params.corrections.sat_spec.power_increase_factor(value)

    def _opx_getter(self):
        return self.params.corrections.sat_spec.power_increase_factor()

    def _opx_setter(self, value):
        self.params.corrections.sat_spec.power_increase_factor(value)


@dataclass
class SinglePeakSNRThreshold(CorrectionParameter):
    name: str = field(default="sat_spec_single_peak_snr", init=False)
    description: str = field(default="SNR threshold for detecting a second peak in the fit residuals", init=False)

    def _qick_getter(self):
        return self.params.corrections.sat_spec.single_peak_snr()

    def _qick_setter(self, value):
        self.params.corrections.sat_spec.single_peak_snr(value)

    def _opx_getter(self):
        return self.params.corrections.sat_spec.single_peak_snr()

    def _opx_setter(self, value):
        self.params.corrections.sat_spec.single_peak_snr(value)


@dataclass
class SinglePeakMaxPowerReductions(CorrectionParameter):
    name: str = field(default="sat_spec_single_peak_max_reductions", init=False)
    description: str = field(default="Maximum number of drive power reductions to try", init=False)

    def _qick_getter(self):
        return int(self.params.corrections.sat_spec.single_peak_max_reductions())

    def _qick_setter(self, value):
        self.params.corrections.sat_spec.single_peak_max_reductions(value)

    def _opx_getter(self):
        return int(self.params.corrections.sat_spec.single_peak_max_reductions())

    def _opx_setter(self, value):
        self.params.corrections.sat_spec.single_peak_max_reductions(value)


@dataclass
class PowerReductionFactor(CorrectionParameter):
    name: str = field(default="sat_spec_power_reduction_factor", init=False)
    description: str = field(default="Multiplicative factor for reducing drive gain (e.g. 0.9 = -10%)", init=False)

    def _qick_getter(self):
        return self.params.corrections.sat_spec.power_reduction_factor()

    def _qick_setter(self, value):
        self.params.corrections.sat_spec.power_reduction_factor(value)

    def _opx_getter(self):
        return self.params.corrections.sat_spec.power_reduction_factor()

    def _opx_setter(self, value):
        self.params.corrections.sat_spec.power_reduction_factor(value)


# ---------------------------------------------------------------------------
# Correction strategies
# ---------------------------------------------------------------------------

class WindowShiftCorrection(Correction):
    name = "window_shift"
    description = "Shift the measurement frequency window by multiples of its original span"
    triggered_by = "fit_quality"

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
        return (f"[{self._last_new_start:.4f}, {self._last_new_end:.4f}] MHz"
                f" (shift={(self._last_new_start - self._original_start):+.1f} MHz)")

    def reset(self) -> None:
        """Restore original window and reset index. Called by higher-level corrections."""
        if self._original_start is not None:
            self.start_param(self._original_start)
            self.end_param(self._original_end)
        self._idx = 0


class IncreaseAveragingCorrection(Correction):
    name = "increase_averaging"
    description = "Increase repetitions and reset window shift"
    triggered_by = "fit_quality"

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


class IncreaseSamplingRateCorrection(Correction):
    name = "increase_sampling_rate"
    description = "Increase frequency step count and reset window shift"
    triggered_by = "fit_quality"

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


class IncreasePowerCorrection(Correction):
    name = "increase_power"
    description = "Increase drive gain and reset window shift"
    triggered_by = "fit_quality"

    def __init__(self, gain_param, window_correction: WindowShiftCorrection,
                 factor_param, max_increases_param):
        self.gain_param = gain_param
        self.window_correction = window_correction
        self.factor_param = factor_param
        self.max_increases_param = max_increases_param
        self._original_gain: float | None = None
        self._count = 0
        self._last_change: str = ""

    def can_apply(self) -> bool:
        return self._count < int(self.max_increases_param())

    def apply(self) -> None:
        if self._original_gain is None:
            self._original_gain = self.gain_param()
        factor = self.factor_param()
        old = self.gain_param()
        new = self._original_gain * (factor ** (self._count + 1))
        self.gain_param(new)
        self._count += 1
        self._last_change = f"gain: {old:.4f} → {new:.4f}"
        self.window_correction.reset()

    def report_output(self) -> str:
        return self._last_change


class ReducePowerCorrection(Correction):
    name = "reduce_power"
    description = "Reduce drive gain to eliminate spurious peaks from over-driving"
    triggered_by = "single_peak"

    def __init__(self, gain_param, factor_param, max_reductions_param):
        self.gain_param = gain_param
        self.factor_param = factor_param
        self.max_reductions_param = max_reductions_param
        self._count = 0
        self._last_change: str = ""

    def can_apply(self) -> bool:
        return self._count < int(self.max_reductions_param())

    def apply(self) -> None:
        factor = self.factor_param()
        old = self.gain_param()
        new = old * factor
        self.gain_param(new)
        self._count += 1
        self._last_change = f"gain: {old:.4f} → {new:.4f}"

    def report_output(self) -> str:
        return self._last_change


# ---------------------------------------------------------------------------
# Synthetic data helper
# ---------------------------------------------------------------------------

@dataclass
class SyntheticSatSpecData:
    fq: float
    delta_fr: float
    f_rabi: float
    gamma1: float
    gamma2: float
    angle: float
    noise_amp: float

    def generate(self, frequencies: ArrayLike) -> ArrayLike:
        """Saturation spec model based on Blais circuit qed eqn 127"""
        signal = 0.5 * self.f_rabi**2 / (
            self.gamma1 * self.gamma2 + (frequencies - self.fq) ** 2 * self.gamma1 / self.gamma2 + self.f_rabi ** 2
        )
        signal_re = signal * np.cos(self.angle) + self.noise_amp * np.random.randn(*frequencies.shape)
        signal_imag = signal * np.sin(self.angle) + self.noise_amp * np.random.randn(*frequencies.shape)
        return signal_re + 1j * signal_imag


# ---------------------------------------------------------------------------
# Operation
# ---------------------------------------------------------------------------

class SaturationSpectroscopy(ProtocolOperation):

    _DUMMY_F_Q = 5e9
    _DUMMY_F_R = 7e9
    _DUMMY_DELTA = _DUMMY_F_R - _DUMMY_F_Q

    _DUMMY_P_IN = 1e-16
    _DUMMY_G = 50e6
    _DUMMY_KAPPA_R = 0.2e6

    _DUMMY_T1 = 50e-6
    _DUMMY_T2 = 50e-6
    _DUMMY_GAMMA_1 = 1 / _DUMMY_T1
    _DUMMY_GAMMA_2 = 1 / (np.pi * _DUMMY_T2)

    _DUMMY_OMEGA = 2 * (_DUMMY_G / _DUMMY_DELTA) * np.sqrt(_DUMMY_KAPPA_R * _DUMMY_P_IN / (h * _DUMMY_F_Q))
    _DUMMY_GAMMA_Q = np.sqrt( (1 / _DUMMY_T2) ** 2 + ((2 * np.pi * _DUMMY_OMEGA) ** 2 * _DUMMY_T1 / _DUMMY_T2) ) / np.pi # blais eq 127

    _DUMMY_NOISE_AMP = 0.05
    _DUMMY_ANGLE = np.pi / 4


    def __init__(self, params):
        super().__init__()
        self.params = params

        self._register_inputs(
            repetitions=Repetition(params),
            steps=SaturationSpecSteps(params),
            start_freq=StartSaturationSpecFrequency(params),
            end_freq=EndSaturationSpecFrequency(params),
            drive_gain=SaturationSpecDriveGain(params),
        )
        self._register_outputs(
            qubit_freq=QubitFrequency(params)
        )

        self._register_correction_params(
            snr_threshold=SNRThreshold(params),
            max_fit_param_error=MaxFitParamError(params),
            max_window_shifts=MaxWindowShifts(params),
            averaging_increase_factor=AveragingIncreaseFactor(params),
            max_averaging_increases=MaxAveragingIncreases(params),
            sampling_increase_factor=SamplingIncreaseFactor(params),
            max_sampling_increases=MaxSamplingIncreases(params),
            max_power_increases=MaxPowerIncreases(params),
            power_increase_factor=PowerIncreaseFactor(params),
            single_peak_threshold=SinglePeakSNRThreshold(params),
            single_peak_max_reductions=SinglePeakMaxPowerReductions(params),
            power_reduction_factor=PowerReductionFactor(params),
        )

        self._window_shift = WindowShiftCorrection(
            self.start_freq, self.end_freq, self.max_window_shifts
        )
        self._increase_averaging = IncreaseAveragingCorrection(
            self.repetitions, self._window_shift,
            self.averaging_increase_factor, self.max_averaging_increases,
        )
        self._increase_sampling = IncreaseSamplingRateCorrection(
            self.steps, self._window_shift,
            self.sampling_increase_factor, self.max_sampling_increases,
        )
        self._increase_power = IncreasePowerCorrection(
            self.drive_gain, self._window_shift,
            self.power_increase_factor, self.max_power_increases,
        )
        self._reduce_power = ReducePowerCorrection(
            self.drive_gain, self.power_reduction_factor, self.single_peak_max_reductions
        )

        self._register_check(
            "fit_quality",
            self._check_fit_quality,
            [self._window_shift, self._increase_power,self._increase_averaging, self._increase_sampling],
        )
        self._register_check(
            "single_peak",
            self._check_single_peak,
            self._reduce_power,
        )

        self._register_success_update(
            self.qubit_freq,
            lambda: self._best_fit_result.params["x0"].value,
        )

        self.condition = "Success if the best-component SNR exceeds threshold and the fit has low parameter errors"

        self.independents = {"frequencies": []}
        self.dependents = {"signal": []}

        self.fit_result = None
        self.snr = None
        self.residuals = None

    def _measure_qick(self) -> Path:
        logger.info("Starting qick saturation spectroscopy measurement")

        sweep = PulseProbeSpectroscopy()
        logger.debug("Sweep created, running measurement")
        loc, da = run_and_save_sweep(sweep, "data", self.name)
        logger.info("Measurement complete")

        return loc

    def _measure_opx(self) -> Path:
        logger.info("Starting opx saturation spectroscopy measurement")
        loc = measure_qubit_ssb_spec_saturation()
        logger.info("Measurement complete")
        return loc

    def _load_data_qick(self):
        data = load_as_xr(self.data_loc)
        rotated = rotate_complex_qubit_data(data)[0]
        self.independents["frequencies"] = rotated["freq"].values
        self.dependents["signal"] = rotated["signal"].values

    def _load_data_opx(self):
        data = load_as_xr(self.data_loc)
        if "repetition" in data.dims:
            data = data.mean("repetition")
        data, _ = rotate_complex_qubit_data(data)
        self.independents["frequencies"] = data["ssb_frequency"].values
        self.dependents["signal"] = data["signal"].values

    def _measure_dummy(self) -> Path:
        """Create synthetic saturation spectroscopy data using a sweep"""
        logger.info("Starting dummy saturation spectroscopy measurement")

        start_f = self.start_freq()
        end_f = self.end_freq()
        num_steps = int(self.steps())

        frequencies = np.linspace(start_f, end_f, num_steps)

        generator = SyntheticSatSpecData(
            fq=self._DUMMY_F_Q,
            delta_fr=self._DUMMY_DELTA,
            f_rabi=self._DUMMY_OMEGA,
            gamma1=self._DUMMY_GAMMA_1,
            gamma2=self._DUMMY_GAMMA_2,
            angle=self._DUMMY_ANGLE,
            noise_amp=self._DUMMY_NOISE_AMP
        )

        sweep = sweep_parameter('frequencies', frequencies, record_as(generator.generate, 'signal'))

        loc, _ = run_and_save_sweep(sweep, "data", self.name)
        logger.info("Dummy saturation spectroscopy measurement complete.")

        return loc

    def _load_data_dummy(self):
        """Load dummy data from disk (same as _load_data_qick)"""
        logger.info("Loading dummy data from disk")
        data = load_as_xr(self.data_loc)
        rotated = rotate_complex_qubit_data(data)[0]
        self.independents["frequencies"] = rotated["frequencies"].values
        self.dependents["signal"] = rotated["signal"].values

    def _fit_lorentzian(self, frequencies, signal, fig_title="") -> tuple:
        fit = Lorentzian(frequencies, signal)
        fit_result = fit.run(fit)
        fit_curve = fit_result.eval()
        residuals = signal - fit_curve
        amp = fit_result.params["A"].value
        noise = np.std(residuals)
        snr = np.abs(amp / (4 * noise))

        fig, ax = plt.subplots()
        ax.set_title(fig_title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Rotated Signal (A.U)")
        ax.plot(frequencies, signal, label="Data")
        ax.plot(frequencies, fit_curve, label="Fit")
        ax.legend()

        return fit_result, residuals, snr, fig

    def analyze(self):
        with DatasetAnalysis(self.data_loc, self.name) as ds:
            self.fit_result, self.residuals, self.snr, fig = self._fit_lorentzian(
                self.independents["frequencies"],
                self.dependents["signal"],
                "Saturation Spectroscopy"
            )
            self._best_fit_result = self.fit_result

            ds.add(
                fit_result=self.fit_result,
                params=serialize_fit_params(self.fit_result.params),
                snr=float(self.snr)
            )

            ds.add_figure(self.name, fig=fig)
            image_path = ds._new_file_path(ds.savefolders[1], self.name, suffix="png")
            self.figure_paths.append(image_path)

    # --- checks (pure assessment) ---

    def _check_fit_quality(self) -> CheckResult:
        threshold = self.snr_threshold()
        snr_passed = self.snr >= threshold

        max_error = self.max_fit_param_error()
        bad_params = []
        for pname, param in self.fit_result.params.items():
            if pname == "of":
                continue
            if param.stderr is None:
                bad_params.append(f"{pname}(no stderr)")
            elif param.value == 0 or abs(param.stderr / param.value) > max_error:
                pct = abs(param.stderr / param.value) * 100 if param.value != 0 else float("inf")
                bad_params.append(f"{pname}({pct:.0f}%)")

        passed = snr_passed and len(bad_params) == 0
        parts = [f"SNR={self.snr:.3f} (threshold={threshold:.3f})"]
        if bad_params:
            parts.append(f"high-error params: {', '.join(bad_params)}")

        return CheckResult("fit_quality", passed, "; ".join(parts))

    def _check_single_peak(self) -> CheckResult:
        # TODO: make sure that the fit frequency is inside the swept range
        residuals = self.residuals
        frequencies = self.independents["frequencies"]

        fit = Lorentzian(frequencies, residuals)
        fit_result = fit.run(fit)
        fit_curve = fit_result.eval()
        residuals_of_residuals = residuals - fit_curve
        amp = fit_result.params["A"].value
        noise = np.std(residuals_of_residuals)
        snr_residual = np.abs(amp / (4 * noise)) if noise > 0 else 0.0

        threshold = self.single_peak_threshold()
        snr_high = snr_residual >= threshold

        # Only flag a real second peak if both SNR is high AND the fit converged well
        max_error = self.max_fit_param_error()
        bad_params = []
        for pname, param in fit_result.params.items():
            if pname == "of":
                continue
            if param.stderr is None:
                bad_params.append(pname)
            elif param.value == 0 or abs(param.stderr / param.value) > max_error:
                bad_params.append(pname)

        fit_converged = len(bad_params) == 0
        multiple_peaks_detected = snr_high and fit_converged
        passed = not multiple_peaks_detected

        description = f"residual_SNR={snr_residual:.3f} (threshold={threshold:.3f})"
        if not passed:
            description += " — multiple peaks detected"
        elif snr_high and not fit_converged:
            description += f" — high residual SNR but fit did not converge ({', '.join(bad_params)}), treating as noise"
        else:
            description += " — single peak confirmed"

        return CheckResult("single_peak", passed, description)

    def correct(self, result: EvaluateResult) -> EvaluateResult:
        figure = self.figure_paths[0] if self.figure_paths else None
        self.figure_paths.clear()

        header = (f"## Saturation Spectroscopy\n"
                  f"Measured saturation spectroscopy for frequencies: "
                  f"{self.start_freq():.3f}–{self.end_freq():.3f} MHz\n"
                  f"Data Path: `{self.data_loc}`\n\n")
        self.report_output.append(header)

        self.report_output.extend([
            f"### Rotated Signal Fit (SNR={self.snr:.3f})\n\n",
            figure,
            f"**Fit Report:**\n```\n{self.fit_result.lmfit_result.fit_report()}\n```\n\n",
        ])

        # Let super() add the check table; no auto-figure since figure_paths is cleared
        result = super().correct(result)

        return result

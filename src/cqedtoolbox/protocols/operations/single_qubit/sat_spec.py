import logging
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")

from labcore.analysis import DatasetAnalysis, Fit
from labcore.analysis.fitfuncs.generic import Lorentzian
from labcore.measurement.storage import run_and_save_sweep
from labcore.data.datadict_storage import datadict_from_hdf5
from labcore.measurement.sweep import sweep_parameter, Sweep
from labcore.measurement.record import record_as
from labcore.data.datagen import Lorentzian as LorentzianDataGen

from labcore.protocols.base import (ProtocolOperation, OperationStatus, serialize_fit_params,
                                    ParamImprovement, CorrectionParameter, CheckResult, Correction,
                                    EvaluateResult)
from cqedtoolbox.protocols.parameters import (
    Repetition,
    SaturationSpecSteps,
    StartSaturationSpecFrequency, EndSaturationSpecFrequency, QubitFrequency,
    SaturationSpecDriveGain,
)
from cqedtoolbox.measurement_lib.qick.single_transmon_v2 import PulseProbeSpectroscopy


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


@dataclass
class MaxFitParamError(CorrectionParameter):
    name: str = field(default="sat_spec_max_fit_param_error", init=False)
    description: str = field(default="Maximum allowed fractional fit parameter error (e.g. 1.0 = 100%)", init=False)

    def _qick_getter(self):
        return self.params.corrections.sat_spec.max_fit_param_error()

    def _qick_setter(self, value):
        self.params.corrections.sat_spec.max_fit_param_error(value)


@dataclass
class MaxWindowShifts(CorrectionParameter):
    name: str = field(default="sat_spec_max_window_shifts", init=False)
    description: str = field(default="Number of ±n window shifts to try", init=False)

    def _qick_getter(self):
        return int(self.params.corrections.sat_spec.max_window_shifts())

    def _qick_setter(self, value):
        self.params.corrections.sat_spec.max_window_shifts(value)


@dataclass
class AveragingIncreaseFactor(CorrectionParameter):
    name: str = field(default="sat_spec_averaging_increase_factor", init=False)
    description: str = field(default="Factor by which to increase repetitions", init=False)

    def _qick_getter(self):
        return self.params.corrections.sat_spec.averaging_factor()

    def _qick_setter(self, value):
        self.params.corrections.sat_spec.averaging_factor(value)


@dataclass
class MaxAveragingIncreases(CorrectionParameter):
    name: str = field(default="sat_spec_max_averaging_increases", init=False)
    description: str = field(default="Maximum number of averaging increases to try", init=False)

    def _qick_getter(self):
        return int(self.params.corrections.sat_spec.max_averaging_increases())

    def _qick_setter(self, value):
        self.params.corrections.sat_spec.max_averaging_increases(value)


@dataclass
class SamplingIncreaseFactor(CorrectionParameter):
    name: str = field(default="sat_spec_sampling_increase_factor", init=False)
    description: str = field(default="Factor by which to increase frequency steps", init=False)

    def _qick_getter(self):
        return self.params.corrections.sat_spec.sampling_factor()

    def _qick_setter(self, value):
        self.params.corrections.sat_spec.sampling_factor(value)


@dataclass
class MaxSamplingIncreases(CorrectionParameter):
    name: str = field(default="sat_spec_max_sampling_increases", init=False)
    description: str = field(default="Maximum number of sampling rate increases to try", init=False)

    def _qick_getter(self):
        return int(self.params.corrections.sat_spec.max_sampling_increases())

    def _qick_setter(self, value):
        self.params.corrections.sat_spec.max_sampling_increases(value)


@dataclass
class MaxPowerIncreases(CorrectionParameter):
    name: str = field(default="sat_spec_max_power_increases", init=False)
    description: str = field(default="Maximum number of drive power increases to try", init=False)

    def _qick_getter(self):
        return int(self.params.corrections.sat_spec.max_power_increases())

    def _qick_setter(self, value):
        self.params.corrections.sat_spec.max_power_increases(value)


@dataclass
class PowerIncreaseFactor(CorrectionParameter):
    name: str = field(default="sat_spec_power_increase_factor", init=False)
    description: str = field(default="Multiplicative factor for increasing drive gain (e.g. 1.1 = +10%)", init=False)

    def _qick_getter(self):
        return self.params.corrections.sat_spec.power_increase_factor()

    def _qick_setter(self, value):
        self.params.corrections.sat_spec.power_increase_factor(value)


@dataclass
class SinglePeakSNRThreshold(CorrectionParameter):
    name: str = field(default="sat_spec_single_peak_snr", init=False)
    description: str = field(default="SNR threshold for detecting a second peak in the fit residuals", init=False)

    def _qick_getter(self):
        return self.params.corrections.sat_spec.single_peak_snr()

    def _qick_setter(self, value):
        self.params.corrections.sat_spec.single_peak_snr(value)


@dataclass
class SinglePeakMaxPowerReductions(CorrectionParameter):
    name: str = field(default="sat_spec_single_peak_max_reductions", init=False)
    description: str = field(default="Maximum number of drive power reductions to try", init=False)

    def _qick_getter(self):
        return int(self.params.corrections.sat_spec.single_peak_max_reductions())

    def _qick_setter(self, value):
        self.params.corrections.sat_spec.single_peak_max_reductions(value)


@dataclass
class PowerReductionFactor(CorrectionParameter):
    name: str = field(default="sat_spec_power_reduction_factor", init=False)
    description: str = field(default="Multiplicative factor for reducing drive gain (e.g. 0.9 = -10%)", init=False)

    def _qick_getter(self):
        return self.params.corrections.sat_spec.power_reduction_factor()

    def _qick_setter(self, value):
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
# Operation
# ---------------------------------------------------------------------------

class SaturationSpectroscopy(ProtocolOperation):

    _DUMMY_F_Q = 5e9
    _DUMMY_GAMMA = 1e6
    _DUMMY_A = 0.5
    _DUMMY_NOISE_AMP = 0.05


    def __init__(self, params):
        super().__init__()

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

        self.fit_result_re = None
        self.fit_result_imag = None
        self.fit_result_mag = None
        self.snr_re = None
        self.snr_imag = None
        self.snr_mag = None
        self.residuals_re = None
        self.residuals_imag = None
        self.residuals_mag = None
        self._best_fit_result = None
        self._winner_key: str | None = None

    def _measure_qick(self) -> Path:
        logger.info("Starting qick saturation spectroscopy measurement")

        sweep = PulseProbeSpectroscopy()
        logger.debug("Sweep created, running measurement")
        loc, da = run_and_save_sweep(sweep, "data", self.name)
        logger.info("Measurement complete")

        return loc

    def _load_data_qick(self):
        path = self.data_loc / "data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)

        self.independents["frequencies"] = data["freq"]["values"]
        self.dependents["signal"] = data["signal"]["values"]

    def _measure_dummy(self) -> Path:
        logger.info("Starting dummy saturation spectroscopy measurement")
        frequencies = np.linspace(self.start_freq(), self.end_freq(), int(self.steps()))
        generator = LorentzianDataGen(x0=self._DUMMY_F_Q, gamma=self._DUMMY_GAMMA, A=self._DUMMY_A, of=0, noise_std=self._DUMMY_NOISE_AMP)
        sweep = sweep_parameter('frequencies', frequencies) * Sweep(record_as(generator.generate(frequencies), 'signal'))
        loc, _ = run_and_save_sweep(sweep, "data", self.name)
        logger.info("Dummy measurement complete")
        return loc

    def _load_data_dummy(self):
        """Load dummy data from disk (same as _load_data_qick)"""
        logger.info("Loading dummy data from disk")
        path = self.data_loc / "data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)

        self.independents["frequencies"] = data["frequencies"]["values"]
        self.dependents["signal"] = data["signal"]["values"]

    def _fit_lorentzian_components(self, frequencies, signal, fig_title="") -> tuple:
        """
        Fit real, imaginary, and magnitude components with Lorentzian fits.
        Returns (fit_result_re, residuals_re, snr_re), (fit_result_imag, ...), (fit_result_mag, ...),
                fig_re, fig_imag, fig_mag
        """
        signal_re = signal.real
        signal_imag = signal.imag
        signal_mag = np.abs(signal)

        # Fit real part
        fit_re = Lorentzian(frequencies, signal_re)
        fit_result_re = fit_re.run(fit_re)
        fit_curve_re = fit_result_re.eval()
        residuals_re = signal_re - fit_curve_re
        amp_re = fit_result_re.params["A"].value
        noise_re = np.std(residuals_re)
        snr_re = np.abs(amp_re / (4 * noise_re))

        # Fit imaginary part
        fit_imag = Lorentzian(frequencies, signal_imag)
        fit_result_imag = fit_imag.run(fit_imag)
        fit_curve_imag = fit_result_imag.eval()
        residuals_imag = signal_imag - fit_curve_imag
        amp_imag = fit_result_imag.params["A"].value
        noise_imag = np.std(residuals_imag)
        snr_imag = np.abs(amp_imag / (4 * noise_imag))

        # Fit magnitude
        fit_mag = Lorentzian(frequencies, signal_mag)
        fit_result_mag = fit_mag.run(fit_mag)
        fit_curve_mag = fit_result_mag.eval()
        residuals_mag = signal_mag - fit_curve_mag
        amp_mag = fit_result_mag.params["A"].value
        noise_mag = np.std(residuals_mag)
        snr_mag = np.abs(amp_mag / (4 * noise_mag))

        # Real plot
        fig_re, ax_re = plt.subplots()
        ax_re.set_title(f"{fig_title} - Real")
        ax_re.set_xlabel("Frequency (Hz)")
        ax_re.set_ylabel("Signal Real (A.U)")
        ax_re.plot(frequencies, signal_re, label="Data")
        ax_re.plot(frequencies, fit_curve_re, label="Fit")
        ax_re.legend()

        # Imaginary plot
        fig_imag, ax_imag = plt.subplots()
        ax_imag.set_title(f"{fig_title} - Imaginary")
        ax_imag.set_xlabel("Frequency (Hz)")
        ax_imag.set_ylabel("Signal Imaginary (A.U)")
        ax_imag.plot(frequencies, signal_imag, label="Data")
        ax_imag.plot(frequencies, fit_curve_imag, label="Fit")
        ax_imag.legend()

        # Magnitude plot
        fig_mag, ax_mag = plt.subplots()
        ax_mag.set_title(f"{fig_title} - Magnitude")
        ax_mag.set_xlabel("Frequency (Hz)")
        ax_mag.set_ylabel("Signal Magnitude (A.U)")
        ax_mag.plot(frequencies, signal_mag, label="Data")
        ax_mag.plot(frequencies, fit_curve_mag, label="Fit")
        ax_mag.legend()

        return (
            (fit_result_re, residuals_re, snr_re),
            (fit_result_imag, residuals_imag, snr_imag),
            (fit_result_mag, residuals_mag, snr_mag),
            fig_re,
            fig_imag,
            fig_mag
        )

    def analyze(self):
        with DatasetAnalysis(self.data_loc, self.name) as ds:
            result_re, result_imag, result_mag, fig_re, fig_imag, fig_mag = self._fit_lorentzian_components(
                self.independents["frequencies"],
                self.dependents["signal"],
                "Saturation Spectroscopy"
            )

            self.fit_result_re, self.residuals_re, self.snr_re = result_re
            self.fit_result_imag, self.residuals_imag, self.snr_imag = result_imag
            self.fit_result_mag, self.residuals_mag, self.snr_mag = result_mag

            # Determine winner (highest SNR) for use in checks and success update
            snr_map = {"re": self.snr_re, "imag": self.snr_imag, "mag": self.snr_mag}
            self._winner_key = max(snr_map, key=lambda k: snr_map[k])
            fit_map = {
                "re": self.fit_result_re,
                "imag": self.fit_result_imag,
                "mag": self.fit_result_mag,
            }
            self._best_fit_result = fit_map[self._winner_key]

            ds.add(
                fit_result_re=self.fit_result_re,
                params_re=serialize_fit_params(self.fit_result_re.params),
                snr_re=float(self.snr_re),
                fit_result_imag=self.fit_result_imag,
                params_imag=serialize_fit_params(self.fit_result_imag.params),
                snr_imag=float(self.snr_imag),
                fit_result_mag=self.fit_result_mag,
                params_mag=serialize_fit_params(self.fit_result_mag.params),
                snr_mag=float(self.snr_mag)
            )

            ds.add_figure(f"{self.name}_real", fig=fig_re)
            image_path_re = ds._new_file_path(ds.savefolders[1], f"{self.name}_real", suffix="png")
            self.figure_paths.append(image_path_re)

            ds.add_figure(f"{self.name}_imag", fig=fig_imag)
            image_path_imag = ds._new_file_path(ds.savefolders[1], f"{self.name}_imag", suffix="png")
            self.figure_paths.append(image_path_imag)

            ds.add_figure(f"{self.name}_mag", fig=fig_mag)
            image_path_mag = ds._new_file_path(ds.savefolders[1], f"{self.name}_mag", suffix="png")
            self.figure_paths.append(image_path_mag)

    # --- checks (pure assessment) ---

    def _check_fit_quality(self) -> CheckResult:
        snr_map = {"re": self.snr_re, "imag": self.snr_imag, "mag": self.snr_mag}
        winner_key = max(snr_map, key=lambda k: snr_map[k])
        winner_snr = snr_map[winner_key]
        winner_fit = {"re": self.fit_result_re, "imag": self.fit_result_imag, "mag": self.fit_result_mag}[winner_key]
        label_map = {"re": "Real", "imag": "Imaginary", "mag": "Magnitude"}

        threshold = self.snr_threshold()
        snr_passed = winner_snr >= threshold

        max_error = self.max_fit_param_error()
        bad_params = []
        for pname, param in winner_fit.params.items():
            if pname == "of":
                continue
            if param.stderr is None:
                bad_params.append(f"{pname}(no stderr)")
            elif param.value == 0 or abs(param.stderr / param.value) > max_error:
                pct = abs(param.stderr / param.value) * 100 if param.value != 0 else float("inf")
                bad_params.append(f"{pname}({pct:.0f}%)")

        passed = snr_passed and len(bad_params) == 0
        parts = [f"best_SNR={winner_snr:.3f} (threshold={threshold:.3f}, component={label_map[winner_key]})"]
        if bad_params:
            parts.append(f"high-error params: {', '.join(bad_params)}")

        return CheckResult("fit_quality", passed, "; ".join(parts))

    def _check_single_peak(self) -> CheckResult:
        residuals_map = {
            "re": self.residuals_re,
            "imag": self.residuals_imag,
            "mag": self.residuals_mag,
        }
        residuals = residuals_map[self._winner_key]
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
        # Pop all figures before super() to control layout
        fig_re = self.figure_paths[0] if len(self.figure_paths) >= 1 else None
        fig_imag = self.figure_paths[1] if len(self.figure_paths) >= 2 else None
        fig_mag = self.figure_paths[2] if len(self.figure_paths) >= 3 else None
        self.figure_paths.clear()

        # Build component report (winner first)
        snr_map = {"re": self.snr_re, "imag": self.snr_imag, "mag": self.snr_mag}
        fig_map = {"re": fig_re, "imag": fig_imag, "mag": fig_mag}
        fit_map = {
            "re": self.fit_result_re,
            "imag": self.fit_result_imag,
            "mag": self.fit_result_mag,
        }
        label_map = {"re": "Real", "imag": "Imaginary", "mag": "Magnitude"}

        sorted_keys = sorted(snr_map, key=lambda k: snr_map[k], reverse=True)
        winner_key = sorted_keys[0]

        header = (f"## Saturation Spectroscopy\n"
                  f"Measured saturation spectroscopy for frequencies: "
                  f"{self.start_freq():.3f}–{self.end_freq():.3f} MHz\n"
                  f"Data Path: `{self.data_loc}`\n\n")
        self.report_output.append(header)

        winner_label = label_map[winner_key]
        self.report_output.extend([
            f"### **{winner_label} Component (SELECTED, SNR={snr_map[winner_key]:.3f})**\n\n",
            fig_map[winner_key],
            f"**Fit Report:**\n```\n{fit_map[winner_key].lmfit_result.fit_report()}\n```\n\n",
        ])

        for k in sorted_keys[1:]:
            self.report_output.extend([
                f"### **{label_map[k]} Component (SNR={snr_map[k]:.3f})**\n\n",
                fig_map[k],
                f"**Fit Report:**\n```\n{fit_map[k].lmfit_result.fit_report()}\n```\n\n",
            ])

        # Let super() add the check table; no auto-figure since figure_paths is cleared
        result = super().correct(result)

        return result
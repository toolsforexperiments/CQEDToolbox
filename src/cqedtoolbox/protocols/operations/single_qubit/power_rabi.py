import logging
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")

from labcore.analysis import DatasetAnalysis
from labcore.analysis.fitfuncs.generic import Cosine
from labcore.measurement.storage import run_and_save_sweep
from labcore.measurement import sweep_parameter, record_as
from labcore.data.datadict_storage import datadict_from_hdf5

from labcore.protocols.base import (
    ProtocolOperation, serialize_fit_params,
    CorrectionParameter, CheckResult, Correction, EvaluateResult,
)
from cqedtoolbox.protocols.parameters import (
    Repetition,
    StartQubitGain,
    EndQubitGain,
    QubitGain,
    NumGainSteps,
    Delay,
)
from cqedtoolbox.measurement_lib.qick.single_transmon_v2 import AmplitudeRabiProgram


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CorrectionParameter subclasses
# ---------------------------------------------------------------------------

@dataclass
class SNRThreshold(CorrectionParameter):
    name: str = field(default="power_rabi_snr_threshold", init=False)
    description: str = field(default="SNR threshold for power rabi quality check", init=False)

    def _qick_getter(self): return self.params.corrections.power_rabi.snr()
    def _qick_setter(self, v): self.params.corrections.power_rabi.snr(v)


@dataclass
class MaxFitParamError(CorrectionParameter):
    name: str = field(default="power_rabi_max_fit_param_error", init=False)
    description: str = field(default="Maximum allowed fractional fit parameter error (e.g. 1.0 = 100%)", init=False)

    def _qick_getter(self): return self.params.corrections.power_rabi.max_fit_param_error()
    def _qick_setter(self, v): self.params.corrections.power_rabi.max_fit_param_error(v)


@dataclass
class AveragingIncreaseFactor(CorrectionParameter):
    name: str = field(default="power_rabi_averaging_factor", init=False)
    description: str = field(default="Factor by which to increase repetitions", init=False)

    def _qick_getter(self): return self.params.corrections.power_rabi.averaging_factor()
    def _qick_setter(self, v): self.params.corrections.power_rabi.averaging_factor(v)


@dataclass
class MaxAveragingIncreases(CorrectionParameter):
    name: str = field(default="power_rabi_max_averaging_increases", init=False)
    description: str = field(default="Maximum number of averaging increases to try", init=False)

    def _qick_getter(self): return int(self.params.corrections.power_rabi.max_averaging_increases())
    def _qick_setter(self, v): self.params.corrections.power_rabi.max_averaging_increases(v)


@dataclass
class SamplingIncreaseFactor(CorrectionParameter):
    name: str = field(default="power_rabi_sampling_factor", init=False)
    description: str = field(default="Factor by which to increase gain steps", init=False)

    def _qick_getter(self): return self.params.corrections.power_rabi.sampling_factor()
    def _qick_setter(self, v): self.params.corrections.power_rabi.sampling_factor(v)


@dataclass
class MaxSamplingIncreases(CorrectionParameter):
    name: str = field(default="power_rabi_max_sampling_increases", init=False)
    description: str = field(default="Maximum number of step count increases to try", init=False)

    def _qick_getter(self): return int(self.params.corrections.power_rabi.max_sampling_increases())
    def _qick_setter(self, v): self.params.corrections.power_rabi.max_sampling_increases(v)


@dataclass
class DelayIncreaseFactor(CorrectionParameter):
    name: str = field(default="power_rabi_delay_factor", init=False)
    description: str = field(default="Factor by which to increase delay between shots", init=False)

    def _qick_getter(self): return self.params.corrections.power_rabi.delay_factor()
    def _qick_setter(self, v): self.params.corrections.power_rabi.delay_factor(v)


@dataclass
class MaxDelayIncreases(CorrectionParameter):
    name: str = field(default="power_rabi_max_delay_increases", init=False)
    description: str = field(default="Maximum number of delay increases to try", init=False)

    def _qick_getter(self): return int(self.params.corrections.power_rabi.max_delay_increases())
    def _qick_setter(self, v): self.params.corrections.power_rabi.max_delay_increases(v)


@dataclass
class GainRangeShrinkFactor(CorrectionParameter):
    name: str = field(default="power_rabi_gain_shrink_factor", init=False)
    description: str = field(default="Factor by which to divide the gain half-span on each shrink", init=False)

    def _qick_getter(self): return self.params.corrections.power_rabi.gain_shrink_factor()
    def _qick_setter(self, v): self.params.corrections.power_rabi.gain_shrink_factor(v)


@dataclass
class MaxGainRangeShrinks(CorrectionParameter):
    name: str = field(default="power_rabi_max_gain_shrinks", init=False)
    description: str = field(default="Maximum number of gain range shrink steps to try", init=False)

    def _qick_getter(self): return int(self.params.corrections.power_rabi.max_gain_shrinks())
    def _qick_setter(self, v): self.params.corrections.power_rabi.max_gain_shrinks(v)


# ---------------------------------------------------------------------------
# Correction subclasses
# ---------------------------------------------------------------------------

class IncreaseAveragingCorrection(Correction):
    name = "increase_averaging"
    description = "Increase number of repetitions"
    triggered_by = "quality_check"

    def __init__(self, reps_param, factor_param, max_increases_param):
        self.reps_param = reps_param
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

    def report_output(self) -> str:
        return self._last_change


class IncreaseStepsCorrection(Correction):
    name = "increase_steps"
    description = "Increase number of gain steps"
    triggered_by = "quality_check"

    def __init__(self, steps_param, factor_param, max_increases_param):
        self.steps_param = steps_param
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

    def report_output(self) -> str:
        return self._last_change


class IncreaseDelayCorrection(Correction):
    name = "increase_delay"
    description = "Increase delay between shots"
    triggered_by = "quality_check"

    def __init__(self, delay_param, factor_param, max_increases_param):
        self.delay_param = delay_param
        self.factor_param = factor_param
        self.max_increases_param = max_increases_param
        self._original_delay: float | None = None
        self._count = 0
        self._last_change: str = ""

    def can_apply(self) -> bool:
        return self._count < int(self.max_increases_param())

    def apply(self) -> None:
        if self._original_delay is None:
            self._original_delay = self.delay_param()
        factor = self.factor_param()
        old = self.delay_param()
        new = self._original_delay * (factor ** (self._count + 1))
        self.delay_param(new)
        self._count += 1
        self._last_change = f"delay: {old} → {new}"

    def report_output(self) -> str:
        return self._last_change


class ShrinkGainRangeCorrection(Correction):
    name = "shrink_gain_range"
    description = "Symmetrically shrink the gain sweep range from both ends"
    triggered_by = "quality_check"

    def __init__(self, start_param, end_param, factor_param, max_shrinks_param):
        self.start_param = start_param
        self.end_param = end_param
        self.factor_param = factor_param
        self.max_shrinks_param = max_shrinks_param
        self._original_center: float | None = None
        self._original_half_span: float | None = None
        self._count = 0
        self._last_new_start: float | None = None
        self._last_new_end: float | None = None

    def can_apply(self) -> bool:
        return self._count < int(self.max_shrinks_param())

    def apply(self) -> None:
        if self._original_center is None:
            start = self.start_param()
            end = self.end_param()
            self._original_center = (start + end) / 2
            self._original_half_span = (end - start) / 2
        factor = self.factor_param()
        half_span = self._original_half_span / (factor ** (self._count + 1))
        self._last_new_start = self._original_center - half_span
        self._last_new_end = self._original_center + half_span
        self.start_param(self._last_new_start)
        self.end_param(self._last_new_end)
        self._count += 1

    def report_output(self) -> str:
        if self._last_new_start is None:
            return ""
        return f"gain range: [{self._last_new_start:.3f}, {self._last_new_end:.3f}]"


# ---------------------------------------------------------------------------
# Synthetic data helper
# ---------------------------------------------------------------------------

@dataclass
class SyntheticPowerRabiData:
    pi_amp: float
    noise_amp: float

    def generate(self, gains: float) -> np.complex128:
        signal = (np.cos(2 * np.pi * gains / (2 * self.pi_amp)) + 2) - 1j * (np.cos(2 * np.pi * gains / (2 * self.pi_amp)) + 2)
        noise = self.noise_amp * (np.random.randn() + 1j * np.random.randn())
        return signal + noise


# ---------------------------------------------------------------------------
# Operation
# ---------------------------------------------------------------------------

class PowerRabi(ProtocolOperation):

    _SIM_PI_AMP = 0.5
    _SIM_NOISE_AMP = 0.05

    def __init__(self, params):
        super().__init__()

        self._register_inputs(
            repetitions=Repetition(params),
            start_gain=StartQubitGain(params),
            end_gain=EndQubitGain(params),
            steps_gain=NumGainSteps(params),
            delay=Delay(params),
        )
        self._register_outputs(
            qubit_gain=QubitGain(params)
        )

        self._register_correction_params(
            snr_threshold=SNRThreshold(params),
            max_fit_param_error=MaxFitParamError(params),
            averaging_increase_factor=AveragingIncreaseFactor(params),
            max_averaging_increases=MaxAveragingIncreases(params),
            sampling_increase_factor=SamplingIncreaseFactor(params),
            max_sampling_increases=MaxSamplingIncreases(params),
            delay_increase_factor=DelayIncreaseFactor(params),
            max_delay_increases=MaxDelayIncreases(params),
            gain_range_shrink_factor=GainRangeShrinkFactor(params),
            max_gain_range_shrinks=MaxGainRangeShrinks(params),
        )

        self._increase_averaging = IncreaseAveragingCorrection(
            self.repetitions,
            self.averaging_increase_factor,
            self.max_averaging_increases,
        )
        self._increase_steps = IncreaseStepsCorrection(
            self.steps_gain,
            self.sampling_increase_factor,
            self.max_sampling_increases,
        )
        self._increase_delay = IncreaseDelayCorrection(
            self.delay,
            self.delay_increase_factor,
            self.max_delay_increases,
        )
        self._shrink_gain_range = ShrinkGainRangeCorrection(
            self.start_gain,
            self.end_gain,
            self.gain_range_shrink_factor,
            self.max_gain_range_shrinks,
        )

        self._register_check(
            "quality_check",
            self._check_quality,
            [self._increase_averaging, self._increase_steps,
             self._increase_delay, self._shrink_gain_range],
        )

        self._register_success_update(
            self.qubit_gain,
            lambda: 1 / (2 * self._winner_fit.params["f"].value),
        )

        self.independents = {"gains": []}
        self.dependents = {"signal": []}

        self.fit_result_re = None
        self.fit_result_imag = None
        self.fit_result_mag = None
        self.snr_re = None
        self.snr_imag = None
        self.snr_mag = None
        self._winner_fit = None
        self._winner_snr = None
        self._winner_key = None
        self._winner_name = None
        self._sorted_components = None

    def _measure_qick(self) -> Path:
        logger.info("Starting qick power rabi measurement")

        sweep = AmplitudeRabiProgram()
        logger.debug("Sweep created, running measurement")
        loc, da = run_and_save_sweep(sweep, "data", self.name)
        logger.info("Measurement complete")

        return loc

    def _measure_dummy(self):
        logger.info("Starting dummy power rabi measurement")
        gains = np.linspace(self.start_gain(), self.end_gain(), int(self.steps_gain()))
        generator = SyntheticPowerRabiData(
            pi_amp = self._SIM_PI_AMP,
            noise_amp = self._SIM_NOISE_AMP
        )

        sweep = sweep_parameter("gains", gains, record_as(generator.generate, "signal"))
        loc, _ = run_and_save_sweep(sweep, "data", self.name)

        logger.info("Dummy measurement complete")
        return loc

    def _load_data_qick(self):
        path = self.data_loc / "data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)

        self.independents["gains"] = data["gain"]["values"]
        self.dependents["signal"] = data["signal"]["values"]

    def _load_data_dummy(self):
        path = self.data_loc / "data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)

        self.independents["gains"] = data["gains"]["values"]
        self.dependents["signal"] = data["signal"]["values"]

    def _fit_cosine_components(self, gains, signal, fig_title="") -> tuple:
        """
        Fit real, imaginary, and magnitude components with Cosine fits.
        Returns (fit_result_re, fit_result_imag, fit_result_mag, fig_re, fig_imag, fig_mag)
        """
        signal_re = signal.real
        signal_imag = signal.imag
        signal_mag = np.abs(signal)

        # Fit real part
        fit_re = Cosine(gains, signal_re)
        fit_result_re = fit_re.run(fit_re)
        fit_curve_re = fit_result_re.eval()
        residuals_re = signal_re - fit_curve_re
        amp_re = fit_result_re.params["A"].value
        noise_re = np.std(residuals_re)
        snr_re = np.abs(amp_re / (4 * noise_re))

        # Fit imaginary part
        fit_imag = Cosine(gains, signal_imag)
        fit_result_imag = fit_imag.run(fit_imag)
        fit_curve_imag = fit_result_imag.eval()
        residuals_imag = signal_imag - fit_curve_imag
        amp_imag = fit_result_imag.params["A"].value
        noise_imag = np.std(residuals_imag)
        snr_imag = np.abs(amp_imag / (4 * noise_imag))

        # Fit magnitude
        fit_mag = Cosine(gains, signal_mag)
        fit_result_mag = fit_mag.run(fit_mag)
        fit_curve_mag = fit_result_mag.eval()
        residuals_mag = signal_mag - fit_curve_mag
        amp_mag = fit_result_mag.params["A"].value
        noise_mag = np.std(residuals_mag)
        snr_mag = np.abs(amp_mag / (4 * noise_mag))

        # Create three separate figures
        fig_re, ax_re = plt.subplots()
        ax_re.set_title(f"{fig_title} - Real")
        ax_re.set_xlabel("Gain (A.U)")
        ax_re.set_ylabel("Signal Real (A.U)")
        ax_re.plot(gains, signal_re, label="Data")
        ax_re.plot(gains, fit_curve_re, label="Fit")
        ax_re.legend()

        fig_imag, ax_imag = plt.subplots()
        ax_imag.set_title(f"{fig_title} - Imaginary")
        ax_imag.set_xlabel("Gain (A.U)")
        ax_imag.set_ylabel("Signal Imaginary (A.U)")
        ax_imag.plot(gains, signal_imag, label="Data")
        ax_imag.plot(gains, fit_curve_imag, label="Fit")
        ax_imag.legend()

        fig_mag, ax_mag = plt.subplots()
        ax_mag.set_title(f"{fig_title} - Magnitude")
        ax_mag.set_xlabel("Gain (A.U)")
        ax_mag.set_ylabel("Signal Magnitude (A.U)")
        ax_mag.plot(gains, signal_mag, label="Data")
        ax_mag.plot(gains, fit_curve_mag, label="Fit")
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
            result_re, result_imag, result_mag, fig_re, fig_imag, fig_mag = self._fit_cosine_components(
                self.independents["gains"],
                self.dependents["signal"],
                "Power Rabi"
            )

            self.fit_result_re, residuals_re, self.snr_re = result_re
            self.fit_result_imag, residuals_imag, self.snr_imag = result_imag
            self.fit_result_mag, residuals_mag, self.snr_mag = result_mag

            # Determine winner (highest SNR) for check and success update
            snr_dict = {
                "Real":      (self.snr_re,   self.fit_result_re,   "re"),
                "Imaginary": (self.snr_imag, self.fit_result_imag, "imag"),
                "Magnitude": (self.snr_mag,  self.fit_result_mag,  "mag"),
            }
            self._sorted_components = sorted(snr_dict.items(), key=lambda x: x[1][0], reverse=True)
            self._winner_name, (self._winner_snr, self._winner_fit, self._winner_key) = self._sorted_components[0]

            # Save all fit results
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

    def _check_quality(self) -> CheckResult:
        threshold = self.snr_threshold()
        snr_passed = self._winner_snr >= threshold

        max_error = self.max_fit_param_error()
        bad_params = []
        for pname, param in self._winner_fit.params.items():
            if param.stderr is None:
                bad_params.append(f"{pname}(no stderr)")
            elif param.value == 0 or abs(param.stderr / param.value) > max_error:
                pct = abs(param.stderr / param.value) * 100 if param.value != 0 else float("inf")
                bad_params.append(f"{pname}({pct:.0f}%)")

        passed = snr_passed and len(bad_params) == 0
        parts = [f"winner={self._winner_name}, SNR={self._winner_snr:.3f} (threshold={threshold:.3f})"]
        if bad_params:
            parts.append(f"high-error params: {', '.join(bad_params)}")
        return CheckResult("quality_check", passed, "; ".join(parts))

    def correct(self, result: EvaluateResult) -> EvaluateResult:
        # Pop all figures before super() auto-appends the last one
        fig_re   = self.figure_paths.pop(0) if len(self.figure_paths) >= 3 else None
        fig_imag = self.figure_paths.pop(0) if self.figure_paths else None
        fig_mag  = self.figure_paths.pop(0) if self.figure_paths else None
        self.figure_paths.clear()  # prevent auto-append

        plot_map = {"re": fig_re, "imag": fig_imag, "mag": fig_mag}

        self.report_output.append(
            f"## Power Rabi\n"
            f"Gain range: {self.start_gain():.3f}–{self.end_gain():.3f}, "
            f"SNR threshold: {self.snr_threshold():.3f}\n"
            f"Data Path: `{self.data_loc}`\n\n"
        )

        for i, (comp_name, (comp_snr, comp_fit, comp_key)) in enumerate(self._sorted_components):
            tag = "(SELECTED)" if i == 0 else "(NOT SELECTED)"
            self.report_output.append(f"### **{comp_name} Component {tag}**\n")
            if plot_map[comp_key]:
                self.report_output.append(plot_map[comp_key])
            self.report_output.append(
                f"SNR={comp_snr:.3f}\n\n"
                f"**Fit Report:**\n```\n{str(comp_fit.lmfit_result.fit_report())}\n```\n\n"
            )

        result = super().correct(result)   # adds check table + success update line
        return result
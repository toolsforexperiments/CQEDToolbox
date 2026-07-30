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
from labcore.data.datadict_storage import datadict_from_hdf5, load_as_xr

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
from cqedtoolbox.measurement_lib.opx.advanced.qubit_tuneup import measure_power_rabi
from cqedtoolbox.measurement_lib.qick.single_transmon_v2 import AmplitudeRabiProgram
from cqedtoolbox.readout.qubit_readout import rotate_complex_qubit_data


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
    def _opx_getter(self): return self.params.corrections.power_rabi.snr()
    def _opx_setter(self, v): self.params.corrections.power_rabi.snr(v)


@dataclass
class MaxFitParamError(CorrectionParameter):
    name: str = field(default="power_rabi_max_fit_param_error", init=False)
    description: str = field(default="Maximum allowed fractional fit parameter error (e.g. 1.0 = 100%)", init=False)

    def _qick_getter(self): return self.params.corrections.power_rabi.max_fit_param_error()
    def _qick_setter(self, v): self.params.corrections.power_rabi.max_fit_param_error(v)
    def _opx_getter(self): return self.params.corrections.power_rabi.max_fit_param_error()
    def _opx_setter(self, v): self.params.corrections.power_rabi.max_fit_param_error(v)


@dataclass
class AveragingIncreaseFactor(CorrectionParameter):
    name: str = field(default="power_rabi_averaging_factor", init=False)
    description: str = field(default="Factor by which to increase repetitions", init=False)

    def _qick_getter(self): return self.params.corrections.power_rabi.averaging_factor()
    def _qick_setter(self, v): self.params.corrections.power_rabi.averaging_factor(v)
    def _opx_getter(self): return self.params.corrections.power_rabi.averaging_factor()
    def _opx_setter(self, v): self.params.corrections.power_rabi.averaging_factor(v)


@dataclass
class MaxAveragingIncreases(CorrectionParameter):
    name: str = field(default="power_rabi_max_averaging_increases", init=False)
    description: str = field(default="Maximum number of averaging increases to try", init=False)

    def _qick_getter(self): return int(self.params.corrections.power_rabi.max_averaging_increases())
    def _qick_setter(self, v): self.params.corrections.power_rabi.max_averaging_increases(v)
    def _opx_getter(self): return int(self.params.corrections.power_rabi.max_averaging_increases())
    def _opx_setter(self, v): self.params.corrections.power_rabi.max_averaging_increases(v)


@dataclass
class SamplingIncreaseFactor(CorrectionParameter):
    name: str = field(default="power_rabi_sampling_factor", init=False)
    description: str = field(default="Factor by which to increase gain steps", init=False)

    def _qick_getter(self): return self.params.corrections.power_rabi.sampling_factor()
    def _qick_setter(self, v): self.params.corrections.power_rabi.sampling_factor(v)
    def _opx_getter(self): return self.params.corrections.power_rabi.sampling_factor()
    def _opx_setter(self, v): self.params.corrections.power_rabi.sampling_factor(v)


@dataclass
class MaxSamplingIncreases(CorrectionParameter):
    name: str = field(default="power_rabi_max_sampling_increases", init=False)
    description: str = field(default="Maximum number of step count increases to try", init=False)

    def _qick_getter(self): return int(self.params.corrections.power_rabi.max_sampling_increases())
    def _qick_setter(self, v): self.params.corrections.power_rabi.max_sampling_increases(v)
    def _opx_getter(self): return int(self.params.corrections.power_rabi.max_sampling_increases())
    def _opx_setter(self, v): self.params.corrections.power_rabi.max_sampling_increases(v)


@dataclass
class DelayIncreaseFactor(CorrectionParameter):
    name: str = field(default="power_rabi_delay_factor", init=False)
    description: str = field(default="Factor by which to increase delay between shots", init=False)

    def _qick_getter(self): return self.params.corrections.power_rabi.delay_factor()
    def _qick_setter(self, v): self.params.corrections.power_rabi.delay_factor(v)
    def _opx_getter(self): return self.params.corrections.power_rabi.delay_factor()
    def _opx_setter(self, v): self.params.corrections.power_rabi.delay_factor(v)


@dataclass
class MaxDelayIncreases(CorrectionParameter):
    name: str = field(default="power_rabi_max_delay_increases", init=False)
    description: str = field(default="Maximum number of delay increases to try", init=False)

    def _qick_getter(self): return int(self.params.corrections.power_rabi.max_delay_increases())
    def _qick_setter(self, v): self.params.corrections.power_rabi.max_delay_increases(v)
    def _opx_getter(self): return int(self.params.corrections.power_rabi.max_delay_increases())
    def _opx_setter(self, v): self.params.corrections.power_rabi.max_delay_increases(v)


@dataclass
class GainRangeShrinkFactor(CorrectionParameter):
    name: str = field(default="power_rabi_gain_shrink_factor", init=False)
    description: str = field(default="Factor by which to divide the gain half-span on each shrink", init=False)

    def _qick_getter(self): return self.params.corrections.power_rabi.gain_shrink_factor()
    def _qick_setter(self, v): self.params.corrections.power_rabi.gain_shrink_factor(v)
    def _opx_getter(self): return self.params.corrections.power_rabi.gain_shrink_factor()
    def _opx_setter(self, v): self.params.corrections.power_rabi.gain_shrink_factor(v)


@dataclass
class MaxGainRangeShrinks(CorrectionParameter):
    name: str = field(default="power_rabi_max_gain_shrinks", init=False)
    description: str = field(default="Maximum number of gain range shrink steps to try", init=False)

    def _qick_getter(self): return int(self.params.corrections.power_rabi.max_gain_shrinks())
    def _qick_setter(self, v): self.params.corrections.power_rabi.max_gain_shrinks(v)
    def _opx_getter(self): return int(self.params.corrections.power_rabi.max_gain_shrinks())
    def _opx_setter(self, v): self.params.corrections.power_rabi.max_gain_shrinks(v)


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
        self.params = params

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
            lambda: 1 / (2 * self.fit_result.params["f"].value),
        )

        self.independents = {"gains": []}
        self.dependents = {"signal": []}

        self.fit_result = None
        self.residuals = None
        self.snr = None

    def _measure_qick(self) -> Path:
        logger.info("Starting qick power rabi measurement")

        sweep = AmplitudeRabiProgram()
        logger.debug("Sweep created, running measurement")
        loc, da = run_and_save_sweep(sweep, "data", self.name)
        logger.info("Measurement complete")

        return loc

    def _measure_opx(self) -> Path:
        logger.info("Starting opx power rabi measurement")
        loc = measure_power_rabi()
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
        data = load_as_xr(self.data_loc)
        rotated = rotate_complex_qubit_data(data)[0]
        self.independents["gains"] = rotated["gain"].values
        self.dependents["signal"] = rotated["signal"].values

    def _load_data_opx(self):
        data = load_as_xr(self.data_loc)
        if "repetition" in data.dims:
            data = data.mean("repetition")
        data, _ = rotate_complex_qubit_data(data)
        self.independents["gains"] = data["amplitude"].values
        self.dependents["signal"] = data["signal"].values

    def _load_data_dummy(self):
        data = load_as_xr(self.data_loc)
        rotated = rotate_complex_qubit_data(data)[0]
        self.independents["gains"] = rotated["gains"].values
        self.dependents["signal"] = rotated["signal"].values

    def _fit_cosine(self, gains, signal, fig_title="") -> tuple:
        fit = Cosine(gains, signal)
        fit_result = fit.run(fit)
        fit_curve = fit_result.eval()
        residuals = signal - fit_curve
        amp = fit_result.params["A"].value
        noise = np.std(residuals)
        snr = np.abs(amp / (4 * noise))

        fig, ax = plt.subplots()
        ax.set_title(fig_title)
        ax.set_xlabel("Gain (A.U)")
        ax.set_ylabel("Rotated Signal (A.U)")
        ax.plot(gains, signal, label="Data")
        ax.plot(gains, fit_curve, label="Fit")
        ax.legend()

        return fit_result, residuals, snr, fig

    def analyze(self):
        with DatasetAnalysis(self.data_loc, self.name) as ds:
            self.fit_result, self.residuals, self.snr, fig = self._fit_cosine(
                self.independents["gains"],
                self.dependents["signal"],
                "Power Rabi"
            )

            # Save all fit results
            ds.add(
                fit_result=self.fit_result,
                params=serialize_fit_params(self.fit_result.params),
                snr=float(self.snr)
            )

            ds.add_figure(self.name, fig=fig)
            image_path = ds._new_file_path(ds.savefolders[1], self.name, suffix="png")
            self.figure_paths.append(image_path)

    def _check_quality(self) -> CheckResult:
        threshold = self.snr_threshold()
        snr_passed = self.snr >= threshold

        max_error = self.max_fit_param_error()
        bad_params = []
        for pname, param in self.fit_result.params.items():
            if param.stderr is None:
                bad_params.append(f"{pname}(no stderr)")
            elif param.value == 0 or abs(param.stderr / param.value) > max_error:
                pct = abs(param.stderr / param.value) * 100 if param.value != 0 else float("inf")
                bad_params.append(f"{pname}({pct:.0f}%)")

        passed = snr_passed and len(bad_params) == 0
        parts = [f"SNR={self.snr:.3f} (threshold={threshold:.3f})"]
        if bad_params:
            parts.append(f"high-error params: {', '.join(bad_params)}")
        return CheckResult("quality_check", passed, "; ".join(parts))

    def correct(self, result: EvaluateResult) -> EvaluateResult:
        figure = self.figure_paths[0] if self.figure_paths else None
        self.figure_paths.clear()

        self.report_output.append(
            f"## Power Rabi\n"
            f"Gain range: {self.start_gain():.3f}–{self.end_gain():.3f}, "
            f"SNR threshold: {self.snr_threshold():.3f}\n"
            f"Data Path: `{self.data_loc}`\n\n"
        )

        self.report_output.append("### Rotated Signal Fit\n")
        if figure:
            self.report_output.append(figure)
        self.report_output.append(
            f"SNR={self.snr:.3f}\n\n"
            f"**Fit Report:**\n```\n{str(self.fit_result.lmfit_result.fit_report())}\n```\n\n"
        )

        result = super().correct(result)   # adds check table + success update line
        return result

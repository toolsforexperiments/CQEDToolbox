import logging
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

from labcore.analysis import DatasetAnalysis, FitResult
from labcore.measurement.storage import run_and_save_sweep
from labcore.data.datadict_storage import datadict_from_hdf5
from labcore.measurement import sweep_parameter, record_as, Sweep
from labcore.data.datagen import HangerResonator

from labcore.protocols.base import (ProtocolOperation, OperationStatus, serialize_fit_params,
                                    ParamImprovement, CorrectionParameter, CheckResult, Correction)
from cqedtoolbox.protocols.parameters import (Repetition,
                                              ResonatorSpecSteps, ReadoutGain, ReadoutLength, StartReadoutFrequency,
                                              EndReadoutFrequency, ReadoutFrequency)
from cqedtoolbox.measurement_lib.qick.single_transmon_v2 import FreqSweepProgram

from cqedtoolbox.fitfuncs.resonators import HangerResponseBruno


logger = logging.getLogger(__name__)


@dataclass
class UnwindAndFitRet:
    signal_unwind: ArrayLike
    magnitude: ArrayLike
    phase: ArrayLike
    fit_curve: ArrayLike
    fit_result: FitResult
    residuals: ArrayLike
    snr: float
    fig: plt.Figure
    ax: plt.Axes



@dataclass
class SNRThreshold(CorrectionParameter):
    name: str = field(default="resonator_spec_SNR_threshold", init=False)
    description: str = field(default="SNR threshold", init=False)

    def _qick_getter(self):
        return self.params.corrections.res_spec.snr()

    def _qick_setter(self, value):
        self.params.corrections.res_spec.snr(value)


@dataclass
class MaxWindowShifts(CorrectionParameter):
    name: str = field(default="res_spec_max_window_shifts", init=False)
    description: str = field(default="Number of ±n window shifts to try", init=False)

    def _qick_getter(self):
        return int(self.params.corrections.res_spec.max_window_shifts())

    def _qick_setter(self, value):
        self.params.corrections.res_spec.max_window_shifts(value)


@dataclass
class SamplingIncreaseFactor(CorrectionParameter):
    name: str = field(default="res_spec_sampling_increase_factor", init=False)
    description: str = field(default="Factor by which to increase frequency steps", init=False)

    def _qick_getter(self):
        return self.params.corrections.res_spec.sampling_factor()

    def _qick_setter(self, value):
        self.params.corrections.res_spec.sampling_factor(value)


@dataclass
class MaxSamplingIncreases(CorrectionParameter):
    name: str = field(default="res_spec_max_sampling_increases", init=False)
    description: str = field(default="Maximum number of sampling rate increases to try", init=False)

    def _qick_getter(self):
        return int(self.params.corrections.res_spec.max_sampling_increases())

    def _qick_setter(self, value):
        self.params.corrections.res_spec.max_sampling_increases(value)


@dataclass
class AveragingIncreaseFactor(CorrectionParameter):
    name: str = field(default="res_spec_averaging_increase_factor", init=False)
    description: str = field(default="Factor by which to increase repetitions", init=False)

    def _qick_getter(self):
        return self.params.corrections.res_spec.averaging_factor()

    def _qick_setter(self, value):
        self.params.corrections.res_spec.averaging_factor(value)


@dataclass
class MaxAveragingIncreases(CorrectionParameter):
    name: str = field(default="res_spec_max_averaging_increases", init=False)
    description: str = field(default="Maximum number of averaging increases to try", init=False)

    def _qick_getter(self):
        return int(self.params.corrections.res_spec.max_averaging_increases())

    def _qick_setter(self, value):
        self.params.corrections.res_spec.max_averaging_increases(value)


@dataclass
class MaxFitParamError(CorrectionParameter):
    name: str = field(default="res_spec_max_fit_param_error", init=False)
    description: str = field(default="Maximum allowed fractional fit parameter error (e.g. 1.0 = 100%)", init=False)

    def _qick_getter(self):
        return self.params.corrections.res_spec.max_fit_param_error()

    def _qick_setter(self, value):
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


class ResonatorSpectroscopy(ProtocolOperation):

    _SIM_F0 = 7e9
    _SIM_QI = 20e3
    _SIM_QC = 20e3
    _SIM_A = 4.0
    _SIM_PHI = 0.0
    _SIM_NOISE_AMP = 0.05

    
    def __init__(self, params):
        super().__init__()

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

        self._register_success_update(
            self.readout_freq,
            lambda: self.fit_result.params["f_0"].value,
        )

        self._register_success_update(
            self.start_frequency,
            lambda: self.fit_result.params["f_0"].value - 5,
        )

        self._register_success_update(
            self.end_frequency,
            lambda: self.fit_result.params["f_0"].value + 5,
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
    
    def _measure_dummy(self):
        logger.info("Starting dummy resonator spectroscopy measurement")
        frequencies = np.linspace(self.start_frequency(), self.end_frequency(), int(self.steps())) + self.readout_freq()
        generator = HangerResonator(f0=self._SIM_F0, Qc=self._SIM_QC, Qi=self._SIM_QI, A=self._SIM_A, phi=self._SIM_PHI, noise_std=self._SIM_NOISE_AMP)
        sweep = sweep_parameter("frequencies", frequencies) * Sweep(record_as(generator.generate(frequencies), "signal"))
        loc, _ = run_and_save_sweep(sweep, "data", self.name)
        logger.info("Dummy measurement complete")
        return loc

    @staticmethod
    def add_mag_and_unwind_and_fit(frequencies, signal_raw, fig_title="") -> UnwindAndFitRet:
        phase_unwrap = np.unwrap(np.angle(signal_raw))
        phase_slope = np.polyfit(frequencies, phase_unwrap, 1)[0]

        signal_unwind = signal_raw * np.exp(-1j * frequencies * phase_slope)
        magnitude = np.abs(signal_raw)
        phase = np.arctan2(signal_unwind.imag, signal_unwind.real)

        fit = HangerResponseBruno(frequencies, signal_unwind)
        fit_result = fit.run(fit)
        fit_curve = fit_result.eval()
        residuals = signal_unwind - fit_curve

        amp = fit_result.params["A"].value
        noise = np.std(residuals)
        snr = np.abs(amp / (4 * noise))

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
            snr=snr,
            fig=fig,
            ax=ax,
        )

        return ret

    def _load_data_qick(self):
        path = self.data_loc/"data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)

        self.independents["frequencies"] = data["freq"]["values"]
        self.dependents["signal"] = data["signal"]["values"]

    def _load_data_dummy(self):
        path = self.data_loc/"data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)

        self.independents["frequencies"] = data["frequencies"]["values"]
        self.dependents["signal"] = data["signal"]["values"]

    def analyze(self):
        with DatasetAnalysis(self.data_loc, self.name) as ds:
            ret = self.add_mag_and_unwind_and_fit(self.independents["frequencies"],
                                                  self.dependents["signal"],
                                                  "Resonator Spectroscopy")

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


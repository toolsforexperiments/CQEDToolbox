import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")

from labcore.analysis import DatasetAnalysis
from labcore.analysis.fitfuncs.generic import ExponentiallyDecayingSine
from labcore.measurement.storage import run_and_save_sweep
from labcore.measurement.sweep import sweep_parameter
from labcore.measurement.record import record_as
from labcore.data.datadict_storage import datadict_from_hdf5

from labcore.protocols.base import (
    ProtocolOperation, serialize_fit_params,
    CorrectionParameter, CheckResult, Correction, EvaluateResult,
)
from cqedtoolbox.protocols.parameters import (
    Repetition,
    T2RSteps,
    QubitGain,
    ReadoutGain,
    ReadoutLength,
    T2R,
    NEchos
)
from cqedtoolbox.measurement_lib.qick.single_transmon_v2 import T2RProgram


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CorrectionParameter subclasses
# ---------------------------------------------------------------------------

@dataclass
class SNRMinThreshold(CorrectionParameter):
    name: str = field(default="t2r_snr_min_threshold", init=False)
    description: str = field(default="Minimum SNR for a valid T2R fit component", init=False)

    def _qick_getter(self): return self.params.corrections.t2r.snr_min()
    def _qick_setter(self, v): self.params.corrections.t2r.snr_min(v)


@dataclass
class MaxFitParamError(CorrectionParameter):
    name: str = field(default="t2r_max_fit_param_error", init=False)
    description: str = field(default="Maximum allowed fractional fit parameter error (e.g. 1.0 = 100%)", init=False)

    def _qick_getter(self): return self.params.corrections.t2r.max_fit_param_error()
    def _qick_setter(self, v): self.params.corrections.t2r.max_fit_param_error(v)


@dataclass
class AveragingIncreaseFactor(CorrectionParameter):
    name: str = field(default="t2r_averaging_factor", init=False)
    description: str = field(default="Factor by which to increase repetitions", init=False)

    def _qick_getter(self): return self.params.corrections.t2r.averaging_factor()
    def _qick_setter(self, v): self.params.corrections.t2r.averaging_factor(v)


@dataclass
class MaxAveragingIncreases(CorrectionParameter):
    name: str = field(default="t2r_max_averaging_increases", init=False)
    description: str = field(default="Maximum number of averaging increases to try", init=False)

    def _qick_getter(self): return int(self.params.corrections.t2r.max_averaging_increases())
    def _qick_setter(self, v): self.params.corrections.t2r.max_averaging_increases(v)


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


# ---------------------------------------------------------------------------
# Operation
# ---------------------------------------------------------------------------

class T2ROperation(ProtocolOperation):

    _SIM_T2R = 10.0
    _SIM_DETUNING = 0.05
    _SIM_AMP = 0.5
    _SIM_NOISE_AMP = 0.02

    def __init__(self, params):
        super().__init__()

        self.params = params

        self._register_inputs(
            repetitions=Repetition(params),
            steps=T2RSteps(params),
            qubit_gain=QubitGain(params),
            readout_gain=ReadoutGain(params),
            readout_length=ReadoutLength(params),
            n_echos=NEchos(params)
        )
        self._register_outputs(
            t2r=T2R(params)
        )

        self._register_correction_params(
            snr_min_threshold=SNRMinThreshold(params),
            max_fit_param_error=MaxFitParamError(params),
            averaging_increase_factor=AveragingIncreaseFactor(params),
            max_averaging_increases=MaxAveragingIncreases(params),
        )

        self._increase_averaging = IncreaseAveragingCorrection(
            self.repetitions,
            self.averaging_increase_factor,
            self.max_averaging_increases,
        )

        self._register_check("quality_check", self._check_quality, self._increase_averaging)

        self._register_success_update(self.t2r, lambda: self._winner_fit.params["tau"].value)

        self.independents = {"delays": []}
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

    def _measure_dummy(self) -> Path:
        logger.info("Starting dummy T2 Ramsey measurement")
        delays = np.linspace(0, 5 * self._SIM_T2R, int(self.steps()))
        signal_gen = lambda delays: (self._SIM_AMP * np.exp(-delays / self._SIM_T2R) * np.exp(2j * np.pi * self._SIM_DETUNING * delays)
                  + self._SIM_NOISE_AMP * (np.random.randn() + 1j * np.random.randn()))
        sweep = sweep_parameter("delays", delays, record_as(signal_gen, "signal"))
        loc, _ = run_and_save_sweep(sweep, "data", self.name)
        logger.info("Dummy measurement complete")
        return loc

    def _load_data_dummy(self):
        path = self.data_loc / "data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)
        self.independents["delays"] = data["delays"]["values"]
        self.dependents["signal"] = data["signal"]["values"]

    def _measure_qick(self) -> Path:
        logger.info("Starting qick T2 Ramsey measurement")
        sweep = T2RProgram()
        logger.debug("Sweep created, running measurement")
        loc, da = run_and_save_sweep(sweep, "data", self.name)
        logger.info("Measurement complete")
        return loc

    def _load_data_qick(self):
        path = self.data_loc / "data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)

        self.independents["delays"] = data["t"]["values"]
        self.dependents["signal"] = data["signal"]["values"]

    def _fit_exponentially_decaying_sine_components(self, delays, signal, fig_title="") -> tuple:
        """
        Fit real, imaginary, and magnitude components with ExponentiallyDecayingSine fits.
        Returns (fit_result_re, fit_result_imag, fit_result_mag, fig_re, fig_imag, fig_mag)
        """
        signal_re = signal.real
        signal_imag = signal.imag
        signal_mag = np.abs(signal)

        # Fit real part
        fit_re = ExponentiallyDecayingSine(delays, signal_re)
        fit_result_re = fit_re.run(fit_re)
        fit_curve_re = fit_result_re.eval()
        residuals_re = signal_re - fit_curve_re
        amp_re = fit_result_re.params["A"].value
        noise_re = np.std(residuals_re)
        snr_re = np.abs(amp_re / (4 * noise_re))

        # Fit imaginary part
        fit_imag = ExponentiallyDecayingSine(delays, signal_imag)
        fit_result_imag = fit_imag.run(fit_imag)
        fit_curve_imag = fit_result_imag.eval()
        residuals_imag = signal_imag - fit_curve_imag
        amp_imag = fit_result_imag.params["A"].value
        noise_imag = np.std(residuals_imag)
        snr_imag = np.abs(amp_imag / (4 * noise_imag))

        # Fit magnitude
        fit_mag = ExponentiallyDecayingSine(delays, signal_mag)
        fit_result_mag = fit_mag.run(fit_mag)
        fit_curve_mag = fit_result_mag.eval()
        residuals_mag = signal_mag - fit_curve_mag
        amp_mag = fit_result_mag.params["A"].value
        noise_mag = np.std(residuals_mag)
        snr_mag = np.abs(amp_mag / (4 * noise_mag))

        # Create three separate figures
        fig_re, ax_re = plt.subplots()
        ax_re.set_title(f"{fig_title} - Real")
        ax_re.set_xlabel("Delay (μs)")
        ax_re.set_ylabel("Signal Real (A.U)")
        ax_re.plot(delays, signal_re, label="Data")
        ax_re.plot(delays, fit_curve_re, label="Fit")
        ax_re.legend()

        fig_imag, ax_imag = plt.subplots()
        ax_imag.set_title(f"{fig_title} - Imaginary")
        ax_imag.set_xlabel("Delay (μs)")
        ax_imag.set_ylabel("Signal Imaginary (A.U)")
        ax_imag.plot(delays, signal_imag, label="Data")
        ax_imag.plot(delays, fit_curve_imag, label="Fit")
        ax_imag.legend()

        fig_mag, ax_mag = plt.subplots()
        ax_mag.set_title(f"{fig_title} - Magnitude")
        ax_mag.set_xlabel("Delay (μs)")
        ax_mag.set_ylabel("Signal Magnitude (A.U)")
        ax_mag.plot(delays, signal_mag, label="Data")
        ax_mag.plot(delays, fit_curve_mag, label="Fit")
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
            result_re, result_imag, result_mag, fig_re, fig_imag, fig_mag = self._fit_exponentially_decaying_sine_components(
                self.independents["delays"],
                self.dependents["signal"],
                "T2 Ramsey Measurement"
            )

            self.fit_result_re, residuals_re, self.snr_re = result_re
            self.fit_result_imag, residuals_imag, self.snr_imag = result_imag
            self.fit_result_mag, residuals_mag, self.snr_mag = result_mag

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

            # Save all three figures separately
            ds.add_figure(f"{self.name}_real", fig=fig_re)
            image_path_re = ds._new_file_path(ds.savefolders[1], f"{self.name}_real", suffix="png")
            self.figure_paths.append(image_path_re)

            ds.add_figure(f"{self.name}_imag", fig=fig_imag)
            image_path_imag = ds._new_file_path(ds.savefolders[1], f"{self.name}_imag", suffix="png")
            self.figure_paths.append(image_path_imag)

            ds.add_figure(f"{self.name}_mag", fig=fig_mag)
            image_path_mag = ds._new_file_path(ds.savefolders[1], f"{self.name}_mag", suffix="png")
            self.figure_paths.append(image_path_mag)

            snr_dict = {
                "Real":      (self.snr_re,   self.fit_result_re,   "re"),
                "Imaginary": (self.snr_imag, self.fit_result_imag, "imag"),
                "Magnitude": (self.snr_mag,  self.fit_result_mag,  "mag"),
            }
            self._sorted_components = sorted(snr_dict.items(), key=lambda x: x[1][0], reverse=True)

    def _check_quality(self) -> CheckResult:
        snr_min = self.snr_min_threshold()

        valid = [
            (name, snr, fit, key)
            for name, (snr, fit, key) in self._sorted_components
            if snr >= snr_min
        ]

        if valid:
            self._winner_name, self._winner_snr, self._winner_fit, self._winner_key = valid[0]
            max_error = self.max_fit_param_error()
            bad_params = []
            for pname, param in self._winner_fit.params.items():
                if param.stderr is None:
                    bad_params.append(f"{pname}(no stderr)")
                elif param.value == 0 or abs(param.stderr / param.value) > max_error:
                    pct = abs(param.stderr / param.value) * 100 if param.value != 0 else float("inf")
                    bad_params.append(f"{pname}({pct:.0f}%)")
            passed = len(bad_params) == 0
            parts = [f"winner={self._winner_name}, SNR={self._winner_snr:.3f} (threshold={snr_min:.1f})"]
            if bad_params:
                parts.append(f"high-error params: {', '.join(bad_params)}")
        else:
            self._winner_name, (self._winner_snr, self._winner_fit, self._winner_key) = self._sorted_components[0]
            passed = False
            parts = [
                f"no component with SNR >= {snr_min:.1f}",
                f"best={self._winner_name}, SNR={self._winner_snr:.3f}",
            ]

        return CheckResult("quality_check", passed, "; ".join(parts))

    def correct(self, result: EvaluateResult) -> EvaluateResult:
        # Pop all figures before super() auto-appends the last one
        fig_re   = self.figure_paths.pop(0) if len(self.figure_paths) >= 3 else None
        fig_imag = self.figure_paths.pop(0) if self.figure_paths else None
        fig_mag  = self.figure_paths.pop(0) if self.figure_paths else None
        self.figure_paths.clear()  # prevent auto-append

        plot_map = {"re": fig_re, "imag": fig_imag, "mag": fig_mag}

        snr_min = self.snr_min_threshold()
        self.report_output.append(
            f"## T2 Ramsey (T2R) Measurement\n"
            f"Measured T2 Ramsey time with SNR threshold: {snr_min:.1f}\n"
            f"Data Path: `{self.data_loc}`\n\n"
        )

        for i, (comp_name, (comp_snr, comp_fit, comp_key)) in enumerate(self._sorted_components):
            tag = "(SELECTED)" if i == 0 else "(NOT SELECTED)"
            self.report_output.append(f"### **{comp_name} Component {tag}**\n")
            if plot_map.get(comp_key):
                self.report_output.append(plot_map[comp_key])
            self.report_output.append(
                f"SNR={comp_snr:.3f}\n\n"
                f"**Fit Report:**\n```\n{str(comp_fit.lmfit_result.fit_report())}\n```\n\n"
            )

        result = super().correct(result)  # adds check table + success update line
        return result
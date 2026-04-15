import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")

from labcore.analysis import DatasetAnalysis
from labcore.analysis.fitfuncs.generic import Gaussian
from labcore.measurement.storage import run_and_save_sweep
from labcore.measurement.sweep import sweep_parameter
from labcore.measurement.record import record_as
from labcore.data.datagen import Gaussian as GaussianDataGen
from labcore.data.datadict_storage import datadict_from_hdf5

from labcore.protocols.base import (ProtocolOperation, OperationStatus, serialize_fit_params,
                                    CorrectionParameter, CheckResult, Correction, EvaluateResult)
from cqedtoolbox.protocols.parameters import (
    Repetition,
    PiSpecSteps,
    StartPiSpecFrequency,
    EndPiSpecFrequency,
    QubitFrequency,
    QubitGain,
    ReadoutGain,
    ReadoutLength
)
from cqedtoolbox.measurement_lib.qick.single_transmon_v2 import PiSpecProgram


logger = logging.getLogger(__name__)


@dataclass
class PiSpecSNRThreshold(CorrectionParameter):
    name: str = field(default="pi_spec_snr_threshold", init=False)
    description: str = field(default="Minimum SNR for a successful pi spectroscopy fit", init=False)

    def _dummy_getter(self): return 2.0

    def _qick_getter(self):
        return self.params.corrections.pi_spec.snr()

    def _qick_setter(self, value):
        self.params.corrections.pi_spec.snr(value)


@dataclass
class PiSpecMaxFitParamError(CorrectionParameter):
    name: str = field(default="pi_spec_max_fit_param_error", init=False)
    description: str = field(default="Max allowed fractional fit parameter error (e.g. 1.0 = 100%)", init=False)

    def _dummy_getter(self): return 1.0

    def _qick_getter(self):
        return self.params.corrections.pi_spec.max_fit_param_error()

    def _qick_setter(self, value):
        self.params.corrections.pi_spec.max_fit_param_error(value)


@dataclass
class PiSpecAveragingFactor(CorrectionParameter):
    name: str = field(default="pi_spec_averaging_factor", init=False)
    description: str = field(default="Factor by which to multiply repetitions on retry", init=False)

    def _dummy_getter(self): return 2.0

    def _qick_getter(self):
        return self.params.corrections.pi_spec.averaging_factor()

    def _qick_setter(self, value):
        self.params.corrections.pi_spec.averaging_factor(value)


@dataclass
class PiSpecMaxAveragingIncreases(CorrectionParameter):
    name: str = field(default="pi_spec_max_averaging_increases", init=False)
    description: str = field(default="Maximum number of repetition increases to attempt", init=False)

    def _dummy_getter(self): return 3

    def _qick_getter(self):
        return int(self.params.corrections.pi_spec.max_averaging_increases())

    def _qick_setter(self, value):
        self.params.corrections.pi_spec.max_averaging_increases(value)


class IncreaseAveragingCorrection(Correction):
    name = "increase_averaging"
    description = "Multiply repetitions by a factor to improve SNR"
    triggered_by = "quality_check"

    def __init__(self, reps_param, factor_param, max_increases_param):
        self.reps_param = reps_param
        self.factor_param = factor_param
        self.max_increases_param = max_increases_param
        self._count = 0
        self._last_change = ""

    def can_apply(self) -> bool:
        return self._count < int(self.max_increases_param())

    def apply(self) -> None:
        old = self.reps_param()
        new = int(old * self.factor_param())
        self.reps_param(new)
        self._count += 1
        self._last_change = f"{old} → {new} reps"

    def report_output(self) -> str:
        return self._last_change


class PiSpectroscopy(ProtocolOperation):

    _SIM_CENTER = 0.0
    _SIM_SIGMA = 3e6  # 3 MHz
    _SIM_AMP = 0.5
    _SIM_NOISE_AMP = 0.02

    def __init__(self, params):
        super().__init__()

        self._register_inputs(
            repetitions=Repetition(params),
            steps=PiSpecSteps(params),
            start_freq=StartPiSpecFrequency(params),
            end_freq=EndPiSpecFrequency(params),
            qubit_gain=QubitGain(params),
            readout_gain=ReadoutGain(params),
            readout_length=ReadoutLength(params)
        )
        self._register_outputs(
            qubit_freq=QubitFrequency(params)
        )

        self._register_correction_params(
            snr_threshold=PiSpecSNRThreshold(params),
            max_fit_param_error=PiSpecMaxFitParamError(params),
            averaging_factor=PiSpecAveragingFactor(params),
            max_averaging_increases=PiSpecMaxAveragingIncreases(params),
        )

        self._increase_averaging = IncreaseAveragingCorrection(
            self.repetitions, self.averaging_factor, self.max_averaging_increases
        )
        self._register_check("quality_check", self._check_quality, self._increase_averaging)
        self._register_success_update(self.qubit_freq, lambda: self.winner_fit.params["x0"].value)

        self.independents = {"frequencies": []}
        self.dependents = {"signal": []}

        self.fit_result_re = None
        self.fit_result_imag = None
        self.fit_result_mag = None
        self.snr_re = None
        self.snr_imag = None
        self.snr_mag = None
        self.winner_name = None
        self.winner_snr = None
        self.winner_fit = None
        self.winner_key = None
        self.sorted_components = None

    def _measure_dummy(self) -> Path:
        logger.info("Starting dummy pi spectroscopy measurement")
        frequencies = np.linspace(self.start_freq(), self.end_freq(), int(self.steps()))
        center = (self.start_freq() + self.end_freq()) / 2 + self._SIM_CENTER

        generator = GaussianDataGen(x0=center, sigma=self._SIM_SIGMA, A=self._SIM_AMP, of=0, noise_std=self._SIM_NOISE_AMP)
        sweep = sweep_parameter("frequencies", frequencies, record_as(lambda frequencies: generator.generate(np.atleast_1d(frequencies)), "signal"))
        loc, _ = run_and_save_sweep(sweep, "data", self.name)
        logger.info("Dummy measurement complete")
        return loc

    def _load_data_dummy(self):
        path = self.data_loc / "data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)
        self.independents["frequencies"] = data["frequencies"]["values"]
        self.dependents["signal"] = data["signal"]["values"]

    def _measure_qick(self) -> Path:
        logger.info("Starting qick pi spectroscopy measurement")

        sweep = PiSpecProgram()
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

    def _fit_gaussian_components(self, frequencies, signal, fig_title="") -> tuple:
        """
        Fit real, imaginary, and magnitude components with Gaussian fits.
        Returns (fit_result_re, fit_result_imag, fit_result_mag, fig_re, fig_imag, fig_mag)
        """
        signal_re = signal.real
        signal_imag = signal.imag
        signal_mag = np.abs(signal)

        # Fit real part
        fit_re = Gaussian(frequencies, signal_re)
        fit_result_re = fit_re.run(fit_re)
        fit_curve_re = fit_result_re.eval()
        residuals_re = signal_re - fit_curve_re
        amp_re = fit_result_re.params["A"].value
        noise_re = np.std(residuals_re)
        snr_re = np.abs(amp_re / (4 * noise_re))

        # Fit imaginary part
        fit_imag = Gaussian(frequencies, signal_imag)
        fit_result_imag = fit_imag.run(fit_imag)
        fit_curve_imag = fit_result_imag.eval()
        residuals_imag = signal_imag - fit_curve_imag
        amp_imag = fit_result_imag.params["A"].value
        noise_imag = np.std(residuals_imag)
        snr_imag = np.abs(amp_imag / (4 * noise_imag))

        # Fit magnitude
        fit_mag = Gaussian(frequencies, signal_mag)
        fit_result_mag = fit_mag.run(fit_mag)
        fit_curve_mag = fit_result_mag.eval()
        residuals_mag = signal_mag - fit_curve_mag
        amp_mag = fit_result_mag.params["A"].value
        noise_mag = np.std(residuals_mag)
        snr_mag = np.abs(amp_mag / (4 * noise_mag))

        # Create three separate figures
        # Real plot
        fig_re, ax_re = plt.subplots()
        ax_re.set_title(f"{fig_title} - Real")
        ax_re.set_xlabel("Frequency (MHz)")
        ax_re.set_ylabel("Signal Real (A.U)")
        ax_re.plot(frequencies, signal_re, label="Data")
        ax_re.plot(frequencies, fit_curve_re, label="Fit")
        ax_re.legend()

        # Imaginary plot
        fig_imag, ax_imag = plt.subplots()
        ax_imag.set_title(f"{fig_title} - Imaginary")
        ax_imag.set_xlabel("Frequency (MHz)")
        ax_imag.set_ylabel("Signal Imaginary (A.U)")
        ax_imag.plot(frequencies, signal_imag, label="Data")
        ax_imag.plot(frequencies, fit_curve_imag, label="Fit")
        ax_imag.legend()

        # Magnitude plot
        fig_mag, ax_mag = plt.subplots()
        ax_mag.set_title(f"{fig_title} - Magnitude")
        ax_mag.set_xlabel("Frequency (MHz)")
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
            result_re, result_imag, result_mag, fig_re, fig_imag, fig_mag = self._fit_gaussian_components(
                self.independents["frequencies"],
                self.dependents["signal"],
                "Pi Spectroscopy"
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

    def _check_quality(self) -> CheckResult:
        snr_dict = {
            "Real":      (self.snr_re,   self.fit_result_re,   "re"),
            "Imaginary": (self.snr_imag, self.fit_result_imag, "imag"),
            "Magnitude": (self.snr_mag,  self.fit_result_mag,  "mag"),
        }
        self.sorted_components = sorted(snr_dict.items(), key=lambda x: x[1][0], reverse=True)
        self.winner_name, (self.winner_snr, self.winner_fit, self.winner_key) = self.sorted_components[0]

        threshold = self.snr_threshold()
        snr_passed = self.winner_snr >= threshold

        max_error = self.max_fit_param_error()
        bad_params = []
        for pname, param in self.winner_fit.params.items():
            if param.stderr is None:
                bad_params.append(f"{pname}(no stderr)")
            elif param.value != 0 and abs(param.stderr / param.value) > max_error:
                pct = abs(param.stderr / param.value) * 100
                bad_params.append(f"{pname}({pct:.0f}%)")

        passed = snr_passed and len(bad_params) == 0
        parts = [f"SNR={self.winner_snr:.3f} (threshold={threshold:.3f}, component={self.winner_name})"]
        if bad_params:
            parts.append(f"high-error params: {', '.join(bad_params)}")
        return CheckResult("quality_check", passed, "; ".join(parts))

    def correct(self, result: EvaluateResult) -> EvaluateResult:
        # Pull all three figures before super() can auto-append the last one.
        # figure_paths order after analyze(): [0]=real, [1]=imag, [2]=mag
        plot_map = {}
        if len(self.figure_paths) >= 3:
            plot_map["re"]   = self.figure_paths[0].resolve()
            plot_map["imag"] = self.figure_paths[1].resolve()
            plot_map["mag"]  = self.figure_paths[2].resolve()
        self.figure_paths.clear()  # prevent auto-append

        header = (f"## Pi Spectroscopy\n"
                  f"Frequencies: {self.start_freq():.3f}–{self.end_freq():.3f} MHz\n"
                  f"Data Path: `{self.data_loc}`\n\n")
        self.report_output.append(header)

        result = super().correct(result)  # adds check table; no auto-figure since list is empty

        if self.sorted_components:
            winner_name, (winner_snr, winner_fit, winner_key) = self.sorted_components[0]
            winner_report = f"**Fit Report:**\n```\n{str(winner_fit.lmfit_result.fit_report())}\n```\n\n"

            if result.status == OperationStatus.SUCCESS:
                self.report_output.extend([
                    f"### **{winner_name} Component (SELECTED)**\n"
                    f"Fit was **SUCCESSFUL** with {winner_name} SNR of {winner_snr:.3f}\n"
                    f"This component was selected because it has the highest SNR.\n\n",
                    plot_map.get(winner_key, ""),
                    winner_report,
                ])
                for comp_name, (comp_snr, comp_fit, comp_key) in self.sorted_components[1:]:
                    threshold = self.snr_threshold()
                    if comp_snr >= threshold:
                        reason = f"SNR of {comp_snr:.3f} is above threshold but lower than {winner_name} (SNR={winner_snr:.3f})"
                    else:
                        reason = f"SNR of {comp_snr:.3f} is below threshold of {threshold:.3f}"
                    self.report_output.extend([
                        f"### **{comp_name} Component (NOT SELECTED)**\n",
                        plot_map.get(comp_key, ""),
                        f"Not used: {reason}\n\n**Fit Report:**\n```\n{str(comp_fit.lmfit_result.fit_report())}\n```\n\n",
                    ])
            else:
                self.report_output.extend([
                    f"### **{winner_name} Component (Highest SNR)**\n"
                    f"Fit was **UNSUCCESSFUL** with {winner_name} SNR of {winner_snr:.3f}\n"
                    f"NO value has been changed.\n\n",
                    plot_map.get(winner_key, ""),
                    winner_report,
                ])
                for comp_name, (comp_snr, comp_fit, comp_key) in self.sorted_components[1:]:
                    self.report_output.extend([
                        f"### **{comp_name} Component**\n",
                        plot_map.get(comp_key, ""),
                        f"SNR of {comp_snr:.3f}\n\n**Fit Report:**\n```\n{str(comp_fit.lmfit_result.fit_report())}\n```\n\n",
                    ])

        return result
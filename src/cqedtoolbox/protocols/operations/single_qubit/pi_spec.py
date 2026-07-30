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
from labcore.data.datadict_storage import datadict_from_hdf5, load_as_xr

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
from cqedtoolbox.measurement_lib.opx.advanced.qubit_tuneup import measure_pi_spec
from cqedtoolbox.measurement_lib.qick.single_transmon_v2 import PiSpecProgram
from cqedtoolbox.readout.qubit_readout import rotate_complex_qubit_data


logger = logging.getLogger(__name__)


@dataclass
class PiSpecSNRThreshold(CorrectionParameter):
    name: str = field(default="pi_spec_snr_threshold", init=False)
    description: str = field(default="Minimum SNR for a successful pi spectroscopy fit", init=False)

    def _qick_getter(self):
        return self.params.corrections.pi_spec.snr()

    def _qick_setter(self, value):
        self.params.corrections.pi_spec.snr(value)

    def _opx_getter(self):
        return self.params.corrections.pi_spec.snr()

    def _opx_setter(self, value):
        self.params.corrections.pi_spec.snr(value)


@dataclass
class PiSpecMaxFitParamError(CorrectionParameter):
    name: str = field(default="pi_spec_max_fit_param_error", init=False)
    description: str = field(default="Max allowed fractional fit parameter error (e.g. 1.0 = 100%)", init=False)

    def _qick_getter(self):
        return self.params.corrections.pi_spec.max_fit_param_error()

    def _qick_setter(self, value):
        self.params.corrections.pi_spec.max_fit_param_error(value)

    def _opx_getter(self):
        return self.params.corrections.pi_spec.max_fit_param_error()

    def _opx_setter(self, value):
        self.params.corrections.pi_spec.max_fit_param_error(value)


@dataclass
class PiSpecAveragingFactor(CorrectionParameter):
    name: str = field(default="pi_spec_averaging_factor", init=False)
    description: str = field(default="Factor by which to multiply repetitions on retry", init=False)

    def _qick_getter(self):
        return self.params.corrections.pi_spec.averaging_factor()

    def _qick_setter(self, value):
        self.params.corrections.pi_spec.averaging_factor(value)

    def _opx_getter(self):
        return self.params.corrections.pi_spec.averaging_factor()

    def _opx_setter(self, value):
        self.params.corrections.pi_spec.averaging_factor(value)


@dataclass
class PiSpecMaxAveragingIncreases(CorrectionParameter):
    name: str = field(default="pi_spec_max_averaging_increases", init=False)
    description: str = field(default="Maximum number of repetition increases to attempt", init=False)

    def _qick_getter(self):
        return int(self.params.corrections.pi_spec.max_averaging_increases())

    def _qick_setter(self, value):
        self.params.corrections.pi_spec.max_averaging_increases(value)

    def _opx_getter(self):
        return int(self.params.corrections.pi_spec.max_averaging_increases())

    def _opx_setter(self, value):
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
        self.params = params

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
        self._register_success_update(self.qubit_freq, lambda: self.fit_result.params["x0"].value)

        self.independents = {"frequencies": []}
        self.dependents = {"signal": []}

        self.fit_result = None
        self.residuals = None
        self.snr = None

    def _measure_dummy(self) -> Path:
        logger.info("Starting dummy pi spectroscopy measurement")
        frequencies = np.linspace(self.start_freq(), self.end_freq(), int(self.steps()))
        center = (self.start_freq() + self.end_freq()) / 2 + self._SIM_CENTER

        def generate(frequencies):
            return (self._SIM_AMP * np.exp(-0.5 * ((frequencies - center) / self._SIM_SIGMA) ** 2)
                    + self._SIM_NOISE_AMP * (np.random.randn() + 1j * np.random.randn()))

        sweep = sweep_parameter("frequencies", frequencies, record_as(generate, "signal"))
        loc, _ = run_and_save_sweep(sweep, "data", self.name)
        logger.info("Dummy measurement complete")
        return loc

    def _load_data_dummy(self):
        data = load_as_xr(self.data_loc)
        rotated = rotate_complex_qubit_data(data)[0]
        self.independents["frequencies"] = rotated["frequencies"].values
        self.dependents["signal"] = rotated["signal"].values

    def _measure_qick(self) -> Path:
        logger.info("Starting qick pi spectroscopy measurement")

        sweep = PiSpecProgram()
        logger.debug("Sweep created, running measurement")
        loc, da = run_and_save_sweep(sweep, "data", self.name)
        logger.info("Measurement complete")

        return loc

    def _measure_opx(self) -> Path:
        logger.info("Starting opx pi spectroscopy measurement")
        loc = measure_pi_spec()
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

    def _fit_gaussian(self, frequencies, signal, fig_title="") -> tuple:
        fit = Gaussian(frequencies, signal)
        fit_result = fit.run(fit)
        fit_curve = fit_result.eval()
        residuals = signal - fit_curve
        amp = fit_result.params["A"].value
        noise = np.std(residuals)
        snr = np.abs(amp / (4 * noise))

        fig, ax = plt.subplots()
        ax.set_title(fig_title)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Rotated Signal (A.U)")
        ax.plot(frequencies, signal, label="Data")
        ax.plot(frequencies, fit_curve, label="Fit")
        ax.legend()

        return fit_result, residuals, snr, fig

    def analyze(self):
        with DatasetAnalysis(self.data_loc, self.name) as ds:
            self.fit_result, self.residuals, self.snr, fig = self._fit_gaussian(
                self.independents["frequencies"],
                self.dependents["signal"],
                "Pi Spectroscopy"
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
        # TODO: make sure that the fit frequency is inside the swept range
        threshold = self.snr_threshold()
        snr_passed = self.snr >= threshold

        max_error = self.max_fit_param_error()
        bad_params = []
        for pname, param in self.fit_result.params.items():
            if param.stderr is None:
                bad_params.append(f"{pname}(no stderr)")
            elif param.value != 0 and abs(param.stderr / param.value) > max_error:
                pct = abs(param.stderr / param.value) * 100
                bad_params.append(f"{pname}({pct:.0f}%)")

        passed = snr_passed and len(bad_params) == 0
        parts = [f"SNR={self.snr:.3f} (threshold={threshold:.3f})"]
        if bad_params:
            parts.append(f"high-error params: {', '.join(bad_params)}")
        return CheckResult("quality_check", passed, "; ".join(parts))

    def correct(self, result: EvaluateResult) -> EvaluateResult:
        figure = self.figure_paths[0].resolve() if self.figure_paths else None
        self.figure_paths.clear()

        header = (f"## Pi Spectroscopy\n"
                  f"Frequencies: {self.start_freq():.3f}–{self.end_freq():.3f} MHz\n"
                  f"Data Path: `{self.data_loc}`\n\n")
        self.report_output.append(header)

        result = super().correct(result)  # adds check table; no auto-figure since list is empty

        status_line = "SUCCESSFUL" if result.status == OperationStatus.SUCCESS else "UNSUCCESSFUL"
        self.report_output.extend([
            f"### Rotated Signal Fit\n"
            f"Fit was **{status_line}** with SNR of {self.snr:.3f}\n\n",
            figure or "",
            f"**Fit Report:**\n```\n{str(self.fit_result.lmfit_result.fit_report())}\n```\n\n",
        ])

        return result

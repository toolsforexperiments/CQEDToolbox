import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")

from labcore.analysis import DatasetAnalysis
from labcore.measurement.storage import run_and_save_sweep
from labcore.measurement.sweep import sweep_parameter
from labcore.measurement.record import record_as
from cqedtoolbox.protocols.operations.single_qubit.res_spec import HangerResonator
from labcore.data.datadict_storage import datadict_from_hdf5

from labcore.protocols.base import (ProtocolOperation, OperationStatus, serialize_fit_params,
                                    CorrectionParameter, CheckResult, EvaluateResult)
from cqedtoolbox.protocols.parameters import (
    Repetition,
    ResonatorSpecSteps,
    StartReadoutFrequency,
    EndReadoutFrequency,
    ReadoutGain,
    ReadoutLength,
    Detuning
)
from cqedtoolbox.measurement_lib.qick.single_transmon_v2 import FreqSweepProgram, ResProbeProgram

from cqedtoolbox.fitfuncs.resonators import HangerResponseBruno
from cqedtoolbox.protocols.operations.single_qubit.res_spec import ResonatorSpectroscopy as _RS


logger = logging.getLogger(__name__)

_EXCLUDED_FIT_PARAMS = {"transmission_slope", "phase_slope", "phase_offset"}


@dataclass
class ResSpecAfterPiSNRThreshold(CorrectionParameter):
    name: str = field(default="res_spec_after_pi_snr_threshold", init=False)
    description: str = field(default="Minimum SNR for a successful resonator fit", init=False)

    def _dummy_getter(self): return 2.0

    def _qick_getter(self):
        return self.params.corrections.res_spec_after_pi.snr()

    def _qick_setter(self, value):
        self.params.corrections.res_spec_after_pi.snr(value)


@dataclass
class ResSpecAfterPiMaxFitParamError(CorrectionParameter):
    name: str = field(default="res_spec_after_pi_max_fit_param_error", init=False)
    description: str = field(default="Max allowed fractional fit parameter error (e.g. 1.0 = 100%)", init=False)

    def _dummy_getter(self): return 1.0

    def _qick_getter(self):
        return self.params.corrections.res_spec_after_pi.max_fit_param_error()

    def _qick_setter(self, value):
        self.params.corrections.res_spec_after_pi.max_fit_param_error(value)


@dataclass
class DetuningThreshold(CorrectionParameter):
    name: str = field(default="res_spec_after_pi_detuning_threshold", init=False)
    description: str = field(default="Minimum Chi to consider the dispersive shift valid (MHz)", init=False)

    def _dummy_getter(self): return 1.0e6

    def _qick_getter(self):
        return self.params.corrections.res_spec_after_pi.detuning_threshold()

    def _qick_setter(self, value):
        self.params.corrections.res_spec_after_pi.detuning_threshold(value)


class ResonatorSpectroscopyAfterPi(ProtocolOperation):

    def __init__(self, params):
        super().__init__()

        self._register_inputs(
            repetitions=Repetition(params),
            steps=ResonatorSpecSteps(params),
            start_freq=StartReadoutFrequency(params),
            end_freq=EndReadoutFrequency(params),
            readout_gain=ReadoutGain(params),
            readout_length=ReadoutLength(params)
        )
        self._register_outputs(
            detuning=Detuning(params)
        )

        self._register_correction_params(
            snr_threshold=ResSpecAfterPiSNRThreshold(params),
            max_fit_param_error=ResSpecAfterPiMaxFitParamError(params),
            detuning_threshold=DetuningThreshold(params),
        )

        self._register_check("quality_check_before", self._check_quality_before, None)
        self._register_check("quality_check_after", self._check_quality_after, None)
        self._register_check("detuning_check", self._check_detuning, None)
        self._register_success_update(self.detuning, lambda: self.chi)

        # Data for before measurement
        self.data_loc_before: Path | None = None
        self.independents_before = {"frequencies": []}
        self.dependents_before = {"signal": []}
        self.unwind_signal_before = None
        self.magnitude_before = None
        self.fit_result_before = None
        self.snr_before = None
        self.f0_before = None

        # Data for after measurement
        self.data_loc_after: Path | None = None
        self.independents_after = {"frequencies": []}
        self.dependents_after = {"signal": []}
        self.unwind_signal_after = None
        self.magnitude_after = None
        self.fit_result_after = None
        self.snr_after = None
        self.f0_after = None

        # Detuning value
        self.chi = None

    _SIM_CHI = 2.0  # MHz dispersive shift between ground and excited state

    def _measure_dummy(self) -> Path:
        logger.info("Starting dummy resonator spectroscopy before/after pi measurement")
        frequencies = np.linspace(self.start_freq(), self.end_freq(), int(self.steps()))

        gen_before = HangerResonator(f0=_RS._SIM_F0, Qc=_RS._SIM_QC, Qi=_RS._SIM_QI, A=_RS._SIM_A, phi=_RS._SIM_PHI, noise_std=_RS._SIM_NOISE_AMP)
        sweep_before = sweep_parameter("frequencies", frequencies, record_as(lambda frequencies: gen_before.generate(np.atleast_1d(frequencies)), "signal"))
        loc_before, _ = run_and_save_sweep(sweep_before, "data", f"{self.name}_before")

        gen_after = HangerResonator(f0=_RS._SIM_F0 + self._SIM_CHI, Qc=_RS._SIM_QC, Qi=_RS._SIM_QI, A=_RS._SIM_A, phi=_RS._SIM_PHI, noise_std=_RS._SIM_NOISE_AMP)
        sweep_after = sweep_parameter("frequencies", frequencies, record_as(lambda frequencies: gen_after.generate(np.atleast_1d(frequencies)), "signal"))
        loc_after, _ = run_and_save_sweep(sweep_after, "data", f"{self.name}_after")

        self.data_loc_before = loc_before
        self.data_loc_after = loc_after
        logger.info("Dummy measurement complete")
        return loc_before

    def _load_data_dummy(self):
        path_before = self.data_loc_before / "data.ddh5"
        if not path_before.exists():
            raise FileNotFoundError(f"File {path_before} does not exist")
        data_before = datadict_from_hdf5(path_before)
        self.independents_before["frequencies"] = data_before["frequencies"]["values"]
        self.dependents_before["signal"] = data_before["signal"]["values"]

        path_after = self.data_loc_after / "data.ddh5"
        if not path_after.exists():
            raise FileNotFoundError(f"File {path_after} does not exist")
        data_after = datadict_from_hdf5(path_after)
        self.independents_after["frequencies"] = data_after["frequencies"]["values"]
        self.dependents_after["signal"] = data_after["signal"]["values"]

    def _measure_qick(self) -> Path:
        logger.info("Starting qick resonator spectroscopy before/after pi measurement")

        # Measure before pi
        sweep_before = FreqSweepProgram()
        logger.debug("Sweep created for before pi, running measurement")
        loc_before, da_before = run_and_save_sweep(sweep_before, "data", f"{self.name}_before")
        logger.info("Before pi measurement complete at %s", loc_before)

        # Measure after pi
        sweep_after = ResProbeProgram()
        logger.debug("Sweep created for after pi, running measurement")
        loc_after, da_after = run_and_save_sweep(sweep_after, "data", f"{self.name}_after")
        logger.info("After pi measurement complete at %s", loc_after)

        self.data_loc_before = loc_before
        self.data_loc_after = loc_after

        # Return the "before" location as the primary data_loc for compatibility
        return loc_before

    def _add_mag_and_unwind_and_fit(self, frequencies, signal_raw, fig_title="") -> tuple:
        """Unwind phase, calculate magnitude, and fit with hanger response"""
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
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Magnitude Signal (A.U)")
        ax.plot(frequencies, magnitude, label="Data")
        ax.plot(frequencies, np.abs(fit_curve), label="Fit")
        ax.legend()

        return signal_unwind, magnitude, phase, fit_result, residuals, snr, fig

    def _load_data_qick(self):
        # Load before data
        path_before = self.data_loc_before / "data.ddh5"
        if not path_before.exists():
            raise FileNotFoundError(f"File {path_before} does not exist")
        data_before = datadict_from_hdf5(path_before)

        self.independents_before["frequencies"] = data_before["freq"]["values"]
        self.dependents_before["signal"] = data_before["signal"]["values"]

        # Load after data
        path_after = self.data_loc_after / "data.ddh5"
        if not path_after.exists():
            raise FileNotFoundError(f"File {path_after} does not exist")
        data_after = datadict_from_hdf5(path_after)

        self.independents_after["frequencies"] = data_after["freq"]["values"]
        self.dependents_after["signal"] = data_after["signal"]["values"]

    def analyze(self):
        # Analyze before measurement
        with DatasetAnalysis(self.data_loc_before, f"{self.name}_before") as ds:
            ret_before = self._add_mag_and_unwind_and_fit(
                self.independents_before["frequencies"],
                self.dependents_before["signal"],
                "Resonator Spectroscopy Before Pi"
            )

            (self.unwind_signal_before, self.magnitude_before, phase_before,
             self.fit_result_before, residuals_before, self.snr_before, fig_before) = ret_before

            self.f0_before = self.fit_result_before.params["f_0"].value

            ds.add(
                fit_curve=self.fit_result_before.eval(),
                fit_result=self.fit_result_before,
                params=serialize_fit_params(self.fit_result_before.params),
                snr=float(self.snr_before)
            )
            ds.add_figure(f"{self.name}_before", fig=fig_before)

            image_path_before = ds._new_file_path(ds.savefolders[1], f"{self.name}_before", suffix="png")
            self.figure_paths.append(image_path_before)

        # Analyze after measurement
        with DatasetAnalysis(self.data_loc_after, f"{self.name}_after") as ds:
            ret_after = self._add_mag_and_unwind_and_fit(
                self.independents_after["frequencies"],
                self.dependents_after["signal"],
                "Resonator Spectroscopy After Pi"
            )

            (self.unwind_signal_after, self.magnitude_after, phase_after,
             self.fit_result_after, residuals_after, self.snr_after, fig_after) = ret_after

            self.f0_after = self.fit_result_after.params["f_0"].value

            ds.add(
                fit_curve=self.fit_result_after.eval(),
                fit_result=self.fit_result_after,
                params=serialize_fit_params(self.fit_result_after.params),
                snr=float(self.snr_after)
            )
            ds.add_figure(f"{self.name}_after", fig=fig_after)

            image_path_after = ds._new_file_path(ds.savefolders[1], f"{self.name}_after", suffix="png")
            self.figure_paths.append(image_path_after)

        # Calculate chi
        self.chi = self.f0_before - self.f0_after

        # Create combined plot
        with DatasetAnalysis(self.data_loc_before, f"{self.name}_combined") as ds:
            fig_combined, ax_combined = plt.subplots()
            ax_combined.set_title(f"Resonator Spec Before/After Pi (χ = {self.chi:.3f} MHz)")
            ax_combined.set_xlabel("Frequency (MHz)")
            ax_combined.set_ylabel("Signal Magnitude (A.U.)")

            ax_combined.plot(
                self.independents_before["frequencies"],
                np.abs(self.unwind_signal_before),
                label="Before Pi",
                alpha=0.7
            )
            ax_combined.plot(
                self.independents_before["frequencies"],
                np.abs(self.fit_result_before.eval()),
                label="Before Pi Fit",
                linestyle="--"
            )
            ax_combined.plot(
                self.independents_after["frequencies"],
                np.abs(self.unwind_signal_after),
                label="After Pi",
                alpha=0.7
            )
            ax_combined.plot(
                self.independents_after["frequencies"],
                np.abs(self.fit_result_after.eval()),
                label="After Pi Fit",
                linestyle="--"
            )
            ax_combined.legend()

            ds.add_figure(f"{self.name}_combined", fig=fig_combined)

            image_path_combined = ds._new_file_path(ds.savefolders[1], f"{self.name}_combined", suffix="png")
            self.figure_paths.append(image_path_combined)

    def _check_fit_quality(self, snr, fit_result, check_name) -> CheckResult:
        threshold = self.snr_threshold()
        snr_passed = snr >= threshold

        max_error = self.max_fit_param_error()
        bad_params = []
        for pname, param in fit_result.params.items():
            if pname in _EXCLUDED_FIT_PARAMS:
                continue
            if param.stderr is None:
                bad_params.append(f"{pname}(no stderr)")
            elif param.value == 0 or abs(param.stderr / param.value) > max_error:
                pct = abs(param.stderr / param.value) * 100 if param.value != 0 else float("inf")
                bad_params.append(f"{pname}({pct:.0f}%)")

        passed = snr_passed and len(bad_params) == 0
        parts = [f"SNR={snr:.3f} (threshold={threshold:.3f})"]
        if bad_params:
            parts.append(f"high-error params: {', '.join(bad_params)}")
        return CheckResult(check_name, passed, "; ".join(parts))

    def _check_quality_before(self) -> CheckResult:
        return self._check_fit_quality(self.snr_before, self.fit_result_before, "quality_check_before")

    def _check_quality_after(self) -> CheckResult:
        return self._check_fit_quality(self.snr_after, self.fit_result_after, "quality_check_after")

    def _check_detuning(self) -> CheckResult:
        threshold = self.detuning_threshold()
        passed = abs(self.chi) >= threshold
        return CheckResult(
            "detuning_check", passed,
            f"χ={self.chi:.3f} MHz (abs={abs(self.chi):.3f}, threshold={threshold:.3f}); "
            f"f_0 before={self.f0_before:.3f} MHz, f_0 after={self.f0_after:.3f} MHz"
        )

    def correct(self, result: EvaluateResult) -> EvaluateResult:
        # Pull figures before super() auto-appends the last one.
        # figure_paths order after analyze(): [0]=before, [1]=after, [2]=combined
        plot_before = self.figure_paths[0].resolve() if len(self.figure_paths) >= 1 else None
        plot_after  = self.figure_paths[1].resolve() if len(self.figure_paths) >= 2 else None
        plot_combined = self.figure_paths[2].resolve() if len(self.figure_paths) >= 3 else None
        self.figure_paths.clear()  # prevent auto-append

        header = (f"## Resonator Spectroscopy Before/After Pi\n"
                  f"Frequencies: {self.start_freq():.3f}–{self.end_freq():.3f} MHz\n"
                  f"Data Path Before: `{self.data_loc_before}`\n"
                  f"Data Path After: `{self.data_loc_after}`\n\n")
        self.report_output.extend([header, plot_combined])

        result = super().correct(result)  # adds check table; no auto-figure since list is empty

        self.report_output.append(
            f"**Detuning (χ): {self.chi:.3f} MHz** "
            f"(f_0 before: {self.f0_before:.3f} MHz, f_0 after: {self.f0_after:.3f} MHz)\n\n"
        )

        self.report_output.extend([
            f"**Before Pi Measurement (SNR: {self.snr_before:.3f}, f_0: {self.f0_before:.3f} MHz):**\n"
            f"```\n{str(self.fit_result_before.lmfit_result.fit_report())}\n```\n\n",
            plot_before,
            f"**After Pi Measurement (SNR: {self.snr_after:.3f}, f_0: {self.f0_after:.3f} MHz):**\n"
            f"```\n{str(self.fit_result_after.lmfit_result.fit_report())}\n```\n\n",
            plot_after,
        ])

        return result
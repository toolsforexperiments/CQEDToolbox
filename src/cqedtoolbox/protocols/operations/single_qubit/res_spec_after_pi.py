import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")

from labcore.analysis import DatasetAnalysis
from labcore.measurement.storage import run_and_save_sweep
from labcore.measurement.sweep import sweep_parameter
from labcore.measurement.record import record_as
from labcore.data.datadict_storage import datadict_from_hdf5

from labcore.protocols.base import ProtocolOperation, OperationStatus, serialize_fit_params, ParamImprovement
from qcui_measurement.protocols.parameters import (
    Repetition,
    ResonatorSpecSteps,
    StartReadoutFrequency,
    EndReadoutFrequency,
    ReadoutGain,
    ReadoutLength,
    Detuning
)
from cqedtoolbox.measurement_lib.qick.single_transmon_v2 import FreqSweepProgram

from cqedtoolbox.fitfuncs.resonators import HangerResponseBruno
from qcui_measurement.protocols.operations.res_spec import SyntheticHangerResonatorData


logger = logging.getLogger(__name__)


class ResonatorSpectroscopyAfterPi(ProtocolOperation):

    SNR_THRESHOLD = 2

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

        self.condition = f"Success if both measurements (before and after pi) have SNR bigger than the current threshold of {self.SNR_THRESHOLD}"

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
        from qcui_measurement.protocols.operations.res_spec import ResonatorSpectroscopy as _RS
        frequencies = np.linspace(self.start_freq(), self.end_freq(), int(self.steps()))

        gen_before = SyntheticHangerResonatorData(
            f0=_RS._SIM_F0, Qi=_RS._SIM_QI, Qc=_RS._SIM_QC,
            A=_RS._SIM_A, phi=_RS._SIM_PHI, noise_amp=_RS._SIM_NOISE_AMP
        )
        signal_before = gen_before.generate(frequencies)
        sweep_before = sweep_parameter("frequencies", frequencies, record_as(lambda f: signal_before, "signal"))
        loc_before, _ = run_and_save_sweep(sweep_before, "data", f"{self.name}_before")

        gen_after = SyntheticHangerResonatorData(
            f0=_RS._SIM_F0 + self._SIM_CHI, Qi=_RS._SIM_QI, Qc=_RS._SIM_QC,
            A=_RS._SIM_A, phi=_RS._SIM_PHI, noise_amp=_RS._SIM_NOISE_AMP
        )
        signal_after = gen_after.generate(frequencies)
        sweep_after = sweep_parameter("frequencies", frequencies, record_as(lambda f: signal_after, "signal"))
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
        sweep_after = FreqSweepProgram()
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

    def evaluate(self) -> OperationStatus:
        """
        Evaluate if both measurements were successful and update chi.
        Success criteria: Both before and after measurements have SNR above threshold.
        """
        header = (f"## Resonator Spectroscopy Before/After Pi\n"
                  f"Measured resonator spectroscopy for frequencies: {self.start_freq():.3f}-{self.end_freq():.3f} MHz with a current SNR threshold of {self.SNR_THRESHOLD}\n"
                  f"Data Path Before: `{self.data_loc_before}`\n"
                  f"Data Path After: `{self.data_loc_after}`\n\n")

        # Get the combined plot (last one added)
        plot_combined = self.figure_paths[2].resolve()
        plot_before = self.figure_paths[0].resolve()
        plot_after = self.figure_paths[1].resolve()

        # Check if both measurements have SNR above threshold
        if self.snr_before >= self.SNR_THRESHOLD and self.snr_after >= self.SNR_THRESHOLD:
            logger.info(f"Both measurements successful. Before SNR: {self.snr_before:.3f}, After SNR: {self.snr_after:.3f}")

            old_value = self.detuning()
            new_value = self.chi

            logger.info(f"Updating chi from {old_value} to {new_value}")
            self.detuning(new_value)

            self.improvements = [ParamImprovement(old_value, new_value, self.detuning)]

            msg_main = (f"### Fit was **SUCCESSFUL**\n"
                       f"Both measurements have SNR above threshold of {self.SNR_THRESHOLD}\n"
                       f"Before Pi SNR: {self.snr_before:.3f}, f_0: {self.f0_before:.3f} MHz\n"
                       f"After Pi SNR: {self.snr_after:.3f}, f_0: {self.f0_after:.3f} MHz\n"
                       f"{self.detuning.name} shift: {old_value:.3f} -> {new_value:.3f} MHz\n\n"
                       f"**Combined Plot:**\n")

            msg_details = (f"\n**Before Pi Measurement:**\n"
                          f"```\n{str(self.fit_result_before.lmfit_result.fit_report())}\n```\n\n")

            msg_after = (f"**After Pi Measurement:**\n"
                        f"```\n{str(self.fit_result_after.lmfit_result.fit_report())}\n```\n\n")

            self.report_output = [header, msg_main, plot_combined, msg_details, plot_before, msg_after, plot_after]

            return OperationStatus.SUCCESS

        # At least one measurement failed
        logger.info(f"At least one measurement failed SNR threshold. Before: {self.snr_before:.3f}, After: {self.snr_after:.3f}")

        failed_measurements = []
        if self.snr_before < self.SNR_THRESHOLD:
            failed_measurements.append(f"Before Pi (SNR: {self.snr_before:.3f})")
        if self.snr_after < self.SNR_THRESHOLD:
            failed_measurements.append(f"After Pi (SNR: {self.snr_after:.3f})")

        msg_main = (f"### Fit was **UNSUCCESSFUL**\n"
                   f"Failed measurements: {', '.join(failed_measurements)}\n"
                   f"Threshold: {self.SNR_THRESHOLD}\n"
                   f"NO value has been changed.\n\n"
                   f"**Combined Plot:**\n")

        msg_details = (f"\n**Before Pi Measurement (SNR: {self.snr_before:.3f}):**\n"
                      f"```\n{str(self.fit_result_before.lmfit_result.fit_report())}\n```\n\n")

        msg_after = (f"**After Pi Measurement (SNR: {self.snr_after:.3f}):**\n"
                    f"```\n{str(self.fit_result_after.lmfit_result.fit_report())}\n```\n\n")

        self.report_output = [header, msg_main, plot_combined, msg_details, plot_before, msg_after, plot_after]

        return OperationStatus.FAILURE
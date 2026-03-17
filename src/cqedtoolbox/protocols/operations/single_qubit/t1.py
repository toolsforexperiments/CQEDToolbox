import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")

from labcore.analysis import DatasetAnalysis
from labcore.analysis.fitfuncs.generic import ExponentialDecay
from labcore.measurement.storage import run_and_save_sweep
from labcore.measurement.sweep import sweep_parameter
from labcore.measurement.record import record_as
from labcore.data.datadict_storage import datadict_from_hdf5

from labcore.protocols.base import ProtocolOperation, OperationStatus, serialize_fit_params, ParamImprovement
from cqedtoolbox.protocols.parameters import (
    Repetition,
    T1Steps,
    QubitGain,
    ReadoutGain,
    ReadoutLength,
    T1
)
from cqedtoolbox.measurement_lib.qick.single_transmon_v2 import T1Program


logger = logging.getLogger(__name__)


class T1Operation(ProtocolOperation):

    SNR_MIN_THRESHOLD = 2
    SNR_MAX_THRESHOLD = 50

    def __init__(self, params):
        super().__init__()

        self.params = params  # Store params for n_echo access

        self._register_inputs(
            repetitions=Repetition(params),
            steps=T1Steps(params),
            qubit_gain=QubitGain(params),
            readout_gain=ReadoutGain(params),
            readout_length=ReadoutLength(params)
        )
        self._register_outputs(
            t1=T1(params)
        )

        self.condition = f"Success if the SNR of any component (real, imaginary, or magnitude) is between {self.SNR_MIN_THRESHOLD} and {self.SNR_MAX_THRESHOLD}"

        self.independents = {"delays": []}
        self.dependents = {"signal": []}

        self.fit_result_re = None
        self.fit_result_imag = None
        self.fit_result_mag = None
        self.snr_re = None
        self.snr_imag = None
        self.snr_mag = None

        self.old_n_echo = None

    _SIM_T1 = 20.0
    _SIM_AMP = 0.5
    _SIM_NOISE_AMP = 0.02

    def _measure_dummy(self) -> Path:
        logger.info("Starting dummy T1 measurement")
        delays = np.linspace(0, 5 * self._SIM_T1, int(self.steps()))
        signal_gen = lambda delays: (self._SIM_AMP * np.exp(-delays / self._SIM_T1)
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
        logger.info("Starting qick T1 measurement")

        # Store old n_echo value and set to 0 for T1 measurement
        self.old_n_echo = self.params.qubit.n_echo()
        logger.debug(f"Storing old n_echo value: {self.old_n_echo}, setting to 0")
        self.params.qubit.n_echo(0)

        try:
            sweep = T1Program()
            logger.debug("Sweep created, running measurement")
            loc, da = run_and_save_sweep(sweep, "data", self.name)
            logger.info("Measurement complete")
            return loc
        finally:
            # Restore n_echo value
            if self.old_n_echo is not None:
                logger.debug(f"Restoring n_echo to {self.old_n_echo}")
                self.params.qubit.n_echo(self.old_n_echo)

    def _load_data_qick(self):
        path = self.data_loc / "data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)

        self.independents["delays"] = data["t"]["values"]
        self.dependents["signal"] = data["signal"]["values"]

    def _fit_exponential_components(self, delays, signal, fig_title="") -> tuple:
        """
        Fit real, imaginary, and magnitude components with ExponentialDecay fits.
        Returns (fit_result_re, fit_result_imag, fit_result_mag, fig_re, fig_imag, fig_mag)
        """
        signal_re = signal.real
        signal_imag = signal.imag
        signal_mag = np.abs(signal)

        # Fit real part
        fit_re = ExponentialDecay(delays, signal_re)
        fit_result_re = fit_re.run(fit_re)
        fit_curve_re = fit_result_re.eval()
        residuals_re = signal_re - fit_curve_re
        amp_re = fit_result_re.params["A"].value
        noise_re = np.std(residuals_re)
        snr_re = np.abs(amp_re / (4 * noise_re))

        # Fit imaginary part
        fit_imag = ExponentialDecay(delays, signal_imag)
        fit_result_imag = fit_imag.run(fit_imag)
        fit_curve_imag = fit_result_imag.eval()
        residuals_imag = signal_imag - fit_curve_imag
        amp_imag = fit_result_imag.params["A"].value
        noise_imag = np.std(residuals_imag)
        snr_imag = np.abs(amp_imag / (4 * noise_imag))

        # Fit magnitude
        fit_mag = ExponentialDecay(delays, signal_mag)
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
        ax_re.set_xlabel("Delay (μs)")
        ax_re.set_ylabel("Signal Real (A.U)")
        ax_re.plot(delays, signal_re, label="Data")
        ax_re.plot(delays, fit_curve_re, label="Fit")
        ax_re.legend()

        # Imaginary plot
        fig_imag, ax_imag = plt.subplots()
        ax_imag.set_title(f"{fig_title} - Imaginary")
        ax_imag.set_xlabel("Delay (μs)")
        ax_imag.set_ylabel("Signal Imaginary (A.U)")
        ax_imag.plot(delays, signal_imag, label="Data")
        ax_imag.plot(delays, fit_curve_imag, label="Fit")
        ax_imag.legend()

        # Magnitude plot
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
            result_re, result_imag, result_mag, fig_re, fig_imag, fig_mag = self._fit_exponential_components(
                self.independents["delays"],
                self.dependents["signal"],
                "T1 Measurement"
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

    def evaluate(self) -> OperationStatus:
        """
        Evaluate if the measurement was successful and update the T1 value.
        Success criteria: At least one component (real, imag, mag) has SNR between min and max thresholds.
        Uses the component with the highest SNR that is still below the maximum threshold.
        """
        # Map component keys to figure paths (in order: real, imag, mag)
        plot_map = {
            "re": self.figure_paths[0].resolve(),
            "imag": self.figure_paths[1].resolve(),
            "mag": self.figure_paths[2].resolve()
        }

        # Collect all components with their SNR values
        snr_dict = {
            "Real": (self.snr_re, self.fit_result_re, "re"),
            "Imaginary": (self.snr_imag, self.fit_result_imag, "imag"),
            "Magnitude": (self.snr_mag, self.fit_result_mag, "mag")
        }

        # Sort by SNR (highest first)
        sorted_components = sorted(snr_dict.items(), key=lambda x: x[1][0], reverse=True)

        # Filter components that have SNR within the valid range
        valid_components = [
            (name, snr, fit, key)
            for name, (snr, fit, key) in sorted_components
            if self.SNR_MIN_THRESHOLD <= snr <= self.SNR_MAX_THRESHOLD
        ]

        header = (f"## T1 Measurement\n"
                  f"Measured T1 relaxation time with SNR range: {self.SNR_MIN_THRESHOLD} - {self.SNR_MAX_THRESHOLD}\n"
                  f"Data Path: `{self.data_loc}`\n\n")

        # Check if we have any valid components
        if valid_components:
            # Pick the component with highest SNR within valid range
            winner_name, winner_snr, winner_fit, winner_key = valid_components[0]

            logger.info(f"{winner_name} component has highest valid SNR of {winner_snr:.3f} (within range {self.SNR_MIN_THRESHOLD}-{self.SNR_MAX_THRESHOLD}). Applying new values")

            old_value = self.t1()
            new_value = winner_fit.params["tau"].value

            logger.info(f"Updating T1 from {old_value} to {new_value}")
            self.t1(new_value)

            self.improvements = [ParamImprovement(old_value, new_value, self.t1)]

            # Main section with winner
            msg_main = (f"### **{winner_name} Component (SELECTED)**\n"
                       f"Fit was **SUCCESSFUL** with {winner_name} SNR of {winner_snr:.3f}\n"
                       f"{self.t1.name} shift: {old_value:.3f} -> {new_value:.3f} μs\n"
                       f"This component was selected because it has the highest SNR within the valid range ({self.SNR_MIN_THRESHOLD}-{self.SNR_MAX_THRESHOLD}).\n\n")

            winner_plot = plot_map[winner_key]

            winner_report = f"**Fit Report:**\n```\n{str(winner_fit.lmfit_result.fit_report())}\n```\n\n"

            # Sections for non-winners
            other_sections = []
            for comp_name, (comp_snr, comp_fit, comp_key) in sorted_components:
                if comp_name == winner_name:
                    continue  # Skip the winner

                # Determine the reason why this component was not selected
                if comp_snr > self.SNR_MAX_THRESHOLD:
                    reason = f"SNR of {comp_snr:.3f} exceeds maximum threshold of {self.SNR_MAX_THRESHOLD}"
                elif comp_snr < self.SNR_MIN_THRESHOLD:
                    reason = f"SNR of {comp_snr:.3f} is below minimum threshold of {self.SNR_MIN_THRESHOLD}"
                else:
                    reason = f"SNR of {comp_snr:.3f} is valid but lower than {winner_name} (SNR={winner_snr:.3f})"

                other_section = f"### **{comp_name} Component (NOT SELECTED)**\n"
                other_sections.append(other_section)
                other_sections.append(plot_map[comp_key])
                other_sections.append(f"Not used: {reason}\n\n**Fit Report:**\n```\n{str(comp_fit.lmfit_result.fit_report())}\n```\n\n")

            self.report_output = [header, msg_main, winner_plot, winner_report] + other_sections

            return OperationStatus.SUCCESS

        # No valid components - all are either too high or too low
        # Find the closest component to the valid range
        best_candidate_name, (best_candidate_snr, best_candidate_fit, best_candidate_key) = sorted_components[0]

        logger.info(f"All components have SNR outside valid range. Best candidate: {best_candidate_name} with SNR {best_candidate_snr:.3f}")

        # Determine failure reason
        all_too_high = all(snr > self.SNR_MAX_THRESHOLD for _, (snr, _, _) in sorted_components)
        all_too_low = all(snr < self.SNR_MIN_THRESHOLD for _, (snr, _, _) in sorted_components)

        if all_too_high:
            failure_reason = f"All components have SNR above maximum threshold of {self.SNR_MAX_THRESHOLD}"
        elif all_too_low:
            failure_reason = f"All components have SNR below minimum threshold of {self.SNR_MIN_THRESHOLD}"
        else:
            failure_reason = f"No components have SNR within valid range ({self.SNR_MIN_THRESHOLD}-{self.SNR_MAX_THRESHOLD})"

        msg_main = (f"### **{best_candidate_name} Component (Best Candidate)**\n"
                   f"Fit was **UNSUCCESSFUL** - {failure_reason}\n"
                   f"Best candidate SNR: {best_candidate_snr:.3f}\n"
                   f"NO value has been changed.\n\n")

        best_plot = plot_map[best_candidate_key]
        best_report = f"**Fit Report:**\n```\n{str(best_candidate_fit.lmfit_result.fit_report())}\n```\n\n"

        # Sections for other components
        other_sections = []
        for comp_name, (comp_snr, comp_fit, comp_key) in sorted_components:
            if comp_name == best_candidate_name:
                continue  # Skip the best candidate

            # Determine status
            if comp_snr > self.SNR_MAX_THRESHOLD:
                status = f"SNR of {comp_snr:.3f} exceeds maximum threshold of {self.SNR_MAX_THRESHOLD}"
            elif comp_snr < self.SNR_MIN_THRESHOLD:
                status = f"SNR of {comp_snr:.3f} is below minimum threshold of {self.SNR_MIN_THRESHOLD}"
            else:
                status = f"SNR of {comp_snr:.3f} is within range but not selected"

            other_section = f"### **{comp_name} Component**\n"
            other_sections.append(other_section)
            other_sections.append(plot_map[comp_key])
            other_sections.append(f"{status}\n\n**Fit Report:**\n```\n{str(comp_fit.lmfit_result.fit_report())}\n```\n\n")

        self.report_output = [header, msg_main, best_plot, best_report] + other_sections

        return OperationStatus.FAILURE
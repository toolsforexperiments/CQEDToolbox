import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")

from labcore.analysis import DatasetAnalysis
from labcore.analysis.fitfuncs.generic import Cosine
from labcore.measurement.storage import run_and_save_sweep
from labcore.measurement import sweep_parameter, record_as
from labcore.data.datadict_storage import datadict_from_hdf5

from labcore.protocols.base import ProtocolOperation, OperationStatus, serialize_fit_params, ParamImprovement
from qcui_measurement.protocols.parameters import (
    Repetition,
    StartQubitGain,
    EndQubitGain,
    QubitGain,
    NumGainSteps
)
from cqedtoolbox.measurement_lib.qick.single_transmon_v2 import AmplitudeRabiProgram


logger = logging.getLogger(__name__)

@dataclass
class SyntheticPowerRabiData:
    pi_amp: float
    noise_amp: float

    def generate(self, gains: float) -> np.complex128:
        signal = (np.cos(2 * np.pi * gains / (2 * self.pi_amp)) + 2) - 1j * (np.cos(2 * np.pi * gains / (2 * self.pi_amp)) + 2)
        noise = self.noise_amp * (np.random.randn() + 1j * np.random.randn())
        return signal + noise


class PowerRabi(ProtocolOperation):

    SNR_THRESHOLD = 2

    _SIM_PI_AMP = 0.5
    _SIM_NOISE_AMP = 0.05

    def __init__(self, params):
        super().__init__()

        self._register_inputs(
            repetitions=Repetition(params),
            start_gain=StartQubitGain(params),
            end_gain=EndQubitGain(params),
            steps_gain=NumGainSteps(params)
        )
        self._register_outputs(
            qubit_gain=QubitGain(params)
        )

        self.condition = f"Success if the SNR of any component (real, imaginary, or magnitude) is bigger than the current threshold of {self.SNR_THRESHOLD}"

        self.independents = {"gains": []}
        self.dependents = {"signal": []}

        self.fit_result_re = None
        self.fit_result_imag = None
        self.fit_result_mag = None
        self.snr_re = None
        self.snr_imag = None
        self.snr_mag = None

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
        # Real plot
        fig_re, ax_re = plt.subplots()
        ax_re.set_title(f"{fig_title} - Real")
        ax_re.set_xlabel("Gain (A.U)")
        ax_re.set_ylabel("Signal Real (A.U)")
        ax_re.plot(gains, signal_re, label="Data")
        ax_re.plot(gains, fit_curve_re, label="Fit")
        ax_re.legend()

        # Imaginary plot
        fig_imag, ax_imag = plt.subplots()
        ax_imag.set_title(f"{fig_title} - Imaginary")
        ax_imag.set_xlabel("Gain (A.U)")
        ax_imag.set_ylabel("Signal Imaginary (A.U)")
        ax_imag.plot(gains, signal_imag, label="Data")
        ax_imag.plot(gains, fit_curve_imag, label="Fit")
        ax_imag.legend()

        # Magnitude plot
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
        Evaluate if the measurement was successful and update the pi pulse gain.
        Success criteria: At least one component (real, imag, mag) has SNR above threshold.
        Uses the component with the highest SNR.
        """
        # Map component keys to figure paths (in order: real, imag, mag)
        plot_map = {
            "re": self.figure_paths[0].resolve(),
            "imag": self.figure_paths[1].resolve(),
            "mag": self.figure_paths[2].resolve()
        }

        # Determine which component has the highest SNR
        snr_dict = {
            "Real": (self.snr_re, self.fit_result_re, "re"),
            "Imaginary": (self.snr_imag, self.fit_result_imag, "imag"),
            "Magnitude": (self.snr_mag, self.fit_result_mag, "mag")
        }

        # Sort by SNR (highest first)
        sorted_components = sorted(snr_dict.items(), key=lambda x: x[1][0], reverse=True)
        winner_name, (winner_snr, winner_fit, winner_key) = sorted_components[0]

        header = (f"## Power Rabi\n"
                  f"Measured power rabi for gain range: {self.start_gain():.3f}-{self.end_gain():.3f} with a current SNR threshold of {self.SNR_THRESHOLD}\n"
                  f"Data Path: `{self.data_loc}`\n\n")

        # Check if the winner's SNR is above threshold
        if winner_snr >= self.SNR_THRESHOLD:
            logger.info(f"{winner_name} component has highest SNR of {winner_snr:.3f}, which is above threshold of {self.SNR_THRESHOLD}. Applying new values")

            old_value = self.qubit_gain()
            # Pi pulse is at half period, so gain = 1/(2*f)
            rabi_frequency = winner_fit.params["f"].value
            new_value = 1 / (2 * rabi_frequency)

            logger.info(f"Updating pi pulse gain from {old_value} to {new_value}")
            self.qubit_gain(new_value)

            self.improvements = [ParamImprovement(old_value, new_value, self.qubit_gain)]

            # Main section with winner
            msg_main = (f"### **{winner_name} Component (SELECTED)**\n"
                       f"Fit was **SUCCESSFUL** with {winner_name} SNR of {winner_snr:.3f}\n"
                       f"Rabi frequency: {rabi_frequency:.6f}\n"
                       f"{self.qubit_gain.name} shift: {old_value:.3f} -> {new_value:.3f}\n"
                       f"This component was selected because it has the highest SNR.\n\n")

            winner_plot = plot_map[winner_key]

            winner_report = f"**Fit Report:**\n```\n{str(winner_fit.lmfit_result.fit_report())}\n```\n\n"

            # Sections for non-winners
            other_sections = []
            for comp_name, (comp_snr, comp_fit, comp_key) in sorted_components[1:]:
                if comp_snr >= self.SNR_THRESHOLD:
                    reason = f"SNR of {comp_snr:.3f} is above threshold but lower than {winner_name} (SNR={winner_snr:.3f})"
                else:
                    reason = f"SNR of {comp_snr:.3f} is below threshold of {self.SNR_THRESHOLD}"

                other_section = f"### **{comp_name} Component (NOT SELECTED)**\n"
                other_sections.append(other_section)
                other_sections.append(plot_map[comp_key])
                other_sections.append(f"Not used: {reason}\n\n**Fit Report:**\n```\n{str(comp_fit.lmfit_result.fit_report())}\n```\n\n")

            self.report_output = [header, msg_main, winner_plot, winner_report] + other_sections

            return OperationStatus.SUCCESS

        # All components failed
        logger.info(f"All components have SNR below threshold. Highest was {winner_name} with SNR {winner_snr:.3f}")

        # Main section with best attempt
        msg_main = (f"### **{winner_name} Component (Highest SNR)**\n"
                   f"Fit was **UNSUCCESSFUL** with {winner_name} SNR of {winner_snr:.3f}, which is below threshold of {self.SNR_THRESHOLD}\n"
                   f"NO value has been changed.\n\n")

        winner_plot = plot_map[winner_key]
        winner_report = f"**Fit Report:**\n```\n{str(winner_fit.lmfit_result.fit_report())}\n```\n\n"

        # Sections for other components
        other_sections = []
        for comp_name, (comp_snr, comp_fit, comp_key) in sorted_components[1:]:
            other_section = f"### **{comp_name} Component**\n"
            other_sections.append(other_section)
            other_sections.append(plot_map[comp_key])
            other_sections.append(f"SNR of {comp_snr:.3f} is below threshold of {self.SNR_THRESHOLD}\n\n**Fit Report:**\n```\n{str(comp_fit.lmfit_result.fit_report())}\n```\n\n")

        self.report_output = [header, msg_main, winner_plot, winner_report] + other_sections

        return OperationStatus.FAILURE
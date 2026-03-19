import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from scipy.constants import h

plt.switch_backend("agg")

from labcore.analysis import DatasetAnalysis, Fit
from labcore.analysis.fitfuncs.generic import Lorentzian
from labcore.measurement.storage import run_and_save_sweep
from labcore.data.datadict_storage import datadict_from_hdf5
from labcore.measurement.sweep import sweep_parameter
from labcore.measurement.record import record_as

from labcore.protocols.base import ProtocolOperation, OperationStatus, serialize_fit_params, ParamImprovement
from cqedtoolbox.protocols.parameters import (
    Repetition,
    SaturationSpecSteps,
    StartSaturationSpecFrequency, EndSaturationSpecFrequency, QubitFrequency
)
from cqedtoolbox.measurement_lib.qick.single_transmon_v2 import PulseProbeSpectroscopy


logger = logging.getLogger(__name__)


@dataclass
class SyntheticSatSpecData:
    fq: float
    delta_fr: float
    f_rabi: float
    gamma1: float
    gamma2: float
    angle: float
    noise_amp: float

    def generate(self, frequencies: ArrayLike) -> ArrayLike:
        """Saturation spec model based on Blais circuit qed eqn 127"""
        signal = 0.5 * self.f_rabi**2 / (
            self.gamma1 * self.gamma2 + (frequencies - self.fq) ** 2 * self.gamma1 / self.gamma2 + self.f_rabi ** 2
        )
        signal_re = signal * np.cos(self.angle) + self.noise_amp * np.random.randn(*frequencies.shape)
        signal_imag = signal * np.sin(self.angle) + self.noise_amp * np.random.randn(*frequencies.shape)
        return signal_re + 1j * signal_imag


class SaturationSpectroscopy(ProtocolOperation):

    SNR_THRESHOLD = 2

    _DUMMY_F_Q = 5e9
    _DUMMY_F_R = 7e9
    _DUMMY_DELTA = _DUMMY_F_R - _DUMMY_F_Q

    _DUMMY_P_IN = 1e-16
    _DUMMY_G = 50e6
    _DUMMY_KAPPA_R = 0.2e6

    _DUMMY_T1 = 50e-6
    _DUMMY_T2 = 50e-6
    _DUMMY_GAMMA_1 = 1 / _DUMMY_T1
    _DUMMY_GAMMA_2 = 1 / (np.pi * _DUMMY_T2)

    _DUMMY_OMEGA = 2 * (_DUMMY_G / _DUMMY_DELTA) * np.sqrt(_DUMMY_KAPPA_R * _DUMMY_P_IN / (h * _DUMMY_F_Q))
    _DUMMY_GAMMA_Q = np.sqrt( (1 / _DUMMY_T2) ** 2 + ((2 * np.pi * _DUMMY_OMEGA) ** 2 * _DUMMY_T1 / _DUMMY_T2) ) / np.pi # blais eq 127

    _DUMMY_NOISE_AMP = 0.05
    _DUMMY_ANGLE = np.pi / 4


    def __init__(self, params):
        super().__init__()

        self._register_inputs(
            repetitions=Repetition(params),
            steps=SaturationSpecSteps(params),
            start_freq=StartSaturationSpecFrequency(params),
            end_freq=EndSaturationSpecFrequency(params),
        )
        self._register_outputs(
            qubit_freq=QubitFrequency(params)
        )

        self.condition = f"Success if the SNR of any component (real, imaginary, or magnitude) is bigger than the current threshold of {self.SNR_THRESHOLD}"

        self.independents = {"frequencies": []}
        self.dependents = {"signal": []}

        self.fit_result_re = None
        self.fit_result_imag = None
        self.fit_result_mag = None
        self.snr_re = None
        self.snr_imag = None
        self.snr_mag = None

    def _measure_qick(self) -> Path:
        logger.info("Starting qick saturation spectroscopy measurement")

        sweep = PulseProbeSpectroscopy()
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

    def _measure_dummy(self) -> Path:
        """Create synthetic saturation spectroscopy data using a sweep"""
        logger.info("Starting dummy saturation spectroscopy measurement")

        # Get parameters for the sweep
        start_f = self.start_freq()
        end_f = self.end_freq()
        num_steps = int(self.steps())

        # Create frequency array
        frequencies = np.linspace(start_f, end_f, num_steps)

        generator = SyntheticSatSpecData(
            fq=self._DUMMY_F_Q,
            delta_fr=self._DUMMY_DELTA,
            f_rabi=self._DUMMY_OMEGA,
            gamma1=self._DUMMY_GAMMA_1,
            gamma2=self._DUMMY_GAMMA_2,
            angle=self._DUMMY_ANGLE,
            noise_amp=self._DUMMY_NOISE_AMP
        )

        # Create sweep over frequencies
        sweep = sweep_parameter('frequencies', frequencies, record_as(generator.generate, 'signal'))

        loc, _ = run_and_save_sweep(sweep, "data", self.name)
        logger.info("Dummy saturation spectroscopy measurement complete.")

        return loc

    def _load_data_dummy(self):
        """Load dummy data from disk (same as _load_data_qick)"""
        logger.info("Loading dummy data from disk")
        path = self.data_loc / "data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)

        self.independents["frequencies"] = data["frequencies"]["values"]
        self.dependents["signal"] = data["signal"]["values"]

    def _fit_lorentzian_components(self, frequencies, signal, fig_title="") -> tuple:
        """
        Fit real, imaginary, and magnitude components with Lorentzian fits.
        Returns (fit_result_re, fit_result_imag, fit_result_mag, fig_re, fig_imag, fig_mag)
        """
        signal_re = signal.real
        signal_imag = signal.imag
        signal_mag = np.abs(signal)

        # Fit real part
        fit_re = Lorentzian(frequencies, signal_re)
        fit_result_re = fit_re.run(fit_re)
        fit_curve_re = fit_result_re.eval()
        residuals_re = signal_re - fit_curve_re
        amp_re = fit_result_re.params["A"].value
        noise_re = np.std(residuals_re)
        snr_re = np.abs(amp_re / (4 * noise_re))

        # Fit imaginary part
        fit_imag = Lorentzian(frequencies, signal_imag)
        fit_result_imag = fit_imag.run(fit_imag)
        fit_curve_imag = fit_result_imag.eval()
        residuals_imag = signal_imag - fit_curve_imag
        amp_imag = fit_result_imag.params["A"].value
        noise_imag = np.std(residuals_imag)
        snr_imag = np.abs(amp_imag / (4 * noise_imag))

        # Fit magnitude
        fit_mag = Lorentzian(frequencies, signal_mag)
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
        ax_re.set_xlabel("Frequency (Hz)")
        ax_re.set_ylabel("Signal Real (A.U)")
        ax_re.plot(frequencies, signal_re, label="Data")
        ax_re.plot(frequencies, fit_curve_re, label="Fit")
        ax_re.legend()

        # Imaginary plot
        fig_imag, ax_imag = plt.subplots()
        ax_imag.set_title(f"{fig_title} - Imaginary")
        ax_imag.set_xlabel("Frequency (Hz)")
        ax_imag.set_ylabel("Signal Imaginary (A.U)")
        ax_imag.plot(frequencies, signal_imag, label="Data")
        ax_imag.plot(frequencies, fit_curve_imag, label="Fit")
        ax_imag.legend()

        # Magnitude plot
        fig_mag, ax_mag = plt.subplots()
        ax_mag.set_title(f"{fig_title} - Magnitude")
        ax_mag.set_xlabel("Frequency (Hz)")
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
            result_re, result_imag, result_mag, fig_re, fig_imag, fig_mag = self._fit_lorentzian_components(
                self.independents["frequencies"],
                self.dependents["signal"],
                "Saturation Spectroscopy"
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
        Evaluate if the measurement was successful and update the qubit frequency.
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

        header = (f"## Saturation Spectroscopy \n"
                  f"Measured saturation spectroscopy for frequencies: {self.start_freq():.3f}-{self.end_freq():.3f} MHz with a current SNR threshold of {self.SNR_THRESHOLD}\n"
                  f"Data Path: `{self.data_loc}`\n\n")

        # Check if the winner's SNR is above threshold
        if winner_snr >= self.SNR_THRESHOLD:
            logger.info(f"{winner_name} component has highest SNR of {winner_snr:.3f}, which is above threshold of {self.SNR_THRESHOLD}. Applying new values")

            old_value = self.qubit_freq()
            new_value = winner_fit.params["x0"].value

            logger.info(f"Updating qubit frequency from {old_value} to {new_value}")
            self.qubit_freq(new_value)

            self.improvements = [ParamImprovement(old_value, new_value, self.qubit_freq)]

            # Main section with winner
            msg_main = (f"### **{winner_name} Component (SELECTED)**\n"
                       f"Fit was **SUCCESSFUL** with {winner_name} SNR of {winner_snr:.3f}\n"
                       f"{self.qubit_freq.name} shift: {old_value:.3f} -> {new_value:.3f}\n"
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

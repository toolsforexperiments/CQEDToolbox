import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

from labcore.analysis import DatasetAnalysis, FitResult
from labcore.measurement.storage import run_and_save_sweep
from labcore.data.datadict_storage import datadict_from_hdf5
from labcore.measurement import sweep_parameter, record_as

from labcore.protocols.base import ProtocolOperation, OperationStatus, serialize_fit_params, ParamImprovement
from qcui_measurement.protocols.parameters import (ReadoutLO, ReadoutIF, Repetition,
                                                   ResonatorSpecSteps, ReadoutGain, ReadoutLength, StartReadoutFrequency, EndReadoutFrequency)
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
class SyntheticHangerResonatorData:
    f0: float
    Qc: float
    Qi: float
    A: float
    phi: float
    noise_amp: float
    
    def generate(self, frequencies: ArrayLike) -> ArrayLike:
        Q_l = 1./(1./self.Qc + 1./self.Qi)
        Q_e_complex = self.Qc * np.exp(-1j * self.phi)
        response = self.A * (1 - (Q_l / Q_e_complex) / (1 + 2j * Q_l * (frequencies - self.f0) / self.f0))
        return response + self.noise_amp * (np.random.randn() + 1j * np.random.randn())

class ResonatorSpectroscopy(ProtocolOperation):

    SNR_THRESHOLD = 2

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
            readout_lo=ReadoutLO(params),
            start_frequency=StartReadoutFrequency(params),
            end_frequency=EndReadoutFrequency(params),
        )
        self._register_outputs(
            readout_if=ReadoutIF(params)
        )
        self.params = params

        self.condition = f"Success if the SNR of the measurement is bigger than the current threshold of {self.SNR_THRESHOLD}"

        self.independents = {"frequencies": []}
        self.dependents = {"signal": []}
        self.unwind_signal = None
        self.magnitude = None
        self.phase = None
        self.snr = None
        self.fit_result = None          

    def _measure_qick(self) -> Path:
        logger.info("Starting qick resonator spectroscopy measurement")

        sweep = FreqSweepProgram()
        logger.debug("Sweep created, running measurement")
        loc, da = run_and_save_sweep(sweep, "data", self.name)
        logger.info("Measurement complete")

        return loc
    
    def _measure_dummy(self):
        logger.info("Starting dummy resonator spectroscopy measurement")
        frequencies = np.linspace(self.start_frequency(), self.end_frequency(), int(self.steps()))
        generator = SyntheticHangerResonatorData(
            f0 = self._SIM_F0,
            Qi = self._SIM_QI,
            Qc = self._SIM_QC,
            A = self._SIM_A,
            phi = self._SIM_PHI,
            noise_amp = self._SIM_NOISE_AMP
        )

        sweep = sweep_parameter("frequencies", frequencies + self.readout_lo(), record_as(generator.generate, "signal"))
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

    def evaluate(self) -> OperationStatus:

        header = (f"## Resonator Spectroscopy \n"
                  f"Measured Resonator spectroscopy for frequencies: {self.independents['frequencies'][0]:.3f}-{self.independents['frequencies'][-1]:.3f} with a current SNR threshold of {self.SNR_THRESHOLD}\n"
                  f"Data Path: `{self.data_loc}`"
                  f"Plot: \n")
        plot_image = self.figure_paths[0].resolve()

        if self.snr >= self.SNR_THRESHOLD:

            logger.info(f"snr of {self.snr} is bigger than threshold of {self.SNR_THRESHOLD}. Applying new values")

            old_value = self.readout_if()
            new_value = self.fit_result.params["f_0"].value

            logger.info(f"Updating f_0 from {old_value} to {new_value}")
            self.readout_if(new_value)
            self.improvements = [ParamImprovement(old_value, new_value, self.readout_if)]

            msg_2 = (f"Fit was **SUCCESSFUL** with and SNR of {self.snr:.3f}. \n"
                     f"{self.readout_if.name} shift: {old_value:.3f} -> {new_value:.3f} \n"
                     f" Fit Report: \n \n ```\n{str(self.fit_result.lmfit_result.fit_report())}\n``` \n")

            self.report_output = [header, plot_image, msg_2]

            return OperationStatus.SUCCESS

        logger.info(f"snr of {self.snr} is smaller than threshold of {self.SNR_THRESHOLD}. Evaluation failed")

        msg_2 = (f"Fit was **UNSUCCESSFUL** with and SNR of {self.snr:.3f}. \n"
                 f"NO value has been changed. \n Fit Report: {str(self.fit_result.lmfit_result.fit_report())} \n")
        self.report_output = [header, plot_image, msg_2]

        return OperationStatus.FAILURE


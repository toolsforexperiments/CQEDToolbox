import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")


from labcore.analysis import DatasetAnalysis
from labcore.measurement.storage import run_and_save_sweep
from labcore.data.datadict_storage import datadict_from_hdf5
from labcore.measurement import sweep_parameter
from labcore.measurement.record import recording, dep, indep

from labcore.protocols.base import ProtocolOperation, OperationStatus, serialize_fit_params, ParamImprovement
from qcui_measurement.protocols.parameters import (
    Repetition,
    Delay,
    StartReadoutFrequency,
    EndReadoutFrequency,
    ReadoutGain,
    ReadoutLength, StartReadoutGain, EndReadoutGain, ResonatorSpecSteps
)
from qcui_measurement.protocols.operations.res_spec import ResonatorSpectroscopy, SyntheticHangerResonatorData
from cqedtoolbox.measurement_lib.qick.single_transmon_v2 import FreqGainSweepProgram


logger = logging.getLogger(__name__)


class ResonatorSpectroscopyVsGain(ProtocolOperation):

    SNR_THRESHOLD = 0.001
    MAX_FREQ_DEVIATION_THRESHOLD = 0.5e6  # MHz threshold for linearity check

    _SIM_N_GAIN_STEPS = 11

    def __init__(self, params):
        super().__init__()

        self._register_inputs(
            repetitions=Repetition(params),
            readout_steps=ResonatorSpecSteps(params),
            start_frequency=StartReadoutFrequency(params),
            end_frequency=EndReadoutFrequency(params),
            readout_length=ReadoutLength(params),
            start_gain=StartReadoutGain(params),
            end_gain=EndReadoutGain(params),
            delay=Delay(params)
        )
        self._register_outputs(
            readout_gain=ReadoutGain(params)
        )

        self.condition = f"Success if every trace has a higher SNR than the current threshold of {self.SNR_THRESHOLD}"

        self.independents = {"frequencies": [], "gains": []}
        self.dependents = {"signal": []}

        self.resonance_frequencies = []
        self.optimal_gain = None
        self.slope = None
        self.max_deviation = None
        self.deviations = []
        self.fit_results = []
        self.snr_values = []

    def _measure_qick(self) -> Path:

        old_delay = self.delay()

        try:
            self.delay(10)
            logger.info("Starting qick resonator spectroscopy vs gain measurement")

            sweep = FreqGainSweepProgram()
            logger.debug("Sweep created, running measurement")
            loc, da = run_and_save_sweep(sweep, "data", self.name)
            logger.info("Measurement complete")

        finally:
            self.delay(old_delay)

        return loc

    def _load_data_qick(self):
        path = self.data_loc / "data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)

        self.independents["frequencies"] = data["freq"]["values"]
        self.independents["gains"] = data["gain"]["values"]
        self.dependents["signal"] = data["signal"]["values"]

    def _measure_dummy(self) -> Path:

        freq_shift_per_gain_unit = -.5e6  # MHz per gain unit

        @recording(
            indep("gains"),
            dep("signal", depends_on=["gains"]),
        )
        def generate_signals(frequencies):
            gains = np.linspace(self.start_gain(), self.end_gain(), self._SIM_N_GAIN_STEPS)
            ret_signal = []
            for i in range(self._SIM_N_GAIN_STEPS):
                shifted_center = ResonatorSpectroscopy._SIM_F0 + freq_shift_per_gain_unit * i
                generator = SyntheticHangerResonatorData(
                    f0=shifted_center,
                    Qi=ResonatorSpectroscopy._SIM_QI,
                    Qc=ResonatorSpectroscopy._SIM_QC,
                    A=ResonatorSpectroscopy._SIM_A,
                    phi=ResonatorSpectroscopy._SIM_PHI,
                    noise_amp=ResonatorSpectroscopy._SIM_NOISE_AMP
                )
                ret_signal.append(generator.generate(frequencies))

            return gains, ret_signal

        sweep = sweep_parameter("frequencies", np.linspace(self.start_frequency(), self.end_frequency(), int(self.readout_steps())), generate_signals)
        loc, _ = run_and_save_sweep(sweep, "data", self.name)

        logger.info("Dummy measurement complete")
        return loc

    def _load_data_dummy(self):
        """Load dummy data from file"""
        path = self.data_loc / "data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)

        self.independents["frequencies"] = np.array([[x for i in range(self._SIM_N_GAIN_STEPS)] for x in data["frequencies"]["values"]])
        self.independents["gains"] = data["gains"]["values"]
        self.dependents["signal"] = data["signal"]["values"]

    def _plot_magnitude_colorbar(self):
        """Create magnitude colorbar plot"""
        mag = np.abs(self.dependents["signal"])
        freq_values = self.independents["frequencies"]
        gain_values = self.independents["gains"]

        # Create meshgrid for plotting
        if len(freq_values.shape) == 1:
            X, Y = np.meshgrid(freq_values, gain_values)
            Z = mag
            if Z.shape != X.shape:
                Z = Z.T  # Transpose if needed to match meshgrid
        else:
            X = freq_values
            Y = gain_values
            Z = mag

        fig, ax = plt.subplots()
        ax.set_title("Resonator Spectroscopy vs Gain Magnitude")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Gain")
        im = ax.pcolormesh(X, Y, Z, cmap="viridis", shading='auto')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Magnitude (A.U.)")

        return fig

    def _plot_gain_vs_resonance_frequency(self, gains, res_f_arr, slope, max_dev_idx):
        """Create gain vs resonance frequency plot"""
        fig, ax = plt.subplots()
        ax.set_title("Gain vs Resonator Frequency")
        ax.set_xlabel("Gain")
        ax.set_ylabel("Resonance Frequency (MHz)")

        ax.plot(gains, res_f_arr, marker='.', linestyle='-', label='Data')
        ax.plot([gains[0], gains[-1]], [res_f_arr[0], res_f_arr[-1]], label='Linear Fit')
        ax.axvline(x=gains[max_dev_idx], linestyle='--', color='red', label='Max Deviation')
        ax.legend()

        return fig

    def analyze(self):
        with DatasetAnalysis(self.data_loc, self.name) as ds:
            # Create magnitude colorbar plot
            mag_fig = self._plot_magnitude_colorbar()
            ds.add_figure("magnitude_colorbar", fig=mag_fig)

            image_path = ds._new_file_path(ds.savefolders[1], "magnitude_colorbar", suffix="png")
            self.figure_paths.append(image_path)

            # Analyze each gain trace individually
            gains = self.independents["gains"][0]
            res_f_arr = []

            for i, g in enumerate(gains):
                trace_signal = self.dependents["signal"].T[i]  # Transpose to achieve gain as axis 0
                freqs = self.independents["frequencies"].T[i]

                folder_name = f"resonator_spec_vs_gain_i={i}_g={g}"

                # Use the static method from ResonatorSpectroscopy
                ret = ResonatorSpectroscopy.add_mag_and_unwind_and_fit(
                    freqs, trace_signal, f"Gain = {g}"
                )

                self.fit_results.append(ret.fit_result)
                self.snr_values.append(ret.snr)
                res_f_arr.append(ret.fit_result.params["f_0"].value)

                # Save individual trace analysis
                with DatasetAnalysis(self.data_loc, folder_name) as trace_ds:
                    trace_ds.add(
                        fit_curve=ret.fit_curve,
                        fit_result=ret.fit_result,
                        params=serialize_fit_params(ret.fit_result.params),
                        snr=float(ret.snr)
                    )
                    trace_ds.add_figure(folder_name, fig=ret.fig)

                    image_path = trace_ds._new_file_path(trace_ds.savefolders[1], folder_name, suffix="png")
                    self.figure_paths.append(image_path)

            self.resonance_frequencies = res_f_arr

            # Calculate linearity and find optimal gain
            self.slope = (res_f_arr[-1] - res_f_arr[0]) / (gains[-1] - gains[0])
            self.deviations = []
            for g, f in zip(gains, res_f_arr):
                val = self.slope * (g - gains[0]) + res_f_arr[0]
                self.deviations.append(np.abs(f - val))

            max_dev_idx = np.argmax(self.deviations)
            self.max_deviation = self.deviations[max_dev_idx]
            self.optimal_gain = gains[max_dev_idx]

            # Create gain vs resonance frequency plot
            gain_vs_freq_fig = self._plot_gain_vs_resonance_frequency(
                gains, res_f_arr, self.slope, max_dev_idx
            )
            ds.add_figure("gain_vs_frequency", fig=gain_vs_freq_fig)

            image_path = ds._new_file_path(ds.savefolders[1], "gain_vs_frequency", suffix="png")
            self.figure_paths.append(image_path)

            # Store results
            ds.add(
                optimal_gain=self.optimal_gain,
                resonance_frequencies=res_f_arr,
                max_deviation=self.max_deviation,
                deviations=self.deviations,
                slope=self.slope
            )

    def evaluate(self) -> OperationStatus:
        """
        Evaluate if the measurement was successful and update the optimal gain.
        Success criteria:
        1. All traces have SNR above threshold
        2. Maximum deviation from linearity is within acceptable range
        """
        current_report = []
        header = (f"## Resonator Spectroscopy vs Gain \n"
                  f"Measured Resonator spectroscopy for frequencies: {self.start_frequency():.3f}-{self.end_frequency():.3f} MHz\n"
                  f"Gain range: {self.start_gain():.3f}-{self.end_gain():.3f}\n"
                  f"SNR threshold: {self.SNR_THRESHOLD}, Max deviation threshold: {self.MAX_FREQ_DEVIATION_THRESHOLD} MHz\n"
                  f"Data Path: `{self.data_loc}`\n\n")
        current_report.append(header)

        # Add main plots (magnitude colorbar and gain vs frequency)
        main_plots = "### Main Plots\n"
        current_report.append(main_plots)

        color_bar_section = "**Magnitude Colorbar:**\n"
        color_bar_plot = self.figure_paths.pop(0)
        gain_vs_frequency_section = "**Gain vs Frequency:**\n"
        gain_vs_frequency_plot = self.figure_paths.pop(-1)

        current_report.append(color_bar_section)
        current_report.append(color_bar_plot)
        current_report.append(gain_vs_frequency_section)
        current_report.append(gain_vs_frequency_plot)


        # Check SNR for all traces
        if not all(snr >= self.SNR_THRESHOLD for snr in self.snr_values):
            failed_snr_indices = [i for i, snr in enumerate(self.snr_values) if snr < self.SNR_THRESHOLD]
            msg = (f"Fit was **UNSUCCESSFUL** - Some traces have SNR below threshold {self.SNR_THRESHOLD}\n"
                   f"Failed trace indices: {failed_snr_indices}\n"
                   f"SNR values: {[f'{snr:.3f}' for snr in self.snr_values]}\n")
            current_report.append(msg)
            self.report_output = current_report
            logger.warning(f"Some traces have SNR below threshold {self.SNR_THRESHOLD}")
            return OperationStatus.FAILURE

        # Check deviation from linearity using pre-calculated values
        if self.max_deviation > self.MAX_FREQ_DEVIATION_THRESHOLD:
            msg = (f"Fit was **UNSUCCESSFUL** - Maximum frequency deviation {self.max_deviation:.3f} Hz exceeds threshold {self.MAX_FREQ_DEVIATION_THRESHOLD} MHz\n"
                   f"Linearity slope: {self.slope:.6f} MHz/gain\n"
                   f"Max deviation: {self.max_deviation:.3f} MHz\n")
            current_report.append(msg)
            self.report_output = current_report
            logger.warning(f"Maximum frequency deviation {self.max_deviation:.3f} Hz exceeds threshold {self.MAX_FREQ_DEVIATION_THRESHOLD} MHz")
            return OperationStatus.FAILURE

        # Update the optimal gain parameter
        old_value = self.readout_gain()
        new_value = self.optimal_gain

        logger.info(f"Updating gain from {old_value} to {new_value}")
        self.readout_gain(new_value)

        self.improvements = [ParamImprovement(old_value, new_value, self.readout_gain)]

        msg = (f"Fit was **SUCCESSFUL**\n"
               f"{self.readout_gain.name} update: {old_value:.3f} -> {new_value:.3f}\n"
               f"Linearity slope: {self.slope:.6f} MHz/gain\n"
               f"Max deviation: {self.max_deviation:.3f} MHz\n"
               f"All {len(self.snr_values)} traces have SNR >= {self.SNR_THRESHOLD}\n")

        current_report.append(msg)


        trace_section = "\n### Individual Gain Traces\n"
        current_report.append(trace_section)
        for i, (fig_path, g) in enumerate(zip(self.figure_paths, self.independents["gains"][0])):

            trace_info = (f"\n**Trace {i}: Gain = {g:.3f}**\n"
                         f"- SNR: {self.snr_values[i]:.3f}\n"
                         f"- f_0: {self.resonance_frequencies[i]:.3f} MHz\n")
            current_report.append(trace_info)
            current_report.append(fig_path)

        self.report_output = current_report

        logger.info(f"Evaluation successful. Optimal gain: {self.optimal_gain}")
        return OperationStatus.SUCCESS
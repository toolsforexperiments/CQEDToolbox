import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")

from labcore.analysis import DatasetAnalysis
from labcore.measurement.storage import run_and_save_sweep
from labcore.data.datadict_storage import datadict_from_hdf5, load_as_xr
from labcore.measurement import sweep_parameter
from labcore.measurement.record import recording, dep, indep

from labcore.protocols.base import (ProtocolOperation, OperationStatus, serialize_fit_params,
                                    CorrectionParameter, CheckResult, Correction, EvaluateResult, PlatformTypes)
from cqedtoolbox.protocols.operations import ResonatorGeometry
from cqedtoolbox.protocols.parameters import (
    Repetition,
    StartReadoutFrequency,
    EndReadoutFrequency,
    ReadoutGain,
    ReadoutLength, StartReadoutGain, EndReadoutGain, ResonatorSpecSteps, ResonatorSpecVsGainSteps,
    nestedAttributeFromString,
)
from cqedtoolbox.protocols.operations.single_qubit.res_spec import (
    ResonatorSpectroscopy,
    SyntheticHangerResonatorData,
)
from cqedtoolbox.measurement_lib.opx.advanced.qubit_tuneup import measure_pulse_resonator_spec_vs_readout_amp
from cqedtoolbox.measurement_lib.qick.single_transmon_v2 import FreqGainSweepProgram


logger = logging.getLogger(__name__)


@dataclass
class ResSpecVsGainSNRThreshold(CorrectionParameter):
    name: str = field(default="res_spec_vs_gain_snr_threshold", init=False)
    description: str = field(default="SNR threshold for low-gain quality check", init=False)

    def _qick_getter(self):
        return self.params.corrections.res_spec_vs_gain.snr()

    def _qick_setter(self, v):
        self.params.corrections.res_spec_vs_gain.snr(v)

    def _opx_getter(self):
        return self.params.corrections.res_spec_vs_gain.snr()

    def _opx_setter(self, v):
        self.params.corrections.res_spec_vs_gain.snr(v)


@dataclass
class ResSpecVsGainMaxFitParamError(CorrectionParameter):
    name: str = field(default="res_spec_vs_gain_max_fit_param_error", init=False)
    description: str = field(default="Max fractional fit parameter error (e.g. 1.0 = 100%)", init=False)

    def _qick_getter(self):
        return self.params.corrections.res_spec_vs_gain.max_fit_param_error()

    def _qick_setter(self, v):
        self.params.corrections.res_spec_vs_gain.max_fit_param_error(v)

    def _opx_getter(self):
        return self.params.corrections.res_spec_vs_gain.max_fit_param_error()

    def _opx_setter(self, v):
        self.params.corrections.res_spec_vs_gain.max_fit_param_error(v)


@dataclass
class ResSpecVsGainHighSNRThreshold(CorrectionParameter):
    name: str = field(default="res_spec_vs_gain_high_snr_threshold", init=False)
    description: str = field(default="High SNR threshold — at least one trace must exceed this", init=False)

    def _qick_getter(self):
        return self.params.corrections.res_spec_vs_gain.high_snr()

    def _qick_setter(self, v):
        self.params.corrections.res_spec_vs_gain.high_snr(v)

    def _opx_getter(self):
        return self.params.corrections.res_spec_vs_gain.high_snr()

    def _opx_setter(self, v):
        self.params.corrections.res_spec_vs_gain.high_snr(v)


@dataclass
class ResSpecVsGainRepetitionFactor(CorrectionParameter):
    name: str = field(default="res_spec_vs_gain_repetition_factor", init=False)
    description: str = field(default="Factor by which repetitions are increased on retry", init=False)

    def _qick_getter(self):
        return self.params.corrections.res_spec_vs_gain.rep_factor()

    def _qick_setter(self, v):
        self.params.corrections.res_spec_vs_gain.rep_factor(v)

    def _opx_getter(self):
        return self.params.corrections.res_spec_vs_gain.rep_factor()

    def _opx_setter(self, v):
        self.params.corrections.res_spec_vs_gain.rep_factor(v)


@dataclass
class ResSpecVsGainMaxRepetitionIncreases(CorrectionParameter):
    name: str = field(default="res_spec_vs_gain_max_rep_increases", init=False)
    description: str = field(default="Maximum number of repetition increases to attempt", init=False)

    def _qick_getter(self):
        return int(self.params.corrections.res_spec_vs_gain.max_rep_increases())

    def _qick_setter(self, v):
        self.params.corrections.res_spec_vs_gain.max_rep_increases(v)

    def _opx_getter(self):
        return int(self.params.corrections.res_spec_vs_gain.max_rep_increases())

    def _opx_setter(self, v):
        self.params.corrections.res_spec_vs_gain.max_rep_increases(v)


class IncreaseRepetitionsCorrection(Correction):
    name = "increase_repetitions"
    description = "Increase repetition count until at least one trace has high SNR"
    triggered_by = "high_snr_check"

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


class ResonatorSpectroscopyVsGain(ProtocolOperation):

    _SIM_N_GAIN_STEPS = 11

    def __init__(self, params, geometry: ResonatorGeometry | str):
        super().__init__()
        self.params = params

        if isinstance(geometry, str):
            try:
                geometry = ResonatorGeometry(geometry.lower())
            except ValueError as err:
                valid = ", ".join(g.value for g in ResonatorGeometry)
                raise ValueError(
                    f"Unsupported resonator geometry '{geometry}'. Expected one of: {valid}"
                ) from err

        self.geometry = geometry
        self._fit_cls = geometry.fit_cls

        self._register_inputs(
            repetitions=Repetition(params),
            readout_steps=ResonatorSpecSteps(params),
            start_frequency=StartReadoutFrequency(params),
            end_frequency=EndReadoutFrequency(params),
            readout_length=ReadoutLength(params),
            start_gain=StartReadoutGain(params),
            end_gain=EndReadoutGain(params),
            gain_steps=ResonatorSpecVsGainSteps(params),
        )
        self._register_outputs(
            readout_gain=ReadoutGain(params)
        )

        self._register_correction_params(
            snr_threshold=ResSpecVsGainSNRThreshold(params),
            max_fit_param_error=ResSpecVsGainMaxFitParamError(params),
            high_snr_threshold=ResSpecVsGainHighSNRThreshold(params),
            repetition_factor=ResSpecVsGainRepetitionFactor(params),
            max_repetition_increases=ResSpecVsGainMaxRepetitionIncreases(params),
        )

        self._increase_repetitions = IncreaseRepetitionsCorrection(
            self.repetitions,
            self.repetition_factor,
            self.max_repetition_increases,
        )

        self._register_check("low_gain_quality_check", self._check_low_gain_quality)
        self._register_check("high_snr_check", self._check_high_snr, self._increase_repetitions)

        self._register_success_update(self.readout_gain, lambda: self.optimal_gain)

        self.independents = {"frequencies": [], "gains": []}
        self.dependents = {"signal": []}

        self.resonance_frequencies = []
        self.optimal_gain = None
        self.slope = None
        self.max_deviation = None
        self.deviations = []
        self.fit_results = []
        self.snr_values = []
        self.trace_valid = []
        self.trace_invalid_reasons = []

    def _trace_in_range(self, freqs, fit_result) -> bool:
        f0 = fit_result.params["f_0"].value
        freqs = np.asarray(freqs, dtype=float)
        return float(np.min(freqs)) <= f0 <= float(np.max(freqs))

    def _trace_invalid_reason(self, freqs, fit_result, snr) -> str | None:
        if not self._trace_in_range(freqs, fit_result):
            f0 = fit_result.params["f_0"].value
            fmin = float(np.min(freqs))
            fmax = float(np.max(freqs))
            return f"f_0={f0:.6g} outside sweep=[{fmin:.6g}, {fmax:.6g}]"

        if snr < self.snr_threshold():
            return f"SNR={snr:.3f} < {self.snr_threshold():.3f}"

        max_error = self.max_fit_param_error()
        for pname, param in fit_result.params.items():
            if pname in ["transmission_slope", "phase_slope", "phase_offset"]:
                continue
            if param.stderr is None:
                return f"{pname}: no stderr"
            if param.value == 0 or abs(param.stderr / param.value) > max_error:
                pct = abs(param.stderr / param.value) * 100 if param.value != 0 else float("inf")
                return f"{pname}: {pct:.0f}% error"

        return None

    def _measure_qick(self) -> Path:
        logger.info("Starting qick resonator spectroscopy vs gain measurement")

        sweep = FreqGainSweepProgram()
        logger.debug("Sweep created, running measurement")
        loc, da = run_and_save_sweep(sweep, "data", self.name)
        logger.info("Measurement complete")

        return loc

    def _measure_opx(self) -> Path:
        logger.info("Starting opx resonator spectroscopy vs gain measurement")
        loc = measure_pulse_resonator_spec_vs_readout_amp()
        logger.info("Measurement complete")
        return loc

    def _load_data_qick(self):
        path = self.data_loc / "data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)

        self.independents["frequencies"] = data["freq"]["values"]
        self.independents["gains"] = data["gain"]["values"]
        self.dependents["signal"] = data["signal"]["values"]

    def _load_data_opx(self):
        data = load_as_xr(self.data_loc).mean("repetition")
        q = nestedAttributeFromString(self.params, "active.qubit")()
        lo = nestedAttributeFromString(self.params, f"{q}.readout.LO")()
        self.independents["frequencies"] = data["ssb_frequency"].values + lo
        self.independents["gains"] = data["amp"].values
        self.dependents["signal"] = data["signal_Re"].values + 1j * data["signal_Im"].values

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

    def _plot_gain_vs_resonance_frequency(self, gains, res_f_arr, optimal_gain):
        """Create gain vs resonance frequency plot"""
        fig, ax = plt.subplots()
        ax.set_title("Gain vs Resonator Frequency")
        ax.set_xlabel("Gain")
        ax.set_ylabel("Resonance Frequency (MHz)")

        ax.plot(gains, res_f_arr, marker='.', linestyle='-', label='Data')
        ax.plot([gains[0], gains[-1]], [res_f_arr[0], res_f_arr[-1]], label='Linear Fit')
        if optimal_gain is not None:
            ax.axvline(x=optimal_gain, linestyle='--', color='red', label='Selected Gain')
        ax.legend()

        return fig

    def _plot_snr_vs_gain(self, gains, snr_values):
        """Create SNR vs gain plot"""
        fig, ax = plt.subplots()
        ax.set_title("SNR vs Gain")
        ax.set_xlabel("Gain")
        ax.set_ylabel("SNR")
        ax.plot(gains, snr_values, marker='.', linestyle='-')
        return fig

    def analyze(self):
        with DatasetAnalysis(self.data_loc, self.name) as ds:
            # Create magnitude colorbar plot
            mag_fig = self._plot_magnitude_colorbar()
            ds.add_figure("magnitude_colorbar", fig=mag_fig)

            image_path = ds._new_file_path(ds.savefolders[1], "magnitude_colorbar", suffix="png")
            self.figure_paths.append(image_path)

            # Analyze each gain trace individually
            gains = self.independents["gains"]
            res_f_arr = []
            self.fit_results = []
            self.snr_values = []
            self.trace_valid = []
            self.trace_invalid_reasons = []

            for i, g in enumerate(gains):
                folder_name = f"resonator_spec_vs_gain_i={i}_g={g}"

                # FIXME: is this the correct pattern for qick?
                if self.platform_type == PlatformTypes.OPX:
                    trace_signal = self.dependents["signal"][i]
                    freqs = self.independents["frequencies"]
                else:
                    trace_signal = self.dependents["signal"].T[i]  # Transpose to achieve gain as axis 0
                    freqs = self.independents["frequencies"].T[i]

                # Use the static method from ResonatorSpectroscopy
                ret = ResonatorSpectroscopy.add_mag_and_unwind_and_fit(
                    freqs, trace_signal, self.platform_type, self._fit_cls, f"Gain = {g}"
                )

                _excluded = {"transmission_slope", "phase_slope", "phase_offset"}
                _null_stderr_params = [
                    pname for pname, param in ret.fit_result.params.items()
                    if pname not in _excluded and param.stderr is None
                ]
                if _null_stderr_params:
                    logger.warning(
                        f"Trace {i} (gain={g}): stderr is None for params "
                        f"{_null_stderr_params} — re-fitting"
                    )
                    ret = ResonatorSpectroscopy.add_mag_and_unwind_and_fit(
                        freqs, trace_signal, self.platform_type, self._fit_cls, f"Gain = {g}"
                    )

                self.fit_results.append(ret.fit_result)
                self.snr_values.append(ret.snr)
                res_f_arr.append(ret.fit_result.params["f_0"].value)
                invalid_reason = self._trace_invalid_reason(freqs, ret.fit_result, ret.snr)
                self.trace_valid.append(invalid_reason is None)
                self.trace_invalid_reasons.append(invalid_reason)

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

            passing_indices = [i for i, valid in enumerate(self.trace_valid) if valid]

            if passing_indices:
                best_idx = max(passing_indices, key=lambda i: self.snr_values[i])
                self.optimal_gain = gains[best_idx]
            else:
                self.optimal_gain = None

            # SNR vs gain plot (appended before gain_vs_frequency)
            snr_fig = self._plot_snr_vs_gain(gains, self.snr_values)
            ds.add_figure("snr_vs_gain", fig=snr_fig)
            image_path = ds._new_file_path(ds.savefolders[1], "snr_vs_gain", suffix="png")
            self.figure_paths.append(image_path)

            # Linearity info (kept for the plot and stored data)
            self.slope = (res_f_arr[-1] - res_f_arr[0]) / (gains[-1] - gains[0])
            self.deviations = [np.abs(f - (self.slope * (g - gains[0]) + res_f_arr[0]))
                               for g, f in zip(gains, res_f_arr)]
            self.max_deviation = max(self.deviations)

            # Create gain vs resonance frequency plot (last)
            gain_vs_freq_fig = self._plot_gain_vs_resonance_frequency(
                gains, res_f_arr, self.optimal_gain
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

    def _check_low_gain_quality(self) -> CheckResult:
        """Quality check (SNR + fit error) for the first 50% of gain traces."""
        n_low = max(1, len(self.snr_values) // 2)
        threshold = self.snr_threshold()
        max_error = self.max_fit_param_error()

        failures = []
        for i in range(n_low):
            if not self.trace_valid[i]:
                failures.append(f"trace {i}: {self.trace_invalid_reasons[i]}")

        passed = len(failures) == 0
        desc = (f"first {n_low} traces pass quality check" if passed
                else "; ".join(failures))
        return CheckResult("low_gain_quality_check", passed, desc)

    def _check_high_snr(self) -> CheckResult:
        """At least one valid trace must exceed the high SNR threshold."""
        threshold = self.high_snr_threshold()
        valid_indices = [i for i, valid in enumerate(self.trace_valid) if valid]
        if not valid_indices:
            return CheckResult(
                "high_snr_check",
                False,
                "no valid traces remain after fit-range and fit-quality filtering",
            )

        best_idx = max(valid_indices, key=lambda i: self.snr_values[i])
        best = self.snr_values[best_idx]
        passed = best >= threshold
        desc = (f"best valid trace={best_idx}, SNR={best:.3f} ≥ {threshold:.3f}" if passed
                else f"best valid trace={best_idx}, SNR={best:.3f} < {threshold:.3f}")
        return CheckResult("high_snr_check", passed, desc)

    def correct(self, result: EvaluateResult) -> EvaluateResult:
        # figure_paths order: [0]=colorbar, [1..N-1]=traces, [-2]=snr_vs_gain, [-1]=gain_vs_freq
        colorbar_plot = self.figure_paths.pop(0) if len(self.figure_paths) >= 3 else None
        gain_vs_freq_plot = self.figure_paths.pop(-1) if self.figure_paths else None
        snr_vs_gain_plot = self.figure_paths.pop(-1) if self.figure_paths else None
        trace_figures = list(self.figure_paths)
        self.figure_paths.clear()

        header = (f"## Resonator Spectroscopy vs Gain\n"
                  f"Frequencies: {self.start_frequency():.3f}–{self.end_frequency():.3f} MHz, "
                  f"Gains: {self.start_gain():.3f}–{self.end_gain():.3f}\n"
                  f"Data Path: `{self.data_loc}`\n\n")
        self.report_output.extend([
            header,
            "### Main Plots\n",
            "**Magnitude Colorbar:**\n", colorbar_plot,
            "**Gain vs Frequency:**\n", gain_vs_freq_plot,
            "**SNR vs Gain:**\n", snr_vs_gain_plot,
        ])

        result = super().correct(result)  # check table + success update (writes readout_gain)

        if result.status == OperationStatus.SUCCESS:
            gains = self.independents["gains"]
            self.report_output.append("\n### Individual Gain Traces\n")
            for i, (fig_path, g) in enumerate(zip(trace_figures, gains)):
                validity = "valid" if self.trace_valid[i] else f"invalid ({self.trace_invalid_reasons[i]})"
                self.report_output.extend([
                    f"\n**Trace {i}: Gain = {g:.3f}**\n"
                    f"- SNR: {self.snr_values[i]:.3f}\n"
                    f"- f_0: {self.resonance_frequencies[i]:.3f} MHz\n"
                    f"- Status: {validity}\n",
                    fig_path,
                ])

        return result

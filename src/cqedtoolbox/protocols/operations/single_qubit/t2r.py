import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")

from labcore.analysis import DatasetAnalysis
from labcore.analysis.fitfuncs.generic import ExponentiallyDecayingSine
from labcore.measurement.storage import run_and_save_sweep
from labcore.measurement.sweep import sweep_parameter
from labcore.measurement.record import record_as
from labcore.data.datadict_storage import datadict_from_hdf5, load_as_xr

from labcore.protocols.base import (
    ProtocolOperation, OperationStatus, serialize_fit_params,
    CorrectionParameter, CheckResult, Correction, EvaluateResult, PlatformTypes
)
from cqedtoolbox.protocols.parameters import (
    Repetition,
    ArtificialDetuning,
    T2RSteps,
    QubitGain,
    ReadoutGain,
    ReadoutLength,
    T2R,
)
from cqedtoolbox.measurement_lib.opx.advanced.qubit_tuneup import measure_t2
from cqedtoolbox.measurement_lib.qick.single_transmon_v2 import T2RProgram
from cqedtoolbox.readout.qubit_readout import rotate_complex_qubit_data


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CorrectionParameter subclasses
# ---------------------------------------------------------------------------

@dataclass
class SNRMinThreshold(CorrectionParameter):
    name: str = field(default="t2r_snr_min_threshold", init=False)
    description: str = field(default="Minimum SNR for a valid T2R fit component", init=False)

    def _qick_getter(self): return self.params.corrections.t2r.snr_min()
    def _qick_setter(self, v): self.params.corrections.t2r.snr_min(v)
    def _dummy_getter(self): return self.params.corrections.t2r.snr_min()
    def _dummy_setter(self, v): self.params.corrections.t2r.snr_min(v)
    def _opx_getter(self): return self.params.corrections.t2r.snr_min()
    def _opx_setter(self, v): self.params.corrections.t2r.snr_min(v)


@dataclass
class MaxFitParamError(CorrectionParameter):
    name: str = field(default="t2r_max_fit_param_error", init=False)
    description: str = field(default="Maximum allowed fractional fit parameter error (e.g. 1.0 = 100%)", init=False)

    def _qick_getter(self): return self.params.corrections.t2r.max_fit_param_error()
    def _qick_setter(self, v): self.params.corrections.t2r.max_fit_param_error(v)
    def _dummy_getter(self): return self.params.corrections.t2r.max_fit_param_error()
    def _dummy_setter(self, v): self.params.corrections.t2r.max_fit_param_error(v)
    def _opx_getter(self): return self.params.corrections.t2r.max_fit_param_error()
    def _opx_setter(self, v): self.params.corrections.t2r.max_fit_param_error(v)


@dataclass
class AveragingIncreaseFactor(CorrectionParameter):
    name: str = field(default="t2r_averaging_factor", init=False)
    description: str = field(default="Factor by which to increase repetitions", init=False)

    def _qick_getter(self): return self.params.corrections.t2r.averaging_factor()
    def _qick_setter(self, v): self.params.corrections.t2r.averaging_factor(v)
    def _dummy_getter(self): return self.params.corrections.t2r.averaging_factor()
    def _dummy_setter(self, v): self.params.corrections.t2r.averaging_factor(v)
    def _opx_getter(self): return self.params.corrections.t2r.averaging_factor()
    def _opx_setter(self, v): self.params.corrections.t2r.averaging_factor(v)


@dataclass
class MaxAveragingIncreases(CorrectionParameter):
    name: str = field(default="t2r_max_averaging_increases", init=False)
    description: str = field(default="Maximum number of averaging increases to try", init=False)

    def _qick_getter(self): return int(self.params.corrections.t2r.max_averaging_increases())
    def _qick_setter(self, v): self.params.corrections.t2r.max_averaging_increases(v)
    def _dummy_getter(self): return int(self.params.corrections.t2r.max_averaging_increases())
    def _dummy_setter(self, v): self.params.corrections.t2r.max_averaging_increases(v)
    def _opx_getter(self): return int(self.params.corrections.t2r.max_averaging_increases())
    def _opx_setter(self, v): self.params.corrections.t2r.max_averaging_increases(v)


# ---------------------------------------------------------------------------
# Correction subclasses
# ---------------------------------------------------------------------------

class IncreaseAveragingCorrection(Correction):
    name = "increase_averaging"
    description = "Increase number of repetitions"
    triggered_by = "quality_check"

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

    # TODO: Move correction restoration and terminal cleanup into labcore's
    # correction lifecycle, including cleanup when an attempt raises or is
    # interrupted (currently restore() only runs from correct()).
    def restore(self) -> None:
        if self._original_reps is not None:
            self.reps_param(self._original_reps)
        self._original_reps = None
        self._count = 0
        self._last_change = ""


# ---------------------------------------------------------------------------
# Operation
# ---------------------------------------------------------------------------

class T2ROperation(ProtocolOperation):

    # Offset guesses smaller than this fraction of |A| are snapped to exactly 0.
    # rotate_complex_qubit_data mean-centers the data (PCA), so the mean-based
    # offset guess can be ~1e-16. lmfit's leastsq uses a relative finite-difference
    # step, which is then too small to change the model: `of` never moves, the
    # covariance is singular, and all stderrs come back None.
    _OFFSET_SNAP_REL = 1e-4

    _SIM_T2R = 10.0
    _SIM_DETUNING = 0.05
    _SIM_AMP = 0.35 + 0.35j
    _SIM_NOISE_AMP = 0.02
    _SIM_OFFSET = 0.4 + 0.4j
    def __init__(self, params):
        super().__init__()

        self.params = params

        self._register_inputs(
            repetitions=Repetition(params),
            steps=T2RSteps(params),
            artificial_detuning=ArtificialDetuning(params),
            qubit_gain=QubitGain(params),
            readout_gain=ReadoutGain(params),
            readout_length=ReadoutLength(params),
        )
        self._register_outputs(
            t2r=T2R(params)
        )

        self._register_correction_params(
            snr_min_threshold=SNRMinThreshold(params),
            max_fit_param_error=MaxFitParamError(params),
            averaging_increase_factor=AveragingIncreaseFactor(params),
            max_averaging_increases=MaxAveragingIncreases(params),
        )

        self._increase_averaging = IncreaseAveragingCorrection(
            self.repetitions,
            self.averaging_increase_factor,
            self.max_averaging_increases,
        )

        self._register_check("quality_check", self._check_quality, self._increase_averaging)

        self._register_success_update(self.t2r, lambda: self.fit_result.params["tau"].value)

        self.independents = {"delays": []}
        self.dependents = {"signal": []}

        self.fit_result = None
        self.residuals = None
        self.snr = None
        self.fit_inverted = False

    def _measure_dummy(self) -> Path:
        logger.info("Starting dummy T2 Ramsey measurement")
        delays = np.linspace(0, 5 * self._SIM_T2R, int(self.steps()))
        signal_gen = lambda delays: (self._SIM_AMP * np.exp(-delays / self._SIM_T2R) * np.exp(2j * np.pi * self._SIM_DETUNING * delays)
                  + self._SIM_OFFSET + self._SIM_NOISE_AMP * (np.random.randn() + 1j * np.random.randn()))
        sweep = sweep_parameter("delays", delays, record_as(signal_gen, "signal"))
        loc, _ = run_and_save_sweep(sweep, "data", self.name)
        logger.info("Dummy measurement complete")
        return loc

    def _load_data_dummy(self):
        data = load_as_xr(self.data_loc)
        rotated = rotate_complex_qubit_data(data)[0]
        self.independents["delays"] = rotated["delays"].values
        self.dependents["signal"] = rotated["signal"].values

    def _measure_qick(self) -> Path:
        logger.info("Starting qick T2 Ramsey measurement")
        sweep = T2RProgram()
        logger.debug("Sweep created, running measurement")
        loc, da = run_and_save_sweep(sweep, "data", self.name)
        logger.info("Measurement complete")
        return loc

    def _measure_opx(self) -> Path:
        logger.info("Starting opx T2 Ramsey measurement")
        loc = measure_t2(n_echos=0)
        logger.info("Measurement complete")
        return loc

    def _load_data_qick(self):
        data = load_as_xr(self.data_loc)
        rotated = rotate_complex_qubit_data(data)[0]
        self.independents["delays"] = rotated["t"].values
        self.dependents["signal"] = rotated["signal"].values

    def _load_data_opx(self):
        data = load_as_xr(self.data_loc)
        if "repetition" in data.dims:
            data = data.mean("repetition")
        data, _ = rotate_complex_qubit_data(data)
        self.independents["delays"] = data["delay"].values
        self.dependents["signal"] = data["signal"].values

    def _single_fit(self, delays, signal):
        """Fit one candidate. Returns (fit_result, fit_curve, residuals, sse)."""
        fit = ExponentiallyDecayingSine(delays, signal)
        guess = fit.guess(delays, signal)
        overrides = {}
        if abs(guess["of"]) < self._OFFSET_SNAP_REL * abs(guess["A"]):
            logger.debug(f"Snapping offset guess {guess['of']:.3g} to 0 (A guess {guess['A']:.3g})")
            overrides["of"] = 0.0
        fit_result = fit.run(params=overrides)
        fit_curve = fit_result.eval()
        residuals = signal - fit_curve
        sse = float(np.sum(np.abs(residuals) ** 2))
        return fit_result, fit_curve, residuals, sse

    def _fit_exponentially_decaying_sine(self, delays, signal, fig_title="") -> tuple:
        # Fit both data and -data; keep whichever agrees best with the data it was fit to.
        signal = np.asarray(signal)
        candidates = {}
        errors = {}
        for inverted, s in ((False, signal), (True, -signal)):
            label = "-data" if inverted else "data"
            try:
                candidates[inverted] = (s, *self._single_fit(delays, s))
            except Exception as e:
                errors[inverted] = e
                logger.warning(f"{label} fit raised an error: {e!r}")
                continue
            logger.info(f"{label} fit: SSE={candidates[inverted][4]:.4g}")

        if not candidates:
            raise RuntimeError(
                f"Fit failed for both data and -data: data: {errors[False]!r}; -data: {errors[True]!r}"
            ) from errors[False]

        inverted = min(candidates, key=lambda k: candidates[k][4])
        fit_signal, fit_result, fit_curve, residuals, _ = candidates[inverted]
        self.fit_inverted = inverted

        amp = fit_result.params["A"].value
        noise = np.std(residuals)
        snr = np.abs(amp / (4 * noise))

        fig, ax = plt.subplots()
        ax.set_title(fig_title + (" (inverted data)" if inverted else ""))
        if self.platform_type == PlatformTypes.OPX:
            ax.set_xlabel("Delay (ns)")
        else:
            ax.set_xlabel("Delay (μs)")
        ax.set_ylabel("Rotated Signal (A.U)" + (" × -1" if inverted else ""))
        ax.plot(delays, fit_signal, label="Data")
        ax.plot(delays, fit_curve, label="Fit")
        ax.legend()

        return fit_result, residuals, snr, fig

    def _analyze_default(self):
        with DatasetAnalysis(self.data_loc, self.name) as ds:
            self.fit_result, self.residuals, self.snr, fig = self._fit_exponentially_decaying_sine(
                self.independents["delays"],
                self.dependents["signal"],
                "T2 Ramsey Measurement"
            )

            # Save all fit results
            ds.add(
                fit_result=self.fit_result,
                params=serialize_fit_params(self.fit_result.params),
                snr=float(self.snr),
                fit_inverted=bool(self.fit_inverted),
            )

            ds.add_figure(self.name, fig=fig)
            image_path = ds._new_file_path(ds.savefolders[1], self.name, suffix="png")
            self.figure_paths.append(image_path)

    def _check_quality(self) -> CheckResult:
        snr_min = self.snr_min_threshold()
        max_error = self.max_fit_param_error()
        param = self.fit_result.params["tau"]
        bad_param = None
        if param.stderr is None:
            bad_param = "tau(no stderr)"
        elif param.value == 0 or abs(param.stderr / param.value) > max_error:
            pct = abs(param.stderr / param.value) * 100 if param.value != 0 else float("inf")
            bad_param = f"tau({pct:.0f}%)"
        passed = self.snr >= snr_min and bad_param is None
        parts = [f"SNR={self.snr:.3f} (threshold={snr_min:.1f})"]
        if bad_param:
            parts.append(f"high-error param: {bad_param}")

        return CheckResult("quality_check", passed, "; ".join(parts))

    def correct(self, result: EvaluateResult) -> EvaluateResult:
        figure = self.figure_paths[0] if self.figure_paths else None
        self.figure_paths.clear()

        snr_min = self.snr_min_threshold()
        self.report_output.append(
            f"## T2 Ramsey (T2R) Measurement\n"
            f"Measured T2 Ramsey time with SNR threshold: {snr_min:.1f}\n"
            f"Data Path: `{self.data_loc}`\n\n"
        )

        self.report_output.append("### Rotated Signal Fit\n")
        if figure:
            self.report_output.append(figure)
        self.report_output.append(
            f"SNR={self.snr:.3f}, fit on {'-data' if self.fit_inverted else 'data'}\n\n"
            f"**Fit Report:**\n```\n{str(self.fit_result.lmfit_result.fit_report())}\n```\n\n"
        )

        result = super().correct(result)  # adds check table + success update line
        if result.status in (OperationStatus.SUCCESS, OperationStatus.FAILURE):
            self._increase_averaging.restore()
        return result

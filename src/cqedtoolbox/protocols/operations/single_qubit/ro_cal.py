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

from labcore.protocols.base import ProtocolOperation, OperationStatus
from qcui_measurement.protocols.parameters import (
    Repetition,
    QubitGain,
    ReadoutGain,
    ReadoutLength
)
from cqedtoolbox.measurement_lib.qick.single_transmon_v2 import SingleShotGroundProgram, SingleShotExcitedProgram


logger = logging.getLogger(__name__)


class ReadoutCalibration(ProtocolOperation):

    def __init__(self, params):
        super().__init__()

        self.params = params  # Store params for reps/steps access

        self._register_inputs(
            repetitions=Repetition(params),
            qubit_gain=QubitGain(params),
            readout_gain=ReadoutGain(params),
            readout_length=ReadoutLength(params)
        )
        self._register_outputs()  # No output parameters for this measurement

        self.condition = "Success if both ground and excited state measurements complete successfully"

        # Data for ground measurement
        self.data_loc_ground: Path | None = None
        self.independents_ground = {}
        self.dependents_ground = {"signal": []}
        self.I_ground = None
        self.Q_ground = None
        self.mean_I_ground = None
        self.mean_Q_ground = None

        # Data for excited measurement
        self.data_loc_excited: Path | None = None
        self.independents_excited = {}
        self.dependents_excited = {"signal": []}
        self.I_excited = None
        self.Q_excited = None
        self.mean_I_excited = None
        self.mean_Q_excited = None

        # Distance between centers
        self.distance = None

        # Store old reps
        self.old_reps = None

    _SIM_N_SHOTS = 1000
    _SIM_SPREAD = 0.1
    _SIM_GROUND_CENTER = (0.2, 0.0)
    _SIM_EXCITED_CENTER = (-0.2, 0.0)

    def _measure_dummy(self) -> Path:
        logger.info("Starting dummy readout calibration measurement")
        shots = np.arange(self._SIM_N_SHOTS)

        ig = (self._SIM_GROUND_CENTER[0] + self._SIM_SPREAD * np.random.randn(self._SIM_N_SHOTS)
              + 1j * (self._SIM_GROUND_CENTER[1] + self._SIM_SPREAD * np.random.randn(self._SIM_N_SHOTS)))
        sweep_ground = sweep_parameter("shots", shots, record_as(lambda s: ig, "g"))
        loc_ground, _ = run_and_save_sweep(sweep_ground, "data", f"{self.name}_ground")

        ie = (self._SIM_EXCITED_CENTER[0] + self._SIM_SPREAD * np.random.randn(self._SIM_N_SHOTS)
              + 1j * (self._SIM_EXCITED_CENTER[1] + self._SIM_SPREAD * np.random.randn(self._SIM_N_SHOTS)))
        sweep_excited = sweep_parameter("shots", shots, record_as(lambda s: ie, "e"))
        loc_excited, _ = run_and_save_sweep(sweep_excited, "data", f"{self.name}_excited")

        self.data_loc_ground = loc_ground
        self.data_loc_excited = loc_excited
        logger.info("Dummy measurement complete")
        return loc_ground

    def _load_data_dummy(self):
        path_ground = self.data_loc_ground / "data.ddh5"
        if not path_ground.exists():
            raise FileNotFoundError(f"File {path_ground} does not exist")
        data_ground = datadict_from_hdf5(path_ground)
        self.dependents_ground["signal"] = data_ground["g"]["values"]
        self.I_ground = data_ground["g"]["values"].real
        self.Q_ground = data_ground["g"]["values"].imag

        path_excited = self.data_loc_excited / "data.ddh5"
        if not path_excited.exists():
            raise FileNotFoundError(f"File {path_excited} does not exist")
        data_excited = datadict_from_hdf5(path_excited)
        self.dependents_excited["signal"] = data_excited["e"]["values"]
        self.I_excited = data_excited["e"]["values"].real
        self.Q_excited = data_excited["e"]["values"].imag

    def _measure_qick(self) -> Path:
        logger.info("Starting qick readout calibration measurement")

        # Store old reps, set to 1000 for single-shot
        self.old_reps = self.repetitions()
        logger.debug(f"Storing old reps: {self.old_reps}")
        logger.debug("Setting reps=1000 for single-shot measurement")
        self.repetitions(1000)

        try:
            # Measure ground state
            sweep_ground = SingleShotGroundProgram()
            logger.debug("Ground state sweep created, running measurement")
            loc_ground, da_ground = run_and_save_sweep(sweep_ground, "data", f"{self.name}_ground")
            logger.info("Ground state measurement complete at %s", loc_ground)

            # Measure excited state
            sweep_excited = SingleShotExcitedProgram()
            logger.debug("Excited state sweep created, running measurement")
            loc_excited, da_excited = run_and_save_sweep(sweep_excited, "data", f"{self.name}_excited")
            logger.info("Excited state measurement complete at %s", loc_excited)

            self.data_loc_ground = loc_ground
            self.data_loc_excited = loc_excited

            # Return the ground location as the primary data_loc for compatibility
            return loc_ground
        finally:
            # Restore reps and steps
            if self.old_reps is not None:
                logger.debug(f"Restoring reps to {self.old_reps}")
                self.repetitions(self.old_reps)

    def _load_data_qick(self):
        # Load ground state data
        path_ground = self.data_loc_ground / "data.ddh5"
        if not path_ground.exists():
            raise FileNotFoundError(f"File {path_ground} does not exist")
        data_ground = datadict_from_hdf5(path_ground)

        self.dependents_ground["signal"] = data_ground["g"]["values"]
        self.I_ground = data_ground["g"]["values"].T.real
        self.Q_ground = data_ground["g"]["values"].T.imag

        # Load excited state data
        path_excited = self.data_loc_excited / "data.ddh5"
        if not path_excited.exists():
            raise FileNotFoundError(f"File {path_excited} does not exist")
        data_excited = datadict_from_hdf5(path_excited)

        self.dependents_excited["signal"] = data_excited["e"]["values"]
        self.I_excited = data_excited["e"]["values"].T.real
        self.Q_excited = data_excited["e"]["values"].T.imag

    def analyze(self):
        # Calculate mean positions
        self.mean_I_ground = np.mean(self.I_ground)
        self.mean_Q_ground = np.mean(self.Q_ground)
        self.mean_I_excited = np.mean(self.I_excited)
        self.mean_Q_excited = np.mean(self.Q_excited)

        # Calculate distance between centers
        self.distance = np.sqrt(
            (self.mean_I_excited - self.mean_I_ground)**2 +
            (self.mean_Q_excited - self.mean_Q_ground)**2
        )

        logger.info(f"Ground state center: I={self.mean_I_ground:.3f}, Q={self.mean_Q_ground:.3f}")
        logger.info(f"Excited state center: I={self.mean_I_excited:.3f}, Q={self.mean_Q_excited:.3f}")
        logger.info(f"Distance between centers: {self.distance:.3f}")

        # Create combined I/Q scatter plot
        with DatasetAnalysis(self.data_loc_ground, f"{self.name}_combined") as ds:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_title("I/Q Scatter Plot - Ground vs Excited States")
            ax.set_xlabel("I (a.u.)")
            ax.set_ylabel("Q (a.u.)")

            # Plot ground state points
            ax.scatter(
                self.I_ground, self.Q_ground,
                color="b", alpha=0.1, label="Ground"
            )

            # Plot excited state points
            ax.scatter(
                self.I_excited, self.Q_excited,
                color="r", alpha=0.1, label="Excited"
            )

            # Plot ground state mean
            ax.scatter(
                [self.mean_I_ground], [self.mean_Q_ground],
                color="k", marker="*", s=100,
                label=f"Ground mean: ({self.mean_I_ground:.3f}, {self.mean_Q_ground:.3f})"
            )

            # Plot excited state mean
            ax.scatter(
                [self.mean_I_excited], [self.mean_Q_excited],
                color="k", marker="o", s=100,
                label=f"Excited mean: ({self.mean_I_excited:.3f}, {self.mean_Q_excited:.3f})"
            )

            ax.legend()

            ds.add_figure(f"{self.name}_iq_scatter", fig=fig)

            image_path = ds._new_file_path(ds.savefolders[1], f"{self.name}_iq_scatter", suffix="png")
            self.figure_paths.append(image_path)

            ds.add(
                mean_I_ground=float(self.mean_I_ground),
                mean_Q_ground=float(self.mean_Q_ground),
                mean_I_excited=float(self.mean_I_excited),
                mean_Q_excited=float(self.mean_Q_excited),
                distance=float(self.distance)
            )

    def evaluate(self) -> OperationStatus:
        """
        Evaluate if the measurement was successful.
        Success criteria: Both measurements completed successfully.
        """
        header = (f"## Readout Calibration\n"
                  f"Measured single-shot readout for ground and excited states\n"
                  f"Ground state data: `{self.data_loc_ground}`\n"
                  f"Excited state data: `{self.data_loc_excited}`\n\n")

        plot_iq = self.figure_paths[0].resolve()

        # Calculate readout fidelity information
        msg_main = (f"### Measurement Complete\n"
                   f"**Ground State Center:**\n"
                   f"- I: {self.mean_I_ground:.6f}\n"
                   f"- Q: {self.mean_Q_ground:.6f}\n\n"
                   f"**Excited State Center:**\n"
                   f"- I: {self.mean_I_excited:.6f}\n"
                   f"- Q: {self.mean_Q_excited:.6f}\n\n"
                   f"**Distance Between Centers:** {self.distance:.6f}\n\n"
                   f"The I/Q scatter plot below shows the distribution of single-shot measurements for both states.\n"
                   f"A larger distance between centers indicates better readout distinguishability.\n\n")

        self.report_output = [header, msg_main, plot_iq]

        logger.info("Readout calibration complete")
        return OperationStatus.SUCCESS
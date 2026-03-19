from pathlib import Path

from labcore.protocols.base import ProtocolBase, BranchBase
from cqedtoolbox.protocols.operations import ResonatorSpectroscopy, ResonatorSpectroscopyVsGain, PiSpectroscopy, PowerRabi, ResonatorSpectroscopyAfterPi, ReadoutCalibration, SaturationSpectroscopy, T1Operation, T2EOperation, T2ROperation


class QubitTuneup(ProtocolBase):

    def __init__(self, params, report_path: Path = Path(".")):
        super().__init__(report_path)

        self.root_branch = BranchBase("QubitTuneup")
        self.root_branch.extend([
            ResonatorSpectroscopy(params),
            ResonatorSpectroscopyVsGain(params),
            SaturationSpectroscopy(params),
            PowerRabi(params),
            PiSpectroscopy(params),
            ResonatorSpectroscopyAfterPi(params),
            T1Operation(params),
            T2ROperation(params),
            T2EOperation(params),
            ReadoutCalibration(params),
        ])
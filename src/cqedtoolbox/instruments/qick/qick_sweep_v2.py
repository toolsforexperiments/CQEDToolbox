""" Tools to enable the usage of QICK in the Sweep framework.

Required packages/hardware:
- FPGA configured with QICK
- QICK package

Example usage of this module:
>>> @QickBoardSweep(
>>>     independent("freqs"),
>>>     ComplexQICKData("signal",
>>>                     depends_on=["freqs"],
>>>                     i_data_stream='I', q_data_stream='Q'),
>>> )
>>> class SingleToneSpecstroscopyProgram(RAveragerProgram):

To run any measurement, you would need the followign parameters in your config
n_echoes
steps
reps
final_delay
""" 

import numpy as np
from collections.abc import Iterable, Generator
from dataclasses import dataclass

from labcore.measurement import independent, dependent, DataSpec
from labcore.measurement.sweep import AsyncRecord
from labcore.measurement.record import make_data_spec

# config has to be set by the users of the program.
# Example:
# conf = QickConfig(paras=params)
# qick_sweep.config = conf
config = None

@dataclass
class ComplexQICKData(DataSpec):
    i_data_stream: str = 'I'
    q_data_stream: str = 'Q'

@dataclass
class PulseVariable(DataSpec):
    pulse_parameter: str = None
    sweep_parameter: str = None

@dataclass
class TimeVariable(DataSpec):
    time_parameter: str = None

@dataclass
class DecimatedVariable(DataSpec):
    ro_index: int = None

@dataclass
class SuperDecimatedVariable(DataSpec):
    tlist: str = None


class QickBoardSweep(AsyncRecord):
    """
    Decorator class to communicate with QICK in the Sweeping framework.
    """
    def __init__(self, *specs, **kwargs):
        """
        Initialize the decorator class by saving incoming DataSpec variables.
        """
        self.communicator = {}
        self.specs = []
        self.decimated_flag = False
        self.superdecimated_flag = False

        for s in specs:
            # Checks the presence of decimated or superdecimated readout.
            # Note that only one decimated or superdecimated readout is supported.
            if isinstance(s, DecimatedVariable):
                if self.decimated_flag is True or self.superdecimated_flag is True:
                    raise ValueError("Only decimated or super-decimated RO allowed.")
                else:
                    self.decimated_flag = True
            elif isinstance(s, SuperDecimatedVariable):
                if self.decimated_flag is True or self.superdecimated_flag is True:
                    raise ValueError("Only decimated or super-decimated RO allowed.")
                else:
                    self.superdecimated_flag = True

            spec = make_data_spec(s)
            self.specs.append(spec)

    def setup(self, func, *args, **kwargs):
        """
        Setup a QICK program.
        """
        # Checks that the config is not None
        if config is None:
            raise Exception("QickSweep: config is not set")

        self.config = config
        conf = config.config()
        qick_program = func(soccfg=conf[0], reps = conf[1]['reps'], final_delay=conf[1]['final_delay'], cfg=conf[1])
        self.communicator["qick_program"] = qick_program

    def collect(self, *args, **kwargs):
        """
        Get the measurement data.
        Note that one can overload the given acquire function if one needs
        to perform other specific tasks. e.g. Needs to plot each non-averaged 
        points in the I-Q plane for a readout fidelity experiment.
        Assumptions
        * Given DataSpecs are either independent, dependent, or ComplexQICKData.
        """
        # TODO: How can I extend this to multiple measurement rounds? (e.g. active reset)
        # TODO: Understand the reason why we have data[0]?

        # Run the program
        cfg = self.config.config()[1]
        return_data = {}
        if self.decimated_flag is True:
            if cfg['reps'] > 1:
                raise ValueError("Rep should be fixed to 1 for Decimated RO.")
            elif len(self.specs) > 2:
                raise ValueError("Decimated RO does not support parameter sweep.")
            else:
                data = {}
                data[0] = self.communicator["qick_program"].acquire_decimated(self.config.soc, progress=False, rounds=self.config.config()[1]['rounds'])[0]
        elif self.superdecimated_flag is True:
            data = {}
            data[0] = self.communicator["qick_program"].acquire(self.config.soc, progress=False, rounds=self.config.config()[1]['rounds'])[0]
        else:
            data = self.communicator["qick_program"].acquire(self.config.soc, progress=False)[0]
        

        measIdx = 0  # To specify the index of the measured data to return
        sweepIdx = 0   # To specify the index of the sweep variable to return
        for ds in self.specs:
            if isinstance(ds, ComplexQICKData):
                return_data[ds.name]= data[measIdx].dot([1,1j])
                measIdx += 1
            elif isinstance(ds, PulseVariable):
                return_data[ds.name]= self.communicator["qick_program"].get_pulse_param(ds.pulse_parameter, ds.sweep_parameter, as_array=True)
                sweepIdx += 1
            elif isinstance(ds, TimeVariable):
                return_data[ds.name]= (self.communicator["qick_program"].get_time_param(ds.time_parameter, 't', as_array=True))*(cfg['n_echoes']+1)
                sweepIdx += 1
            elif isinstance(ds, DecimatedVariable):
                return_data[ds.name] = self.communicator["qick_program"].get_time_axis(ro_index=ds.ro_index)
                sweepIdx += 1
            elif isinstance(ds, SuperDecimatedVariable):
                # It is under user's responsibility to generate an appropriate tlist under their qick_program.
                return_data[ds.name] = self.communicator["qick_program"].tlist
                sweepIdx += 1
            else:
                return_data[ds.name] = np.arange(cfg['steps'])
                sweepIdx += 1

        # Reformat the independent variables
        # Independent (sweep) variables have to be reformatted such that xArray and plottr can
        # correctly recognize the axis being swept. Without reformatting the swept variable,
        # the program won't be able to correctly set up the axis being swept.
        shapeIdx = 0
        for ds in self.specs:
            if isinstance(ds, ComplexQICKData):
                return_data[ds.name] = np.transpose(return_data[ds.name])
            else:
                dimList = [1] * sweepIdx
                dimList[shapeIdx] = len(return_data[ds.name])
                dimTuple = tuple(dimList)
                diffList = [1] * sweepIdx
                diffIdx = 0
                for dp in self.specs:
                    if isinstance(dp, ComplexQICKData):
                        pass
                    else:
                        if diffIdx != shapeIdx:
                            diffList[diffIdx] = len(return_data[dp.name])
                        diffIdx += 1
                diffTuple = tuple(diffList)
                return_data[ds.name] = np.reshape(return_data[ds.name], dimTuple)
                return_data[ds.name] = np.tile(return_data[ds.name], diffTuple)

                shapeIdx += 1

        yield return_data


        

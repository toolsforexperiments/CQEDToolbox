"""Keysight-compatible Copper Mountain M5180 driver built on qcodes-contrib.

The original QCoDeS-contrib driver remains responsible for the standard M5180
parameters.  The Trace and POL-reading helper classes below are copied from
the established standalone driver so their behaviour can be compared directly.

It requires the original driver to be importable, for example from the
``qcodes_contrib_drivers`` checkout/package that contains
``drivers/CopperMountain/M5180.py``.
"""

from __future__ import annotations

import cmath
import math
from typing import Any, List, Tuple

import numpy as np
from qcodes.instrument import InstrumentChannel
from qcodes.parameters import DelegateParameter, ParamRawDataType
from qcodes.validators import Enum, Numbers

from qcodes_contrib_drivers.drivers.CopperMountain.M5180 import (
    FrequencySweepMagPhase as _BaseFrequencySweepMagPhase,
    M5180 as _ContribM5180,
    PointIQ as _BasePointIQ,
    PointMagPhase as _BasePointMagPhase,
)


class Trace(InstrumentChannel):
    """
    Per-trace/measurement parameters, mirroring the shape of
    Keysight_P9374A_SingleChannel's Trace channel (npts, frequency, data,
    s_parameter) for cross-driver compatibility.
    """

    def __init__(self, parent: "M5180", number: int, name: str, **kwargs):
        self._number = number
        super().__init__(parent, name=name, **kwargs)

        self.add_parameter('npts',
                           unit='',
                           get_cmd=lambda: self.parent.num_points(),
                           set_cmd=self._reject_read_only_set,
                           snapshot_exclude=True,
                           docstring='number of points in the trace')

        self.add_parameter('frequency',
                           unit='Hz',
                           get_cmd=self._get_frequency,
                           set_cmd=self._reject_read_only_set,
                           snapshot_exclude=True)

        self.add_parameter('data',
                           unit='',
                           get_cmd=self._get_data,
                           set_cmd=self._reject_read_only_set,
                           snapshot_exclude=True)

        self.add_parameter('s_parameter',
                           get_cmd=self._get_s_parameter,
                           set_cmd=self._set_s_parameter,
                           vals=Enum('S11', 'S12', 'S21', 'S22'),
                           get_parser=str,
                           snapshot_exclude=True)

    def _reject_read_only_set(self, value: Any) -> None:
        """Keep trace acquisition parameters read-only.

        Older InstrumentServer GUIs call ``setValue`` on every broadcast
        parameter update, including read-only array parameters.  Giving these
        parameters a setter makes that GUI use a value-capable editor.  A real
        set remains forbidden and returns this clear error to the caller.
        """
        raise RuntimeError(f"{self.full_name} is read-only")

    def _get_frequency(self) -> np.ndarray:
        freq_raw = self.ask("SENS1:FREQ:DATA?")
        return np.fromstring(freq_raw, dtype=float, sep=',')

    # def _get_data(self) -> np.ndarray:
    #     raw = self.ask(f"CALC1:TRAC{self._number}:DATA:FDAT?")
    #     return np.fromstring(raw, dtype=float, sep=',')
    def _get_data(self) -> np.ndarray:
        prev_format = self.ask(f"CALC1:TRAC{self._number}:FORM?")
        self.write(f"CALC1:TRAC{self._number}:FORM POL")
        raw = self.ask(f"CALC1:TRAC{self._number}:DATA:FDAT?")
        self.write(f"CALC1:TRAC{self._number}:FORM {prev_format}")
        data = np.fromstring(raw, dtype=float, sep=',')

        # Validity check
        expected_size = 2 * self.parent.num_points()
        if data.size != expected_size:
            raise RuntimeError(
                f"Expected {expected_size} real/imaginary values from trace "
                f"{self._number}, but received {data.size}."
            )

        return data[0::2] + 1j*data[1::2]

    def _get_s_parameter(self) -> str:
        return self.ask(f"CALC1:PAR{self._number}:DEF?").strip().strip('"')

    def _set_s_parameter(self, s_param: str) -> None:
        self.write(f"CALC1:PAR{self._number}:DEF {s_param}")


class FrequencySweepMagPhase(_BaseFrequencySweepMagPhase):
    def get_raw(self) -> Tuple[ParamRawDataType, ParamRawDataType]:
        """Gets data from instrument

        Returns:
            Tuple[ParamRawDataType, ...]: magnitude, phase
        """
        assert isinstance(self.instrument, M5180)
        self.instrument.write('CALC1:PAR:COUN 1') # 1 trace
        self.instrument.write('CALC1:PAR1:DEF {}'.format(self.name))
        self.instrument.trigger_source('bus') # set the trigger to bus
        self.instrument.write('TRIG:SEQ:SING') # Trigger a single sweep
        self.instrument.ask('*OPC?') # Wait for measurement to complete

        # get data from instrument
        self.instrument.write('CALC1:TRAC1:FORM POL')  # ensure correct format
        sxx_raw = self.instrument.ask("CALC1:TRAC1:DATA:FDAT?")
        self.instrument.write('CALC1:TRAC1:FORM MLOG')

        # Get data as numpy array
        sxx = np.fromstring(sxx_raw, dtype=float, sep=',')
        sxx = sxx[0::2] + 1j*sxx[1::2]

        return self.instrument._db(sxx), np.unwrap(np.angle(sxx))


class PointMagPhase(_BasePointMagPhase):
    def get_raw(self) -> Tuple[ParamRawDataType, ParamRawDataType]:
        """Gets data from instrument

        Returns:
            Tuple[ParamRawDataType, ...]: magnitude, phase
        """

        assert isinstance(self.instrument, M5180)
        # check that npts, start and stop fullfill requirements if point_check_sweep_first is True.
        if self.instrument.point_check_sweep_first():
            if self.instrument.num_points() != 2:
                raise ValueError('Npts is not 2 but {}. Please set it to 2'.format(self.instrument.num_points()))
            if self.instrument.fstop() - self.instrument.fstart() != 1:
                raise ValueError('Stop-start is not 1 Hz but {} Hz. Please adjust'
                                'start or stop.'.format(self.instrument.fstop()-self.instrument.fstart()))

        self.instrument.write('CALC1:PAR:COUN 1') # 1 trace
        self.instrument.write('CALC1:PAR1:DEF {}'.format(self.name[-3:]))
        self.instrument.trigger_source('bus') # set the trigger to bus
        self.instrument.write('TRIG:SEQ:SING') # Trigger a single sweep
        self.instrument.ask('*OPC?') # Wait for measurement to complete

        # get data from instrument
        self.instrument.write('CALC1:TRAC1:FORM POL')  # ensure correct format
        sxx_raw = self.instrument.ask("CALC1:TRAC1:DATA:FDAT?")

        # Get data as numpy array
        sxx = np.fromstring(sxx_raw, dtype=float, sep=',')
        sxx = sxx[0::2] + 1j*sxx[1::2]

        # Return the average of the trace, which will have "start" as
        # its setpoint
        sxx_mean = np.mean(sxx)
        return 20*math.log10(abs(sxx_mean)), (cmath.phase(sxx_mean))


class PointIQ(_BasePointIQ):
    def get_raw(self) -> Tuple[ParamRawDataType, ParamRawDataType]:
        """Gets data from instrument

        Returns:
            Tuple[ParamRawDataType, ...]: I, Q
        """

        assert isinstance(self.instrument, M5180)
        # check that npts, start and stop fullfill requirements if point_check_sweep_first is True.
        if self.instrument.point_check_sweep_first():
            if self.instrument.num_points() != 2:
                raise ValueError('Npts is not 2 but {}. Please set it to 2'.format(self.instrument.num_points()))
            if self.instrument.fstop() - self.instrument.fstart() != 1:
                raise ValueError('Stop-start is not 1 Hz but {} Hz. Please adjust'
                                'start or stop.'.format(self.instrument.fstop()-self.instrument.fstart()))

        self.instrument.write('CALC1:PAR:COUN 1') # 1 trace
        # These parameters are named ``point_sXX_iq``, so the upstream driver's fixed
        # ``self.name[-3:]`` slice sends the ``_iq`` suffix instead of the S parameter.
        s_parameter = self.name.removeprefix('point_').removesuffix('_iq')
        self.instrument.write('CALC1:PAR1:DEF {}'.format(s_parameter))
        self.instrument.trigger_source('bus') # set the trigger to bus
        self.instrument.write('TRIG:SEQ:SING') # Trigger a single sweep
        self.instrument.ask('*OPC?') # Wait for measurement to complete

        # get data from instrument
        self.instrument.write('CALC1:TRAC1:FORM POL')  # ensure correct format
        sxx_raw = self.instrument.ask("CALC1:TRAC1:DATA:FDAT?")

        # Get data as numpy array
        sxx = np.fromstring(sxx_raw, dtype=float, sep=',')

        # Return the average of the trace, which will have "start" as
        # its setpoint
        return np.mean(sxx[0::2]), np.mean(sxx[1::2])


class M5180(_ContribM5180):
    """M5180 with the established Keysight-compatible and Trace API.

    The parent retains its native parameter names (``start``, ``npts``, etc.)
    because its own sweep helpers use them.  Registered ``DelegateParameter``
    aliases provide the names expected by :mod:`vna` and InstrumentServer.
    """

    _ALIASES = {
        "rfout": "output",
        "ifbw": "if_bandwidth",
        "averaging": "averages_enabled",
        "avg_num": "averages",
        "fstart": "start",
        "fstop": "stop",
        "fcenter": "center",
        "fspan": "span",
        "num_points": "npts",
    }

    # ``_ContribM5180.__init__`` invokes ``self.add_parameter``.  Replace its
    # SMITH-reading helpers with subclasses that override only ``get_raw``.
    _POLAR_PARAMETER_CLASSES = {
        _BaseFrequencySweepMagPhase: FrequencySweepMagPhase,
        _BasePointMagPhase: PointMagPhase,
        _BasePointIQ: PointIQ,
    }

    def add_parameter(self, name: str, *args: Any, **kwargs: Any) -> Any:
        parameter_class = kwargs.get("parameter_class")
        replacement_class = self._POLAR_PARAMETER_CLASSES.get(parameter_class)
        if replacement_class is not None:
            kwargs["parameter_class"] = replacement_class
        return super().add_parameter(name, *args, **kwargs)

    def __init__(
        self,
        name: str,
        address: str,
        terminator: str = "\n",
        timeout: int = 100000,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            address=address,
            terminator=terminator,
            timeout=timeout,
            **kwargs,
        )

        # The POL substitution in add_parameter depends on the contrib driver passing
        # parameter_class as a keyword and on the three imported base classes keeping
        # their names.  If any of that changes, the substitution stops applying and
        # every read silently returns SMITH impedance data that _db() converts into
        # plausible-looking dB -- so verify it took effect rather than trusting it.
        for parameter_name, expected_class in (
            ("s21", FrequencySweepMagPhase),
            ("point_s21", PointMagPhase),
            ("point_s21_iq", PointIQ),
        ):
            assert isinstance(self.parameters.get(parameter_name), expected_class), (
                f"{name}.{parameter_name} is not a {expected_class.__name__}. You are "
                "most likely using a newer qcodes_contrib_drivers than this driver was "
                "written against: its M5180 no longer matches _POLAR_PARAMETER_CLASSES, "
                "so readings would be SMITH impedance instead of a+bi."
            )

        # Preserve the current driver's zero-span API.  The native contrib
        # driver uses a minimum of one hertz, which rejects vna.py's CW setup.
        self.span.vals = Numbers(min_value=0, max_value=18e9 - 1)

        # Keep vna.py's Keysight spelling plus the currently accepted short
        # Copper Mountain spellings.  IMM is translated to Copper Mountain's
        # INT in _set_trigger.
        self.trigger_source.vals = Enum(
            "bus",
            "external",
            "internal",
            "manual",
            "BUS",
            "EXTERNAL",
            "INTERNAL",
            "MANUAL",
            "EXT",
            "INT",
            "MAN",
            "IMM",
            "ext",
            "int",
            "man",
            "imm",
        )

        for alias, source_name in self._ALIASES.items():
            self.add_parameter(
                alias,
                parameter_class=DelegateParameter,
                source=getattr(self, source_name),
                bind_to_instrument=True,
            )

        self.add_parameter(
            "sweep_mode",
            get_cmd=self._get_sweep_mode,
            set_cmd=self._set_sweep_mode,
            vals=Enum("HOLD", "CONT", "SING", "GRO"),
        )
        self.add_parameter(
            "trigger_mode",
            get_cmd=self._get_trigger_mode,
            set_cmd=self._set_trigger_mode,
            vals=Enum("CHAN", "SWE", "POIN", "TRAC"),
        )

        self.register_active_traces()

    def _set_trigger(self, trigger: str) -> None:
        """Sets trigger source.

        Args:
            trigger (str): Trigger source
        """
        # Keysight calls its internal trigger source IMM.  Copper Mountain uses INT, so we translate here.
        source = 'INT' if trigger.upper() == 'IMM' else trigger.upper()
        self.write('TRIG:SOUR ' + source)

    def _set_trigger_mode(self, val: str) -> None:
        """Approximates Keysight's 4-state trigger_mode using this instrument's
        boolean TRIG:POIN (point-trigger on/off). 'POIN' -> ON (each trigger
        advances one point); any other value -> OFF (each trigger runs a full
        sweep). This is an approximation -- Copper Mountain only distinguishes
        point-vs-sweep trigger response, not four separate scopes."""
        self.write('TRIG:POIN {}'.format(1 if val.upper() == 'POIN' else 0))

    def _get_trigger_mode(self) -> str:
        return 'POIN' if self.ask('TRIG:POIN?').strip() in ('1', 'ON') else 'SWE'

    def _set_sweep_mode(self, val: str) -> None:
        """Approximates Keysight's 4-state sweep_mode using this instrument's
        boolean INIT:CONT. CONT/GRO -> continuous ON; HOLD/SING -> OFF.
        This is an approximation, not a faithful port -- Copper Mountain has
        no direct equivalent of Keysight's SING/GRO states."""
        self.write('INIT1:CONT {}'.format(1 if val.upper() in ('CONT', 'GRO') else 0))

    def _get_sweep_mode(self) -> str:
        return 'CONT' if self.ask('INIT1:CONT?').strip() in ('1', 'ON') else 'HOLD'

    def register_active_traces(self) -> List[str]:
        """Register submodules for every trace currently active on the VNA.

        The instrument server reads every registered Trace when its GUI tab
        opens, so inactive trace numbers must not be registered.  Call this
        after increasing ``nb_traces`` on the physical VNA:

        >>> vna.nb_traces(4)
        >>> vna.register_active_traces()

        Existing submodules are retained; only newly active traces are added.

        Returns:
            Names of the Trace submodules registered by this call.
        """
        added: List[str] = []
        for number in range(1, self.nb_traces() + 1):
            name = f"trace_{number}"
            if name not in self.submodules:
                trace = Trace(self, number=number, name=name)
                self.add_submodule(name, trace)
                added.append(name)
        return added

    def clear_averages(self) -> None:
        """
        Resets average count to 0
        """
        self.write("SENS1:AVER:CLE")

    def reset_averages(self) -> None:
        """Correct the inherited reset command and retain its public method."""
        self.clear_averages()

    def get_s(self) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Return all S parameters as magnitude in dB and phase in rad.

        Returns:
            Tuple[np.ndarray]: frequency [GHz],
            s11 magnitude [dB], s11 phase [rad],
            s12 magnitude [dB], s12 phase [rad],
            s21 magnitude [dB], s21 phase [rad],
            s22 magnitude [dB], s22 phase [rad]
        """

        self.write('CALC1:PAR:COUN 4') # 4 trace
        self.write('CALC1:PAR1:DEF S11') # Choose S11 for trace 1
        self.write('CALC1:PAR2:DEF S12') # Choose S12 for trace 2
        self.write('CALC1:PAR3:DEF S21') # Choose S21 for trace 3
        self.write('CALC1:PAR4:DEF S22') # Choose S22 for trace 4
        self.write('CALC1:TRAC1:FORM POL')  # Trace format
        self.write('CALC1:TRAC2:FORM POL')  # Trace format
        self.write('CALC1:TRAC3:FORM POL')  # Trace format
        self.write('CALC1:TRAC4:FORM POL')  # Trace format
        self.write('TRIG:SEQ:SING') # Trigger a single sweep
        self.ask('*OPC?') # Wait for measurement to complete

        # Get data as string
        freq_raw = self.ask("SENS1:FREQ:DATA?")
        s11_raw = self.ask("CALC1:TRAC1:DATA:FDAT?")
        s12_raw = self.ask("CALC1:TRAC2:DATA:FDAT?")
        s21_raw = self.ask("CALC1:TRAC3:DATA:FDAT?")
        s22_raw = self.ask("CALC1:TRAC4:DATA:FDAT?")

        # Get data as numpy array
        freq = np.fromstring(freq_raw, dtype=float, sep=',')
        s11 = np.fromstring(s11_raw, dtype=float, sep=',')
        s11 = s11[0::2] + 1j*s11[1::2]
        s12 = np.fromstring(s12_raw, dtype=float, sep=',')
        s12 = s12[0::2] + 1j*s12[1::2]
        s21 = np.fromstring(s21_raw, dtype=float, sep=',')
        s21 = s21[0::2] + 1j*s21[1::2]
        s22 = np.fromstring(s22_raw, dtype=float, sep=',')
        s22 = s22[0::2] + 1j*s22[1::2]

        return (np.array(freq), self._db(s11), np.array(np.angle(s11)),
                                self._db(s12), np.array(np.angle(s12)),
                                self._db(s21), np.array(np.angle(s21)),
                                self._db(s22), np.array(np.angle(s22)))
 
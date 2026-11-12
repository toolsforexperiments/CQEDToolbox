from qcodes.instrument_drivers.Keysight import KeysightN5222B as qcodes_KeysightN5222B


class KeysightN5222B(qcodes_KeysightN5222B):

    def set_port_freq(self, range_no, start_freq, stop_freq):
        """Set the start and stop frequencies of a frequency-offset range.

        Note that ``range_no`` does NOT correspond to the physical ports, but
        rather to the "ranges" in the PNA (see :meth:`get_range_dict`).
        """
        self.write(f"SENS:FOM:RANG{range_no}:FREQ:STAR {start_freq}")
        self.write(f"SENS:FOM:RANG{range_no}:FREQ:STOP {stop_freq}")

    def get_port_freq(self, range_no):
        """Return the ``(start_freq, stop_freq)`` of a frequency-offset range.

        Note that ``range_no`` does NOT correspond to the physical ports, but
        rather to the "ranges" in the PNA (see :meth:`get_range_dict`).
        """
        start_freq = self.ask_scpi(f"SENS:FOM:RANG{range_no}:FREQ:STAR?")
        stop_freq = self.ask_scpi(f"SENS:FOM:RANG{range_no}:FREQ:STOP?")

        return start_freq, stop_freq

    def set_port(self, port, state: str):
        """Set the source power mode of ``port`` (e.g. ``"ON"``/``"OFF"``)."""
        self.write(f":SOUR1:POW{port}:MODE {state}")

    def set_port_power(self, port, power):
        """Set the source power on ``port`` and turn it on.

        Disables port power coupling first so that ``port`` can be set
        independently of the other ports.
        """
        self.write("SOUR:POW:COUP OFF")
        self.write(f":SOUR:POW{port} %f" % power)
        self.set_port(port, "ON")

    def level_ports_to_open_loop(self, port):
        """Switch the ALC leveling of ``port`` to open-loop.

        This is required for pulse-modulated measurements. Otherwise the ALC
        will try to level the source with the detected power level during the
        pulse ON and OFF states, causing a source-unleveled error.
        """
        self.write(f"SOUR1:POW{port}:ALC:MODE OPEN")

    def set_FOM(self, state: int):
        """Enable (1) or disable (0) the Frequency Offset Mode."""
        self.write(f"SENS:FOM:STAT {state}")

    def set_freq_coupling(self, range_no, state: int):
        """Set coupling of the range ``range_no`` with the primary range."""
        self.write(f"SENS:FOM:RANG{range_no}:COUP {state}")

    def set_pulse(self, pulse, state: int):
        """Enable (1) or disable (0) the given ``pulse`` generator."""
        self.write(f"SENS:PULS{pulse} {state}")

    def set_inverting(self, pulse, state: int):
        """Enable (1) or disable (0) inverting of the given ``pulse``."""
        self.write(f"SENS:PULS{pulse}:INV {state}")

    def set_pulse_width(self, pulse, width):
        """Set the width (in seconds) of the given ``pulse``."""
        self.write(f"SENS:PULS{pulse}:WIDT %.12f" % width)

    def set_pulse_delay(self, pulse, delay):
        """Set the delay (in seconds) of the given ``pulse``."""
        self.write(f"SENS:PULS{pulse}:DEL %.12f" % delay)

    def set_expt_period(self, time):
        """Set the pulse experiment period (in seconds)."""
        self.write("SENS:PULS:PER %.12f" % time)

    def set_sweep_points(self, Nstep):
        """Set the number of points in the sweep."""
        self.write(f"SENS:SWE:POIN {Nstep}")

    def write_scpi(self, command):
        """Write a raw SCPI ``command`` to the PNA.

        Keeps scope for debugging the PNA through SCPI commands from a
        measurement file. Can be removed once the code is standardized.
        """
        return self.write(command)

    def ask_scpi(self, command):
        """Query a raw SCPI ``command`` from the PNA and return the response.

        Keeps scope for debugging the PNA through SCPI commands from a
        measurement file. Can be removed once the code is standardized.
        """
        return self.ask(command)

    def get_range_dict(self):
        """Return a ``{range_name: range_number}`` mapping of all FOM ranges.

        Used to look up a range by number (as expected by the ``range_no``
        arguments of the other methods) from its name.
        """
        names = self.ask_scpi("SENS:FOM:CAT?").replace('"', '').split(',')

        out = {}
        for name in names:
            out[name] = int(self.ask_scpi(f'SENS:FOM:RNUM? "{name}"'))

        return out

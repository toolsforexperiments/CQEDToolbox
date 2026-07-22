from qcodes.instrument_drivers.Keysight import KeysightN5222B as qcodes_KeysightN5222B
from typing import Any


class KeysightN5222B(qcodes_KeysightN5222B):

    def set_port_freq(self, range_no, start_freq, stop_freq):
        # Note that range_no does NOT correspond to the physical ports, but rather to the "ranges" in the PNA
        self.write(f"SENS:FOM:RANG{range_no}:FREQ:STAR {start_freq}")
        self.write(f"SENS:FOM:RANG{range_no}:FREQ:STOP {stop_freq}")

    def get_port_freq(self, range_no):
        # Note that range_no does NOT correspond to the physical ports, but rather to the "ranges" in the PNA
        start_freq = self.ask_scpi(f"SENS:FOM:RANG{range_no}:FREQ:STAR?")
        stop_freq = self.ask_scpi(f"SENS:FOM:RANG{range_no}:FREQ:STOP?")

        return start_freq, stop_freq

    def set_port(self, port, state: str):
        self.write(f":SOUR1:POW{port}:MODE {state}")

    def set_port_power(self, port, power):
        self.write("SOUR:POW:COUP OFF")
        self.write(f":SOUR:POW{port} %f" % power)
        self.set_port(port, "ON")

    def level_ports_to_open_loop(self, port):
        # Turning the leveling of the ports to open-loop
        # This is required for pulse-modulated measurements
        # Otherwise the ALC will try to level the source with the detected power level
        # during pulse ON and OFF states, causing a source unleveled error
        self.write(f"SOUR1:POW{port}:ALC:MODE OPEN")

    def set_FOM(self, state: int):
        # Sets the Frequency Offset Mode
        self.write(f"SENS:FOM:STAT {state}")

    def set_coupling(self, range_no, state: str):
        # Sets coupling of other ranges with the primary range
        self.write(f"SENS:FOM:RANG{range_no}:COUP {state}")

    def set_freq_coupling(self, range_no, state: int):
        self.write(f"SENS:FOM:RANG{range_no}:COUP {state}")

    def set_pulse(self, pulse, state: int):
        self.write(f"SENS:PULS{pulse} {state}")

    def set_inverting(self, pulse, state: int):
        # Sets pulse inverting
        self.write(f"SENS:PULS{pulse}:INV {state}")

    def set_pulse_width(self, pulse, width):
        self.write(f"SENS:PULS{pulse}:WIDT %.12f" % width)

    def set_pulse_delay(self, pulse, delay):
        self.write(f"SENS:PULS{pulse}:DEL %.12f" % delay)

    def set_expt_period(self, time):
        self.write("SENS:PULS:PER %.12f" % time)

    def set_sweep_points(self, Nstep):
        self.write(f"SENS:SWE:POIN {Nstep}")

    def write_scpi(self, command):
        # Keeps scope for debugging PNA through SCPI commands from a measurement file
        # Can be removed once the code is standardized
        return self.write(command)

    def ask_scpi(self, command):
        # Keeps scope for debugging PNA through SCPI commands from a measurement file
        # Can be removed once the code is standardized
        return self.ask(command)

    def get_range_dict(self):
        # Calls the range by number
        names = self.ask_scpi("SENS:FOM:CAT?").replace('"', '').split(',')

        out = {}
        for name in names:
            out[name] = int(self.ask_scpi(f'SENS:FOM:RNUM? "{name}"'))

        return out

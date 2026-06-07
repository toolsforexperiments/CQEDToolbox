
from cqedtoolbox.instruments.qcodes_drivers.Keysight import N52xx

from typing import Any




class qcodes_KeysightN5222B(N52xx.PNABase):
    def __init__(self, name: str, address: str, **kwargs: Any):
        """Driver for Keysight PNA N5222B."""
        super().__init__(
            name,
            address,
            min_freq=10e6,
            max_freq=26.5e9,
            min_power=-30,
            max_power=13,
            nports=4,
            **kwargs
        )

        attenuators_options = {"217", "219", "220", "417", "419", "420"}
        options = set(self.get_options())
        if attenuators_options.intersection(options):
            self._set_power_limits(min_power=-95, max_power=13)


class KeysightN5222B(qcodes_KeysightN5222B):
    # def __init__(self, name: str, address: str, **kwargs: Any):
    #     """Pfafflab Driver for Keysight PNA N5222B."""
    #     super().__init__(
    #         name,
    #         address,
    #         **kwargs)
        

    def set_port_freq(self,range_no,start_freq,stop_freq):
        self.write(f'SENS:FOM:RANG{range_no}:FREQ:STAR {start_freq}' ) 
        self.write(f'SENS:FOM:RANG{range_no}:FREQ:STOP {stop_freq}')  

    def set_port(self,port,state:str):
        self.write(f":SOUR1:POW{port}:MODE {state}")

    def set_port_power(self,port,power):
        self.write("SOUR:POW:COUP OFF")
        self.write(f":SOUR:POW{port} %f" % power) 
        self.set_port(port,'ON')
         
    def level_ports_to_open_loop(self,port): 


        # Turning the leveling of the ports to 'open-loop'. This is required when trying to do pulse modulated measurements. 
        # Otherwise the ALC (Automatic Level Control) will try to level the source with the detected power level with pulse ON and OFF, 
        # causing a source unleveled error.       
                        
        self.write(f"SOUR1:POW{port}:ALC:MODE OPEN")

    def set_FOM(self,state:int):

        ## Sets the Frequency Offset Mode

        self.write(f'SENS:FOM:STAT {state}')

    def set_coupling(self,range_no,state: str):

        ## This sets coupling of other ranges with the "Primary" range"
        self.write(f"SENS:FOM:RANG{range_no}:COUP {state}")

    def set_freq_coupling(self,range_no,state: int):
        self.write(f"SENS:FOM:RANG{range_no}:COUP {state}")

    def set_pulse(self,pulse,state:int):
        self.write(f"SENS:PULS{pulse} {state}")

    def set_inverting(self,pulse,state:int):

        ## sets pulse inverting
        self.write(f"SENS:PULS{pulse}:INV {state}")

    def set_pulse_width(self,pulse,width):
        self.write(f"SENS:PULS{pulse}"+":WIDT %.12f" % width)

    def set_pulse_delay(self,pulse,delay):
        self.write(f"SENS:PULS{pulse}"+":DEL %.12f" % delay)

    def set_expt_period(self,time):
        self.write("SENS:PULS:PER %.12f" % (time)) 

    def set_sweep_points(self,Nstep):
        self.write(f"SENS:SWE:POIN {Nstep}")

    def write_scpi(self, command):

        ## This function keeps the scope to debug PNA through SCPI commands from a measurement file
        ## Once the code is standard, this might be deleted
        return self.write(command)
    
    def ask_scpi(self, command):

        ## This function keeps the scope to debug PNA through SCPI commands from a measurement file
        ## Once the code is standard, this might be deleted

        return self.ask(command)
    
    def get_range_dict(self):

        ## This function calls the range by number
        
        names = self.ask_scpi("SENS:FOM:CAT?").replace('"','').split(',')
        out = {}
        for name in names:


            out[name] = int(self.ask_scpi(f'SENS:FOM:RNUM? "{name}"'))

        return out





        





    

    
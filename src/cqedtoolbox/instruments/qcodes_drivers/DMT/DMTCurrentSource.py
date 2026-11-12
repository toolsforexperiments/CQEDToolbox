"""
A driver to control the DM Technologies Multichannel current source

@author: Kaushik & Angela Kou

Some basic concepts for this current source:

VisaInstrument: 
    This basic current source has up to 8 channels that are accessible to the user. Most functions will be contained
    within the channel function but the temperature of the current source is general and needs to be 
    checked to ensure that it is not overheating. 
 
Channel:
    Each channel only has two basic functions that we can set: the current and the range (range of 4 mA or 40 mA with
     resolution of 0.125 uA or 1.25uA) 
    The only string each channel returns is the following: #chk;channelnumber;range;value
    For some weird reason the range is in mA but the current value is in uA. 

"""

import logging

import numpy as np
import re
from typing import Literal
import pyvisa
import time
from qcodes import (VisaInstrument, Parameter, ParameterWithSetpoints, InstrumentChannel,validators as vals)


def arange_inclusive(start, stop, step):
    """
    Ensures the endpoint is included for both increasing 
    and decreasing sequences.
    """
    # Use absolute values to calculate the required number of points
    num = int(round(abs(start - stop) / abs(step))) + 1
    return np.round(np.linspace(start, stop, num), 2)



class DMTSingleChannel(InstrumentChannel):
    def __init__(
            self, 
            parent: 'DMTCurrentSource', 
            name: str, 
            channel: str,
            **kwargs,
            ) -> None:
        if channel not in ["1","2","3","4","5","6","7","8"]:
            raise ValueError('channel must be an integer string from 1 to 8')
        
        super().__init__(parent, name)
        self._file_name = None

        self.add_parameter("range",
                           label="current range",
                           unit='mA',
                           vals=vals.Enum(4,40),
                           get_cmd=self.get_range,
                           set_cmd=self.set_range)

        self.add_parameter("current",
                           label="current",
                           unit='uA',
                           vals=vals.Numbers(-40000.0, 40000.0),
                           get_cmd=self.get_current,
                           set_cmd=self.set_current)
        self.channel = channel
    
    def get_current(self):
        """Check range and current of the channel"""
        check_response = self.ask('!chk;'+self.channel)
        params = check_response.strip().split(';')
        curr_value = float(params[3].rstrip("uA"))
        return curr_value
    
    def get_range(self):
        """Check range and current of the channel"""
        check_response = self.ask('!chk;'+self.channel)
        params = check_response.strip().split(';')
        curr_range = float(params[2].rstrip("mA"))
        return curr_range

    def set_range(self, new_range: Literal[4,40]):
        """Set current range for the channel"""
        curr_range = self.get_range()
        curr_value = self.get_current()
        if curr_value != 0:
            raise TypeError("You must set the current to zero to change the range")
        elif new_range == curr_range:
            print('This is the range already, nothing will happen')
        else:
            set_string="!set;"+self.channel+";"+str(new_range)+"mA;0"
            success = self.ask(set_string)
            if len(success.strip().split(';'))<3:
                raise TypeError("The source did not accept your input")
    
    def set_current(self, new_current):
        """Set current value for the channel"""
        curr_range = self.get_range()
        temperature = self.root_instrument.get_temp()
        if abs(new_current/1000.0) > abs(curr_range):
            raise TypeError("The value you are setting is larger than the range")
        elif any(ele > 58 for ele in temperature):
            raise TypeError("The source is above 58C. Cool it down before proceeding")
        else:
           set_string="!set;"+self.channel+";"+str(int(curr_range))+"mA;"+str(new_current)
           success = self.ask(set_string)
           if len(success.strip().split(';'))<3:
               raise TypeError("The source did not accept your input.")
           
    def ramp_current(self, 
                     new_current: float, 
                     step: float, 
                     delay: float) -> None:
        curr_value = self.get_current()
        curr_ramp_vals = arange_inclusive(curr_value, new_current +(step/2), step)
        for curr in curr_ramp_vals:
            self.set_current(curr)
            time.sleep(delay)
        self.set_current(new_current)



class DMTCurrentSource(VisaInstrument):
    """
    This is a very simple driver for the MultiChannel Current Source
    that just measures the temperature of the current source and adds the 8 channels

    """

    def __init__(self, name, address=None, **kwargs):

        """
        Initializes the DMT Current Source, and communicates with the wrapper.

        Input:
          name (string)    : name of the instrument
          address (string) : Serial Address
        """
        if address is None:
            raise Exception('Serial address needed')
        logging.info(__name__ + ' : Initializing instrument DMT Current Source')

        super().__init__(name, address, terminator='\n', **kwargs)

        self.ask("!idn")
        channels = []
        for ch in np.linspace(1,8,8,dtype=int): #the instrument supports 8 channels
            ch_name = f"ch{ch}"
            channel = DMTSingleChannel(self, ch_name,f"{ch}")
            self.add_submodule(ch_name, channel)
            channels.append(channel)

    def _open_resource( #local visa open resource that sets the baud rate
        self, address: str, visalib: str | None
        ) -> tuple[pyvisa.resources.MessageBasedResource, str, pyvisa.ResourceManager]:

        resource = super()._open_resource(address, visalib)
        resource.baud_rate = 115200
        return resource
    
    def get_temp(self):
        temp_response = self.ask("!tmp")
        temp_parts = temp_response.strip().split(';')
        front_temp = float(temp_parts[2])
        self.tmp_front = front_temp
        back_temp = float(temp_parts[2])
        self.tmp_back = back_temp
        return (front_temp,back_temp)

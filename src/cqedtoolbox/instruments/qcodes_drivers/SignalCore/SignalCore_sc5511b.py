import ctypes
import ctypes.wintypes
import os
import sys
import numpy as np
from typing import Dict, Optional, Tuple, List, TYPE_CHECKING
from qcodes.instrument import Instrument
from qcodes.validators import Enum, Numbers, PermissiveMultiples, MultiTypeAnd, Ints, Anything, Bool, OnOff
from qcodes.validators import Dict as QCDict
from qcodes.parameters import Group, GroupParameter

if True: #TYPE_CHECKING:
    from collections.abc import Mapping

    from qcodes.parameters import ParamRawDataType

MAXDEVICES = 50
MAXDESCRIPTORSIZE = 9

class ManDate(ctypes.Structure):
    _fields_ = [('year', ctypes.c_uint8),
                ('month', ctypes.c_uint8),
                ('day', ctypes.c_uint8),
                ('hour', ctypes.c_uint8)]


class DeviceInfoT(ctypes.Structure):
    _fields_ = [('product_serial_number', ctypes.c_uint32),
                ('hardware_revision', ctypes.c_float),
                ('firmware_revision', ctypes.c_float),
                ('device_interfaces', ctypes.c_uint8),
                ('man_date', ManDate)]


class ListModeT(ctypes.Structure):
    _fields_ = [('sss_mode', ctypes.c_uint8), #'sss_mode' for sc5511b instead of 'sweep_mode'
                ('sweep_dir', ctypes.c_uint8),
                ('tri_waveform', ctypes.c_uint8),
                ('hw_trigger', ctypes.c_uint8),
                ('step_on_hw_trig', ctypes.c_uint8),
                ('return_to_start', ctypes.c_uint8),
                ('trig_out_enable', ctypes.c_uint8),
                ('trig_out_on_cycle', ctypes.c_uint8)]


class PLLStatusT(ctypes.Structure):
    _fields_ = [('sum_pll_ld', ctypes.c_uint8),
                ('crs_pll_ld', ctypes.c_uint8),
                ('fine_pll_ld', ctypes.c_uint8),
                ('crs_ref_pll_ld', ctypes.c_uint8),
                ('crs_aux_pll_ld', ctypes.c_uint8),
                ('ref_100_pll_ld', ctypes.c_uint8),
                ('ref_10_pll_ld', ctypes.c_uint8),
                ('rf2_pll_ld', ctypes.c_uint8)] #RF2 for SC5510/11 only

class ClockConfigT(ctypes.Structure): #doesn't exist for SC5521
    _fields_ = [('ext_ref_lock_enable', ctypes.c_uint8),
                ('ref_out_select', ctypes.c_uint8),
                ('ext_direct_clocking', ctypes.c_uint8),
                ('ext_ref_freq', ctypes.c_uint8)]

class OperateStatusT(ctypes.Structure):
    _fields_ = [('rf1_lock_mode', ctypes.c_uint8),
                ('rf1_loop_gain', ctypes.c_uint8),
                ('device_access', ctypes.c_uint8),
                ('rf2_standby', ctypes.c_uint8),
                ('rf1_standby', ctypes.c_uint8),
                ('auto_pwr_disable', ctypes.c_uint8),
                ('alc_mode', ctypes.c_uint8),
                ('rf1_out_enable', ctypes.c_uint8),
                ('ext_ref_lock_enable', ctypes.c_uint8),
                ('ext_ref_detect', ctypes.c_uint8),
                ('ref_out_select', ctypes.c_uint8),
                ('list_mode_running', ctypes.c_uint8),
                ('rf1_mode', ctypes.c_uint8),
                ('over_temp', ctypes.c_uint8),
                #('pxi_clk_enable', ctypes.c_uint8), #only for SC5510B
                ('harmonic_ss', ctypes.c_uint8)]


class DeviceStatusT(ctypes.Structure):
    _fields_ = [('list_mode', ListModeT),
                ('operate_status', OperateStatusT),
                ('pll_status', PLLStatusT)]


class DeviceRFParamsT(ctypes.Structure):
    _fields_ = [('rf1_freq', ctypes.c_double),
                ('start_freq', ctypes.c_double),
                ('stop_freq', ctypes.c_double),
                ('step_freq', ctypes.c_double),
                ('sweep_dwell_time', ctypes.c_uint32),
                ('sweep_cycles', ctypes.c_uint32),
                ('buffer_points', ctypes.c_uint32),
                ('rf1_power', ctypes.c_float),
                ('rf2_freq', ctypes.c_uint16)]


error_dict = {0:  'SUCCESS',
            -1: 'ERROR_INVALID_DEVICE_HANDLE',
            -2: 'ERROR_NO_DEVICE',
            -3: 'ERROR_INVALID_DEVICE',
            -4: 'ERROR_MEM_UNALLOCATE',
            -5: 'ERROR_MEM_EXCEEDED',
            -6: 'ERROR_INVALID_REG',
            -7: 'ERROR_INVALID_ARGUMENT',
            -8: 'ERROR_COMM_FAIL',
            -9: 'ERROR_OUT_OF_RANGE',
            -10:'ERROR_PLL_LOCK',
            -11:'ERROR_TIMED_OUT',
            -12:'ERROR_COMM_INIT',
            -13:'ERROR_TIMED_OUT_READ',
            -14:'ERROR_INVALID_INTERFACE'}


def getdict(struct):
    """
    This is copied from online: 
    https://stackoverflow.com/questions/3789372/python-can-we-convert-a-ctypes-structure-to-a-dictionary
    """
    result = {}
    for field, _ in struct._fields_:
         value = getattr(struct, field)
         # if the type is not a primitive and it evaluates to False ...
         if (type(value) not in [int, float, bool]) and not bool(value):
             # it's a null pointer
             value = None
         elif hasattr(value, "_length_") and hasattr(value, "_type_"):
             # Probably an array
             value = list(value)
         elif hasattr(value, "_fields_"):
             # Probably another struct
             value = getdict(value)
         result[field] = value
    return result



class SetFunctionGroup(Group):
    '''This custom Group allows set_cmd to be a function rather than just a string'''
    
    def _set_from_dict(self, calling_dict: Mapping[str, ParamRawDataType]) -> None:
        """
        Use ``set_cmd`` to parse a dict that maps parameter names to parameter
        raw values, and actually perform setting the values.
        """
        if self._set_cmd is None:
            raise RuntimeError("Calling set but no `set_cmd` defined")
        if isinstance(self._set_cmd, str):
            command_str = self._set_cmd.format(**calling_dict)
            if self.instrument is None:
                raise RuntimeError(
                    "Trying to set GroupParameter not attached to any instrument."
                )
            self.instrument.write(command_str)
        else:
            #parameter names need to match the keyword args of self._set_cmd
            #values are already mapped to raw values
            self._set_cmd(**calling_dict)
        for name, p in list(self.parameters.items()):
            p.cache._set_from_raw_value(calling_dict[name])
            
            
    def update(self) -> None:
        """
        Update the values of all the parameters within the group by calling
        the ``get_cmd``.
        """
        if self.instrument is None:
            raise RuntimeError(
                "Trying to update GroupParameter not attached to any instrument."
            )
        if self._get_cmd is None:
            parameter_names = ", ".join(p.full_name for p in self.parameters.values())
            raise RuntimeError(
                f"Cannot update values in the group with "
                f"parameters - {parameter_names} since it "
                f"has no `get_cmd` defined."
            )
        if isinstance(self._get_cmd, str):
            ret = self.get_parser(self.instrument.ask(self._get_cmd))
        else:
            #self._get_cmd needs to return a dict with keywords matching parameter names
            ret = self._get_cmd()
            assert isinstance(ret, dict)
        for name, p in list(self.parameters.items()):
            p.cache._set_from_raw_value(ret[name])


class SC5511B(Instrument):

    __doc__ = 'QCoDeS python driver for the Signal Core SC5511B.'

    def _declareAPI(self):
        self._device_info = DeviceInfoT()
        self._device_status = DeviceStatusT()
        self._rf_parameters = DeviceRFParamsT()
        self._clock_config = ClockConfigT()
        self._list_mode = ListModeT()

        self._dll.sc5511b_usb_search_devices.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_uint32)]
        self._dll.sc5511b_usb_search_devices.restype = ctypes.c_int32
        self._dll.sc5511b_usb_open_device.argtypes = [ctypes.c_char_p, ctypes.wintypes.PHANDLE]
        self._dll.sc5511b_usb_open_device.restype = ctypes.c_int32
        self._dll.sc5511b_usb_close_device.argtypes = [ctypes.wintypes.HANDLE]
        self._dll.sc5511b_usb_close_device.restype = ctypes.c_int32
        self._dll.sc5511b_usb_get_temperature.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(ctypes.c_float)]
        self._dll.sc5511b_usb_get_temperature.restype = ctypes.c_int32
        self._dll.sc5511b_usb_get_device_info.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(DeviceInfoT)]
        self._dll.sc5511b_usb_get_device_info.restype = ctypes.c_int32
        self._dll.sc5511b_usb_get_device_status.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(DeviceStatusT)]
        self._dll.sc5511b_usb_get_device_status.restype = ctypes.c_int32
        self._dll.sc5511b_usb_get_rf_parameters.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(DeviceRFParamsT)]
        self._dll.sc5511b_usb_get_rf_parameters.restype = ctypes.c_int32
        self._dll.sc5511b_usb_get_clock_config.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(ClockConfigT)]
        self._dll.sc5511b_usb_get_clock_config.restype = ctypes.c_int32
        self._dll.sc5511b_usb_set_rf_mode.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_uint8]
        self._dll.sc5511b_usb_set_rf_mode.restype = ctypes.c_int32
        self._dll.sc5511b_usb_set_freq.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_double]
        self._dll.sc5511b_usb_set_freq.restype = ctypes.c_int32
        self._dll.sc5511b_usb_get_signal_phase.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(ctypes.c_float)]
        self._dll.sc5511b_usb_get_signal_phase.restype = ctypes.c_int32
        self._dll.sc5511b_usb_set_signal_phase.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_float]
        self._dll.sc5511b_usb_set_signal_phase.restype = ctypes.c_int32
        self._dll.sc5511b_usb_set_level.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_float]
        self._dll.sc5511b_usb_set_level.restype = ctypes.c_int32
        self._dll.sc5511b_usb_set_standby.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_uint8]
        self._dll.sc5511b_usb_set_standby.restype = ctypes.c_int32
        self._dll.sc5511b_usb_set_output.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_uint8]
        self._dll.sc5511b_usb_set_output.restype = ctypes.c_int32
        self._dll.sc5511b_usb_set_rf2_freq.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_uint16]
        self._dll.sc5511b_usb_set_rf2_freq.restype = ctypes.c_int32
        self._dll.sc5511b_usb_set_rf2_standby.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_uint8]
        self._dll.sc5511b_usb_set_rf2_standby.restype = ctypes.c_int32
        self._dll.sc5511b_usb_get_pulse_mode.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(ctypes.c_uint8)]
        self._dll.sc5511b_usb_get_pulse_mode.restype = ctypes.c_int32
        self._dll.sc5511b_usb_set_pulse_mode.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_uint8]
        self._dll.sc5511b_usb_set_pulse_mode.restype = ctypes.c_int32
        self._dll.sc5511b_usb_get_trigger_edge.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(ctypes.c_uint8)]
        self._dll.sc5511b_usb_get_trigger_edge.restype = ctypes.c_int32
        self._dll.sc5511b_usb_set_trigger_edge.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_uint8]
        self._dll.sc5511b_usb_set_trigger_edge.restype = ctypes.c_int32
        self._dll.sc5511b_usb_synth_self_cal.argtypes = [ctypes.wintypes.HANDLE]
        self._dll.sc5511b_usb_synth_self_cal.restype = ctypes.c_int32
        self._dll.sc5511b_usb_set_auto_level_disable.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_uint8]
        self._dll.sc5511b_usb_set_auto_level_disable.restype = ctypes.c_int32
        self._dll.sc5511b_usb_set_alc_mode.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_uint8]
        self._dll.sc5511b_usb_set_alc_mode.restype = ctypes.c_int32
        self._dll.sc5511b_usb_set_clock_reference.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8]
        self._dll.sc5511b_usb_set_clock_reference.restype = ctypes.c_int32
        self._dll.sc5511b_usb_list_mode_config.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(ListModeT)]
        self._dll.sc5511b_usb_list_mode_config.restype = ctypes.c_int32
        self._dll.sc5511b_usb_list_start_freq.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_double]
        self._dll.sc5511b_usb_list_start_freq.restype = ctypes.c_int32
        self._dll.sc5511b_usb_list_stop_freq.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_double]
        self._dll.sc5511b_usb_list_stop_freq.restype = ctypes.c_int32
        self._dll.sc5511b_usb_list_step_freq.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_double]
        self._dll.sc5511b_usb_list_step_freq.restype = ctypes.c_int32
        self._dll.sc5511b_usb_list_dwell_time.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_uint32]
        self._dll.sc5511b_usb_list_dwell_time.restype = ctypes.c_int32
        self._dll.sc5511b_usb_list_cycle_count.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_uint32]
        self._dll.sc5511b_usb_list_cycle_count.restype = ctypes.c_int32
        self._dll.sc5511b_usb_list_soft_trigger.argtypes = [ctypes.wintypes.HANDLE]
        self._dll.sc5511b_usb_list_soft_trigger.restype = ctypes.c_int32
        self._dll.sc5511b_usb_list_buffer_points.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_uint32]
        self._dll.sc5511b_usb_list_buffer_points.restype = ctypes.c_int32
        self._dll.sc5511b_usb_list_buffer_write.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_float),
                                                            ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
        self._dll.sc5511b_usb_list_buffer_write.restype = ctypes.c_int32
        self._dll.sc5511b_usb_list_buffer_read.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_int32, ctypes.POINTER(ctypes.c_double), 
                                                            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        self._dll.sc5511b_usb_list_buffer_read.restype = ctypes.c_int32
        self._dll.sc5511b_usb_list_buffer_transfer.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_uint8]
        self._dll.sc5511b_usb_list_buffer_transfer.restype = ctypes.c_int32
        self._dll.sc5511b_usb_set_reference_dac.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_uint16]
        self._dll.sc5511b_usb_set_reference_dac.restype = ctypes.c_int32
        self._dll.sc5511b_usb_set_alc_dac.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_uint16]
        self._dll.sc5511b_usb_set_alc_dac.restype = ctypes.c_int32
        self._dll.sc5511b_usb_get_alc_dac.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(ctypes.c_uint16)]
        self._dll.sc5511b_usb_get_alc_dac.restype = ctypes.c_int32
        self._dll.sc5511b_usb_set_synth_mode.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8]
        self._dll.sc5511b_usb_set_synth_mode.restype = ctypes.c_int32
        self._dll.sc5511b_usb_store_default_state.argtypes = [ctypes.wintypes.HANDLE]
        self._dll.sc5511b_usb_store_default_state.restype = ctypes.c_int32
        self._dll.sc5511b_usb_reg_write.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_uint8, ctypes.c_uint64]
        self._dll.sc5511b_usb_reg_write.restype = ctypes.c_int32
        self._dll.sc5511b_usb_reg_read.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_uint8, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
        self._dll.sc5511b_usb_reg_read.restype = ctypes.c_int32
    
    def __init__(self, name: str, sn: str, 
                       dll_path: str='C:\\Program Files\\SignalCore\\SC5510B_11B\\api\\c\\x64\\sc5511b_usb.dll',
                       **kwargs):
        """
        QCoDeS driver for the Signal Core SC5511B.
        This driver has been tested when only one SignalCore is connected to the
        computer but it should work with multiple.

        Args:
        name (str): Name of the instrument.
        sn (str): Serial number of the target instrument
        dll_path (str): Path towards the instrument DLL.
        """

        (super().__init__)(name, **kwargs)

        if type(sn) is not str:
            try:
                sn = str(sn)
            except Exception as e:
                print('could not convert serial number to string')
                raise e
        self._sn = sn
        self._is_open = False
        self._dev_list = []

        # Adapt the path to the computer language
        if sys.platform == 'win32':
            dll_path = os.path.join(os.environ['PROGRAMFILES'], dll_path)
            self._dll = ctypes.WinDLL(dll_path)
            self._handle = ctypes.wintypes.HANDLE()
            self._declareAPI()
        else:
            raise EnvironmentError(f"{self.__class__.__name__} is supported only on Windows platform")

        self._initialize()

        self.add_parameter(name='temperature',
                           docstring='Return the microwave source internal temperature.',
                           label='Device temperature',
                           unit='celsius',
                           get_cmd=self._get_temperature)
                           
        self.add_parameter(name='over_temperature',
                           docstring='Flag for showing if device is over temperature',
                           label='Over temperature warning',
                           val_mapping={'ok':0,'over_temp':1},
                           get_cmd=lambda: self._get_operate_status('over_temp'))

        self.add_parameter(name='state',
                           docstring='Output on/off state',
                           val_mapping={'on':1,'off':0},
                           set_cmd=self._set_state,
                           get_cmd=self._get_state)

        self.add_parameter(name='standby',
                           docstring='Enter/exit standby mode for power savings',
                           val_mapping = {'on':1, 'off':0},
                           set_cmd=self._set_standby,
                           get_cmd=self._get_standby,
                           post_delay=1) #wait 1 second after setting to stabilize in case we came out of standby 
        
        self.add_parameter(name='standby_rf2',
                           docstring='Enter/exit standby mode for power savings',
                           val_mapping = {'on':1, 'off':0},
                           set_cmd=self._set_standby_rf2,
                           get_cmd=self._get_standby_rf2,
                           post_delay=1) #wait 1 second after setting to stabilize in case we came out of standby
        
        self.add_parameter(name='power',
                           docstring='.',
                           label='Power',
                           unit='dbm',
                           vals=Numbers(-35,20),
                           set_cmd=self._set_power,
                           get_cmd=self._get_power)

        self.add_parameter(name='frequency',
                           docstring='.',
                           label='Frequency',
                           unit='Hz',
                           vals=Numbers(100e6,20e9),
                           set_cmd=self._set_frequency,
                           get_cmd=self._get_frequency)
                           
        self.add_parameter(name='frequency_rf2',
                           docstring='Auxiliary channel frequency - must be multiple of 25 MHz',
                           label='RF2 Frequency',
                           unit='Hz',
                           vals=MultiTypeAnd(PermissiveMultiples(25e6),Numbers(100e6,3e9)),
                           set_cmd=self._set_frequency,
                           get_cmd=self._get_frequency)
        
        self.add_parameter(name='phase',
                           docstring='.',
                           label='Phase',
                           unit='deg',
                           vals=Numbers(-360,360),
                           set_cmd=self._set_phase,
                           get_cmd=self._get_phase)
                           
        self.add_parameter(name='alc',
                           label='ALC',
                           docstring='Enable/disable ALC',
                           val_mapping={'on':0,'off':1},
                           set_cmd=self._disable_alc,
                           get_cmd=lambda: self._get_operate_status('alc_mode'))                    
                           
        self.add_parameter(name='auto_level',
                           label='Enable/disable automatic levelling when frequency is changed',
                           docstring='Enable/disable auto-levelling',
                           val_mapping={'on':0,'off':1},
                           set_cmd=self._disable_sw_auto_level,
                           get_cmd=lambda: self._get_operate_status('auto_pwr_disable'))                              
                           
        self.add_parameter(name='pulse_modulation',
                           label='Pulse Modulation',
                           docstring='Enable/disable pulse modulation',
                           val_mapping={'on':1,'off':0},
                           set_cmd=self._set_pulse_mode,
                           get_cmd=self._get_pulse_mode) 


        self.add_parameter(name='ref_dac',
                           docstring='DAC to adjust frequency of internal TCXO',
                           label='Reference DAC',
                           vals=Ints(0,16383),
                           get_cmd=None, #device doesn't support read of ref_dac but I'll return the cached value
                           set_cmd=self._set_reference_dac)

        self.add_parameter(name='alc_dac',
                           docstring='DAC for fine control of output power',
                           label='ALC DAC',
                           vals=Ints(0,16383),
                           set_cmd=self._set_alc_dac,
                           get_cmd=self._get_alc_dac)

        self.add_parameter(name='sum_pll_locked',
                           docstring='Flag for showing if summing PLL is locked',
                           label='Sum PLL locked',
                           val_mapping={'unlocked':0,'locked':1},
                           get_cmd=lambda: self._get_pll_status('sum_pll_ld'))

        self.add_parameter(name='coarse_pll_locked',
                           docstring='Flag for showing if coarse PLL is locked',
                           label='Coarse PLL locked',
                           val_mapping={'unlocked':0,'locked':1},
                           get_cmd=lambda: self._get_pll_status('crs_pll_ld'))
                           
        self.add_parameter(name='fine_pll_locked',
                           docstring='Flag for showing if fine PLL is locked',
                           label='Fine PLL locked',
                           val_mapping={'unlocked':0,'locked':1},
                           get_cmd=lambda: self._get_pll_status('fine_pll_ld'))                         
                           
        self.add_parameter(name='coarse_ref_pll_locked',
                           docstring='Flag for showing if coarse reference PLL is locked',
                           label='Coarse reference PLL locked',
                           val_mapping={'unlocked':0,'locked':1},
                           get_cmd=lambda: self._get_pll_status('crs_ref_pll_ld'))                           
                           
        self.add_parameter(name='coarse_aux_pll_locked',
                           docstring='Flag for showing if coarse aux PLL is locked',
                           label='Coarse aux PLL locked',
                           val_mapping={'unlocked':0,'locked':1},
                           get_cmd=lambda: self._get_pll_status('crs_aux_pll_ld'))                           
                           
        self.add_parameter(name='ref_100_pll_locked',
                           docstring='Flag for showing if 100 MHz reference PLL is locked',
                           label='100MHz reference PLL locked',
                           val_mapping={'unlocked':0,'locked':1},
                           get_cmd=lambda: self._get_pll_status('ref_100_pll_ld'))                           
                           
        self.add_parameter(name='ref_10_pll_locked',
                           docstring='Flag for showing if 10 MHz reference PLL is locked',
                           label='10MHz reference PLL locked',
                           val_mapping={'unlocked':0,'locked':1},
                           get_cmd=lambda: self._get_pll_status('ref_10_pll_ld'))                           
                           
        self.add_parameter(name='rf2_pll_locked',
                           docstring='Flag for showing if RF2 PLL is locked',
                           label='RF2 PLL locked',
                           val_mapping={'unlocked':0,'locked':1},
                           get_cmd=lambda: self._get_pll_status('rf2_pll_ld'))                           
                           
        self.add_parameter(name='start_frequency',
                           docstring='Sweep start frequency',
                           label='Sweep start frequency',
                           unit='Hz',
                           vals=Numbers(100e6,20e9),
                           set_cmd=self._set_list_start_freq,
                           get_cmd=lambda: self._get_rf_params('start_freq'))

        self.add_parameter(name='stop_frequency',
                           docstring='Sweep stop frequency',
                           label='Sweep stop frequency',
                           unit='Hz',
                           vals=Numbers(100e6,20e9),
                           set_cmd=self._set_list_stop_freq,
                           get_cmd=lambda: self._get_rf_params('stop_freq'))

        self.add_parameter(name='step_frequency',
                           docstring='Sweep step frequency',
                           label='Sweep step frequency',
                           unit='Hz',
                           vals=Numbers(-20e9,20e9),
                           set_cmd=self._set_list_step_freq,
                           get_cmd=lambda: self._get_rf_params('step_freq'))

        self.add_parameter(name='cycle_count',
                           docstring='Number of repeat cycles for list/sweep',
                           label='Sweep/list cycle count',
                           vals=Ints(0,int(2**32)-1),
                           set_cmd=self._set_list_cycle_count,
                           get_cmd=lambda: self._get_rf_params('sweep_cycles'))

        self.add_parameter(name='dwell_time',
                           docstring='Dwell time for each step of list/sweep',
                           label='Sweep/list dwell time',
                           unit='ms',
                           vals=MultiTypeAnd(PermissiveMultiples(0.5),Numbers(0.5,2**31-1)),
                           set_cmd=lambda x: self._set_list_dwell_time(int(x//0.5)),
                           get_cmd=lambda: self._get_rf_params('sweep_dwell_time')/2)

        self.add_parameter(name='trigger_edge',
                           docstring='Trigger on rising vs falling  edge',
                           label='Coarse aux PLL locked',
                           val_mapping={'rising':1,'falling':0},
                           set_cmd=self._set_trigger_edge,
                           get_cmd=self._get_trigger_edge) 


        self.add_parameter(name='sweep_shape',
                           label='Sweep shape',
                           docstring='Sweep as sawtooth or triangle',
                           val_mapping={"sawtooth":0,"triangle":1},
                           initial_value='sawtooth',
                           parameter_class=GroupParameter)

        self.add_parameter(name='sweep_direction',
                           label='Sweep direction',
                           docstring='Sweep first-to-last or last-to-first',
                           val_mapping={"first_to_last":0,"last_to_first":1},
                           initial_value='first_to_last',
                           parameter_class=GroupParameter)

        self.add_parameter(name='sweep_type',
                           label='Sweep type',
                           docstring='Configure to use sweep mode or list mode',
                           val_mapping={"list":0,"sweep":1},
                           initial_value='sweep',
                           parameter_class=GroupParameter)

        self.add_parameter(name='sweep_end',
                           label='Sweep end behavior',
                           docstring='Configure sweep to remain at last frequency or return to starting frequency when complete',
                           val_mapping={"last":0,"start":1},
                           initial_value='start',
                           parameter_class=GroupParameter)

        self.add_parameter(name='trigger_source',
                           label='Trigger source',
                           docstring='Select the trigger source(internal or external)',
                           val_mapping={"internal":0,"external":1},
                           initial_value='internal',
                           parameter_class=GroupParameter)

        self.add_parameter(name='hw_trigger_mode',
                           label='Hardware trigger mode',
                           docstring='Single step per trigger or playback indefinitely on first trigger',
                           val_mapping={"start_stop":0,"step":1},
                           initial_value='step',
                           parameter_class=GroupParameter)

        self.add_parameter(name='trigger_out',
                           label='Trigger out enable',
                           docstring='Enable/disable the trigger out signal',
                           val_mapping={'on':1,'off':0},
                           initial_value='off',
                           parameter_class=GroupParameter)

        self.add_parameter(name='trigger_out_mode',
                           label='Trigger out mode',
                           docstring='Output trigger once per step or once per cycle',
                           val_mapping={"step":0,"cycle":1},
                           initial_value='step',
                           parameter_class=GroupParameter)
        
        self.list_mode_group = SetFunctionGroup([self.sweep_shape, self.sweep_direction, 
                                        self.sweep_type, self.sweep_end, 
                                        self.trigger_source, self.hw_trigger_mode, 
                                        self.trigger_out, self.trigger_out_mode],
                                        set_cmd = self._set_list_mode,
                                        get_cmd = self._get_list_mode)
                        

        self.add_parameter(name='spur_suppression',
                           label='Spur suppression',
                           docstring='Enables/disables automatic spur suppression (changes PLL mode to avoid boundry spurs)',
                           val_mapping={'on':1,'off':0},
                           parameter_class=GroupParameter)

        self.add_parameter(name='loop_gain',
                           label='PLL loop gain',
                           docstring='Set PLL loop gain (normal=low close in noise, low=low far out noise and spur suppression)',
                           val_mapping={"normal":0,"low":1},
                           parameter_class=GroupParameter)

        self.add_parameter(name='lock_mode',
                           label='PLL lock mode',
                           docstring='Set PLL to harmonic offset or fractional N mode',
                           val_mapping={"harmonic":0,"frac_n":1},
                           parameter_class=GroupParameter)

        self.synth_mode_group = SetFunctionGroup([self.spur_suppression, self.loop_gain, self.lock_mode],
                                set_cmd=self._set_synth_mode,
                                get_cmd=self._get_synth_mode)
                           
                           
        self.add_parameter(name='ext_ref_freq',
                           label='External reference frequency',
                           docstring='Must match frequency of external reference provided (in Hz)',
                           val_mapping={10e6:0, 100e6:1},
                           initial_value=10e6,
                           parameter_class=GroupParameter)  
                           
        self.add_parameter(name='ref_out_freq',
                           label='Reference output frequency',
                           docstring='Sets frequency of reference output (in Hz)',
                           val_mapping={10e6:0, 100e6:1},
                           initial_value=10e6,
                           parameter_class=GroupParameter)                            

        self.add_parameter(name='ext_direct_clocking',
                           label='External direct clocking',
                           docstring='Directly clock RF1 PLL from external reference signal (must be 100 MHz)',
                           val_mapping={'on':1,'off':0},
                           initial_value='off',
                           parameter_class=GroupParameter) 

        self.add_parameter(name='clock_reference',
                           label='Clock Reference',
                           docstring='Select the clock reference (internal or external)',
                           val_mapping={"internal":0,"external":1},
                           initial_value='internal',
                           parameter_class=GroupParameter) 
        
        self.clock_ref_group = SetFunctionGroup([self.ext_ref_freq, self.ref_out_freq, self.ext_direct_clocking, self.clock_reference],
                                set_cmd=self._set_clock_reference,
                                get_cmd=self._get_clock_reference)
                    
        
        self.add_parameter(name='list_num_points',
                           docstring='Number of entries to use for list mode',
                           label='Number of list entries',
                           vals=Ints(1,2048),
                           set_cmd=self._set_list_buffer_points,
                           get_cmd=lambda: self._get_rf_params('buffer_points'))
        
        #TODO - consider making a custom validator because current validator is insufficient
        #Dict validator only checks for illegal keys - it doesn't verify all keys are present
        #Dict validator also doesn't support checking values within dict
        self.add_parameter(name='list_buffer',
                           docstring='List of entries to use for list mode',
                           label='List buffer',
                           set_cmd=self._write_list_buffer_auto,
                           get_cmd=self._read_list_buffer_auto,
                           vals=QCDict(['freq','pow','dwell'])) 

        self.add_parameter(name='sw_trigger',
                           docstring='Generate a software trigger',
                           label='Software trigger',
                           vals=Anything(),
                           set_cmd=lambda x: self._set_list_soft_trigger(),
                           inter_delay=2e-4) #enforcing max 200 us trigger rate
        self.add_parameter(name='store_default',
                           docstring='Store the current state as default',
                           label='Store current state as default',
                           vals=Anything(),
                           set_cmd=lambda x: self._store_default_state())
        self.add_parameter(name='transfer_buffer',
                           docstring='Transfer the list buffer from RAM to EEPROM or vice versa',
                           label='Transfer buffer to/from RAM/EEPROM',
                           val_mapping={'to_EEPROM':0,'from_EEPROM':1},
                           set_cmd=self._transfer_list_buffer)
        self.add_parameter(name='self_cal',
                           docstring='Begin the self calibration cycle. Only needed if sum PLL fails in harmonic mode.',
                           label='Start self calibration',
                           vals=Anything(),
                           set_cmd=lambda x: self._start_self_calibration(),
                           post_delay=3)
        self.add_parameter(name='ext_ref_detected',
                           label='External reference detected',
                           docstring='Indicator for whether an external reference signal is detected by the instrument',
                           vals=Bool(),
                           get_cmd=lambda: self._get_operate_status('ext_ref_detect'),
                           get_parser=bool) 
        self.add_parameter(name='active_sweep',
                           label='Actively sweeping',
                           docstring='Indicator for whether a sweep is currently active',
                           vals=Bool(),
                           get_cmd=lambda: self._get_operate_status('list_mode_running'),
                           get_parser=bool)         

        
        self.connect_message()
        
        
    def _err_check(self,code,description='') -> None:
        if code:
            raise RuntimeError(f"Error in {description}: " + error_dict[code])
       
        
    def close(self) -> None:
        self._finalize()
        super().close()
        
       
# Init --------------------------------------------------------------    
    def _initialize(self) -> None: 
        self._search_devices()
        self._open_device(self._sn)
        return

    def _finalize(self) -> None:
        if self._is_open:
            self._close_device()
        return
        

# Open-close device -------------------------------------------------------------
    def _search_devices(self) -> None:
        _devices_number = ctypes.c_uint32()
        buffers = [ctypes.create_string_buffer(MAXDESCRIPTORSIZE + 1) for bid in range(MAXDEVICES)]
        _buffer_pointer_array = (ctypes.c_char_p * MAXDEVICES)()
        for device in range(MAXDEVICES):
            _buffer_pointer_array[device] = ctypes.cast(buffers[device], ctypes.c_char_p)
        _buffer_pointer_array_p = ctypes.cast(_buffer_pointer_array, ctypes.POINTER(ctypes.c_char_p))
        ans = self._dll.sc5511b_usb_search_devices(_buffer_pointer_array_p, ctypes.byref(_devices_number))
        self._err_check(ans,'search devices')
        self._dev_list = [_buffer_pointer_array_p[i].decode("utf-8") for i in range(0, _devices_number.value)] 

    def _open_device(self, sn:str) -> None:
        if self._is_open:
            self._close_device()
            self._open_device(sn)
        else:
            if sn in self._dev_list:
                ser_num = ctypes.c_char_p(sn.encode('utf-8'))
                ans = self._dll.sc5511b_usb_open_device(ser_num, ctypes.byref(self._handle))
                self._err_check(ans,'open device')
                self._is_open=True
            else:
                raise RuntimeError(f'Device with serial number {sn} not found on system. Found devices are {self._dev_list}.')

    def _close_device(self):
        ans = self._dll.sc5511b_usb_close_device(self._handle)
        self._is_open=False
        self._err_check(ans,'close device')         

# Device info
    def _get_temperature(self) -> float:
        temp = ctypes.c_float()
        ans = self._dll.sc5511b_usb_get_temperature(self._handle, ctypes.byref(temp))
        self._err_check(ans,'get temperature')
        return temp.value

    def _get_dev_info(self, key:str|None = None):
        ans = self._dll.sc5511b_usb_get_device_info(self._handle, ctypes.byref(self._device_info))
        self._err_check(ans,'get device info')
        if key is not None:
            return getdict(self._device_info)[key]
        else:
            return getdict(self._device_info)

    def _get_dev_status(self, key:str|None = None):
        ans = self._dll.sc5511b_usb_get_device_status(self._handle, ctypes.byref(self._device_status))
        self._err_check(ans,'get device status')
        if key is not None:
            return getdict(self._device_status)[key]
        else:
            return getdict(self._device_status)
        
    def _get_clock_config(self, key:str|None = None):
        ans = self._dll.sc5511b_usb_get_clock_config(self._handle, ctypes.byref(self._clock_config))
        self._err_check(ans,'get clock configuration')
        if key is not None:
            return getdict(self._clock_config)[key]
        else:
            return getdict(self._clock_config)

    def _get_list_mode_status(self, key:str|None = None):
        if key is not None:
            return self._get_dev_status('list_mode')[key]
        else:
            return self._get_dev_status('list_mode')

    def _get_operate_status(self, key:str|None = None):
        if key is not None:
            return self._get_dev_status('operate_status')[key]
        else:
            return self._get_dev_status('operate_status')

    def _get_pll_status(self, key:str|None = None):
        if key is not None:
            return self._get_dev_status('pll_status')[key]
        else:
            return self._get_dev_status('pll_status')

    def _get_rf_params(self, key:str|None = None):
        ans = self._dll.sc5511b_usb_get_rf_parameters(self._handle, ctypes.byref(self._rf_parameters))
        self._err_check(ans,'get rf parameters')
        if key is not None:
            return getdict(self._rf_parameters)[key]
        else:
            return getdict(self._rf_parameters)

    def _get_rf_mode(self):
        return self._get_operate_status('rf1_mode')

    def _set_rf_mode(self, enable):
        """
        values = {'Sweep': 1, 'Fixed': 0}
        """
        set_mode = ctypes.c_uint8(enable)
        ans = self._dll.sc5511b_usb_set_rf_mode(self._handle, set_mode)
        self._err_check(ans,'set rf mode')

    def _get_frequency(self) -> float:
        return self._get_rf_params('rf1_freq')

    def _set_frequency(self, freq:float) -> None:
        set_f = ctypes.c_double(freq)
        ans = self._dll.sc5511b_usb_set_freq(self._handle, set_f)
        self._err_check(ans,'set frequency')

    def _get_phase(self) -> float:
        sig_phase = ctypes.c_float()
        ans = self._dll.sc5511b_usb_get_signal_phase(self._handle, ctypes.byref(sig_phase))
        self._err_check(ans,'get signal phase')
        return sig_phase.value

    def _set_phase(self, sig_phase:float) -> None:
        set_p = ctypes.c_float(round(float(sig_phase), 1))
        ans = self._dll.sc5511b_usb_set_signal_phase(self._handle, set_p)
        self._err_check(ans,'set signal phase')

    def _get_power(self) -> float:
        return self._get_rf_params('rf1_power')

    def _set_power(self, sig_power:float) -> None:
        set_power = ctypes.c_float(float(sig_power))
        ans = self._dll.sc5511b_usb_set_level(self._handle, set_power)
        self._err_check(ans,'set power level')

    def _get_standby(self) -> bool:
        return bool(self._get_operate_status('rf1_standby'))

    def _set_standby(self, enable:bool) -> None:
        set_standby = ctypes.c_uint8(int(enable))
        ans = self._dll.sc5511b_usb_set_standby(self._handle, set_standby)
        self._err_check(ans,'set standby')

    def _get_state(self) -> bool:
        return bool(self._get_operate_status('rf1_out_enable'))

    def _set_state(self, enable:bool) -> None:
        set_o = ctypes.c_uint8(int(enable))
        ans = self._dll.sc5511b_usb_set_output(self._handle, set_o)
        self._err_check(ans,'set state')

    def _get_frequency_rf2(self) -> float:
        return self._get_rf_params('rf2_freq')

    def _set_frequency_rf2(self, freq:float) -> None:
        set_freq = ctypes.c_uint16(freq)
        ans = self._dll.sc5511b_usb_set_rf2_freq(self._handle, set_freq)
        self._err_check(ans,'set rf2 frequency')

    def _get_standby_rf2(self) -> bool:
        return bool(self._get_operate_status('rf2_standby'))

    def _set_standby_rf2(self, enable:bool) -> None:
        set_standby = ctypes.c_uint8(int(enable))
        ans = self._dll.sc5511b_usb_set_rf2_standby(self._handle, set_standby)
        self._err_check(ans,'set rf2 standby')

    def _get_pulse_mode(self) -> bool:
        p_mode = ctypes.c_uint8()
        ans = self._dll.sc5511b_usb_get_pulse_mode(self._handle, ctypes.byref(p_mode))
        self._err_check(ans,'get pulse mode')
        return bool(p_mode.value)

    def _set_pulse_mode(self, enable:bool) -> None:
        '''
        enable: False = disable, True = enable to use external pulse pin 
        '''
        set_p_mode = ctypes.c_uint8(bool(enable))
        ans = self._dll.sc5511b_usb_set_pulse_mode(self._handle, set_p_mode)
        self._err_check(ans,'set pulse mode')

    def _get_trigger_edge(self) -> int:
        """
        values = {'Rise': 1, 'Fall': 0}
        """
        trigger = ctypes.c_uint8()
        ans = self._dll.sc5511b_usb_get_trigger_edge(self._handle, ctypes.byref(trigger))
        self._err_check(ans,'get trigger edge')
        return trigger.value

    def _set_trigger_edge(self, edge:int) -> None:
        set_trigger = ctypes.c_uint8(edge)
        ans = self._dll.sc5511b_usb_set_trigger_edge(self._handle, set_trigger)
        self._err_check(ans,'set trigger edge')

    def _start_self_calibration(self) -> None:
        ans = self._dll.sc5511b_usb_synth_self_cal(self._handle)
        self._err_check(ans,'self calibration')
            
    def _set_list_start_freq(self, start_freq:float) -> None:
        """frequency in Hz"""
        frequency=ctypes.c_double(start_freq)
        ans = self._dll.sc5511b_usb_list_start_freq(self._handle, frequency)
        self._err_check(ans,'set list start frequency')

    def _set_list_stop_freq(self, stop_freq:float) -> None:
        """Sets the list stop frequency"""
        frequency = ctypes.c_double(stop_freq)
        ans = self._dll.sc5511b_usb_list_stop_freq(self._handle, frequency)
        self._err_check(ans,'set list stop frequency')
            
    def _set_list_step_freq(self, step_freq:float) -> None:
        """Sets the list step frequency"""
        frequency = ctypes.c_double(step_freq)
        ans = self._dll.sc5511b_usb_list_step_freq(self._handle, frequency)
        self._err_check(ans,'set list step frequency')
    
    def _set_list_cycle_count(self, cycle_num:int = 0) -> None:
        """Sets the list cycle count value"""
        cycle_count = ctypes.c_uint(cycle_num)
        ans = self._dll.sc5511b_usb_list_cycle_count(self._handle, cycle_count)
        self._err_check(ans,'set list cycle count')

    def _set_list_dwell_time(self, tunit:int) -> None:
        """Dwell time = tunit*500 us, min value of tunit = 1"""
        dwell = ctypes.c_uint32(tunit)
        ans = self._dll.sc5511b_usb_list_dwell_time(self._handle, dwell)
        self._err_check(ans,'set list dwell time')
    
    def _set_list_soft_trigger(self) -> None:
        """Sets the list soft trigger
        """
        ans = self._dll.sc5511b_usb_list_soft_trigger(self._handle)
        self._err_check(ans,'set list soft trigger')
            
    
    def _set_list_mode(self, sweep_type:bool=1, sweep_direction:bool=0, sweep_shape:bool=0, 
                        trigger_source:bool=0, hw_trigger_mode:bool=0, sweep_end:bool=0,
                        trigger_out:bool=1, trigger_out_mode:bool=0) -> None:
        """
        Configures list mode
        sweep_type= 0: List mode, 1:sweep mode
        sweep_direction= 0:Forward, 1: reverse
        sweep_shape= 0:Sawtooth waveform, 1: Triangular waveform
        trigger_source= 0:Software trigger, 1: Hardware trigger
        hw_trigger_mode= 0:Start/stop behavior, 1: Step on trigger, see manual for more details
        sweep_end = 0:stop at end of sweep/list, 1:return to start
        trigger_out= 0:No output trigger, 1: Output trigger enabled on trigger pin
        trigger_out_mode= 0: puts out a trigger pulse at each frequency change, 1: trigger pulse at the completion of each sweep/list cycle
        """
        lm = ListModeT(sss_mode=sweep_type, sweep_dir=sweep_direction, tri_waveform=sweep_shape, 
                        hw_trigger=trigger_source, step_on_hw_trig=hw_trigger_mode, 
                        return_to_start=sweep_end, trig_out_enable=trigger_out,
                        trig_out_on_cycle=trigger_out_mode)
        ans = self._dll.sc5511b_usb_list_mode_config(self._handle, ctypes.byref(lm))
        self._err_check(ans,'set list mode')
    
    def _get_list_mode(self) -> dict:
        lm = self._get_list_mode_status()
        return {'sweep_shape':lm['tri_waveform'], 'sweep_direction':lm['sweep_dir'],
                'sweep_type':lm['sss_mode'], 'sweep_end':lm['return_to_start'],
                'trigger_source':lm['hw_trigger'], 'hw_trigger_mode':lm['step_on_hw_trig'],
                'trigger_out':lm['trig_out_enable'], 'trigger_out_mode':lm['trig_out_on_cycle']}
    
    def _set_clock_reference(self, ext_ref_freq:bool=0, ext_direct_clocking:bool=0, 
                            ref_out_freq:bool=0, clock_reference:bool=1) -> None:
        """Sets the clock reference
        input:    ext_ref_freq: 0 10 MHz, 1 100 MHz
            ext_direct_clocking: If ext_ref_freq is set to 100 MHz, it may be used to directly clock
                            the core synthesizer, improving unit to unit phase stability.
            clock_reference: 1 locks the 100 MHz reference to the external 10 MHz source
            ref_out_freq:    1 exports 100 MHz instead of the default 10 MHz
        """
        ext_ref = ctypes.c_ubyte(ext_ref_freq)
        ext_direct_lock = ctypes.c_ubyte(ext_direct_clocking)
        high = ctypes.c_ubyte(ref_out_freq)
        lock = ctypes.c_ubyte(clock_reference)
        ans = self._dll.sc5511b_usb_set_clock_reference(self._handle, ext_ref, ext_direct_lock, high, lock)
        self._err_check(ans,'set clock reference')
    
    def _get_clock_reference(self) -> dict:
        clk_cfg = self._get_clock_config()
        return {'clock_reference':clk_cfg['ext_ref_lock_enable'],
                'ref_out_freq':clk_cfg['ref_out_select'],
                'ext_direct_clocking':clk_cfg['ext_direct_clocking'],
                'ext_ref_freq':clk_cfg['ext_ref_freq']}
    
    def _set_ref_out(self, ref_out_select:bool=0):
        #not sure if this will work and also subject to breaking if parameter names change
        #inst.param() is supposed to get the parameter value
        erf = self.ext_ref_freq() #self._get_clock_config('ext_ref_freq')
        edc = self.ext_direct_clocking()
        erle = self.ext_ref_lock_enable()
        self._set_clock_reference(erf, edc, ref_out_select, erle)
    
    def _disable_sw_auto_level(self, enable:bool) -> None:
        """Sets the auto level to either enable or disable
        """
        c_enable = ctypes.c_ubyte(enable)
        ans = self._dll.sc5511b_usb_set_auto_level_disable(self._handle, c_enable)
        self._err_check(ans,'set auto level disable')  
      
    def _disable_alc(self, disable:bool = 0) -> None:
        """Sets the ALC to close(0) or open (1) mode operation for channel RF1"""
        mode = ctypes.c_ubyte(disable)
        ans = self._dll.sc5511b_usb_set_alc_mode(self._handle, mode)
        self._err_check(ans,'set alc mode') 

    def _set_synth_mode(self, spur_suppression:bool = 0, loop_gain:bool = 0, lock_mode:bool = 0) -> None:
        """
        sets the rf mode of the device
        spur_suppression: (only takes effect when lock mode is harmonic)
            input: integer = 0 = disable spur suppress, 1 = spur suppress by lowering loop gain and/or ping ponging between lock modes automatically
        loop_gain:
            input: integer = 0 = Normal gain, 1 = low gain
        lock_mode:
            input: integer = 0 = harmonic, 1 = fractional N
        """
        dss = ctypes.c_uint8(spur_suppression)
        llg = ctypes.c_uint8(loop_gain)
        lm = ctypes.c_uint8(lock_mode)
        print(f"spur={dss},gain={llg},mode={lm}")
        ans = self._dll.sc5511b_usb_set_synth_mode(self._handle, dss, llg, lm)
        self._err_check(ans,'set synth mode')
        
    def _get_synth_mode(self) -> dict:
        full_stat = self._get_operate_status()
        return {'spur_suppression':full_stat['harmonic_ss'], 
                'loop_gain':full_stat['rf1_loop_gain'],
                'lock_mode':full_stat['rf1_lock_mode']}

    def _set_spur_suppression(self, disable_spur_suppress:bool = 0) -> None:
        #not sure if this will work and also subject to breaking if parameter names change
        #inst.param() is supposed to get the parameter value
        llg = self.low_loop_gain()
        lm = self.lock_mode()
        self._set_synth_mode(disable_spur_suppress, llg, lm)

    def _set_reference_dac(self, d_value:int = 0) -> None:
        """Sets a value to the reference dac which is a correction to the TCXO frequency
        input: integer
        """
        dac_value = ctypes.c_uint16(d_value)
        ans = self._dll.sc5511b_usb_set_reference_dac(self._handle, dac_value)
        self._err_check(ans,'set reference dac')

    def _set_alc_dac(self, alc_value:int = 0) -> None:
        """Sets the ALC DAC value which is the way to get fine amplitude control
        input: integer
        """
        dac_value = ctypes.c_uint16(alc_value)
        ans = self._dll.sc5511b_usb_set_alc_dac(self._handle, dac_value)
        self._err_check(ans,'set alc dac')
        
    def _get_alc_dac(self) -> int:
        """Gets the alc dac value
        """
        dac_value = ctypes.c_uint16()
        ans = self._dll.sc5511b_usb_get_alc_dac(self._handle, ctypes.byref(dac_value))
        self._err_check(ans,'get alc dac')
        return dac_value.value

    def _store_default_state(self) -> None:
        """Stores the default state of the device
        """
        ans = self._dll.sc5511b_usb_store_default_state(self._handle)
        self._err_check(ans,'store default state')

    def _set_list_buffer_points(self, l_points:int = 0) -> None:
        """Sets the number of list points in the list buffer to sweep or step through
        input: integer
        """
        list_points = ctypes.c_uint32(l_points)
        ans = self._dll.sc5511b_usb_list_buffer_points(self._handle, list_points)
        self._err_check(ans,'list buffer points')

    def _write_list_buffer(self, lFreq:List[float], lPow:List[float], lDwell:List[float], buflen:int) -> None:
        """Writes to the list buffer
        input:    freq double array in Hz.
            level float array in 100th of dBm
            float dwell_time in milliseconds, step .5ms
            len size of array
        """
        lbuf = ctypes.c_int32(buflen)
        freqs = (ctypes.c_double * buflen)(*lFreq)
        pows = (ctypes.c_float * buflen)(*lPow)
        dwells = (ctypes.c_float * buflen)(*lDwell)
        ans = self._dll.sc5511b_usb_list_buffer_write(self._handle, freqs, pows, dwells, lbuf)
        self._err_check(ans,'list buffer write')
    
    def _write_list_buffer_auto(self, entries:dict):
        """Wrapper around _write_list_buffer that uses a single dictionary argument
        input: dict with format
                key 'freq', val = list or array of doubles in Hz
                key 'pow', val = list or array of doubles (.01 dBm increments)
                key 'dwell', val = list or array of doubles in ms (.5ms increments)                   
            """
        required_keys = ('freq','pow','dwell')
        assert all(key in entries.keys() for key in required_keys)
        assert all(isinstance(entries[key],(list,np.ndarray)) for key in required_keys)
        lens = [len(entries[key]) for key in required_keys]
        assert len(set(lens))==1 #for now we only support equal length lists
        self._write_list_buffer(list(entries['freq']), list(entries['pow']), list(entries['dwell']), lens[0])
        
    def _read_list_buffer(self, buflen:int) -> Tuple[List[float],List[float],List[float]]:
        """Reads back values from the list buffer and returns as dict
        """
        lbuf = ctypes.c_int32(buflen)
        freqs = (ctypes.c_double * buflen)()
        pows = (ctypes.c_float * buflen)()
        dwells = (ctypes.c_float * buflen)()
        ans = self._dll.sc5511b_usb_list_buffer_read(self._handle, lbuf, freqs, pows, dwells)
        self._err_check(ans,'list buffer read')
        return {'freq':freqs[:],'pow':pows[:],'dwell':dwells[:] }

    def _read_list_buffer_auto(self) -> Tuple[List[float],List[float],List[float]]:
        """Reads back values from the list buffer
        Uses the list_num_points parameter to determine how many entries to read
        """
        buflen = self._get_rf_params('buffer_points')
        return self._read_list_buffer(buflen)
        
    def _transfer_list_buffer(self, transfer_mode:bool = 0) -> None:
        """Transfers frequency list buffer from RAM to EEPROM or vice versa
        mode=0: RAM to EEPROM, mode=1: EEPROM to RAM
        """
        transfer = ctypes.c_ubyte(transfer_mode)
        ans = self._dll.sc5511b_usb_list_buffer_transfer(self._handle, transfer)
        self._err_check(ans,'list buffer transfer')     
    
    #Could be dangerous to expose this to a user - nothing uses it at the moment
    def _write_reg(self, reg:int, ins_word:int) -> None:
        """Writes the specific value to the register, see documentation for more info
        """
        reg_byte = ctypes.c_uint8(reg)
        instruct_word = ctypes.c_uint64(ins_word)
        ans = self._dll.sc5511b_usb_reg_write(self._handle, reg_byte, instruct_word)
        self._err_check(ans,'register write')

    def _read_reg(self, reg:int, ins_word:int) -> int:
        """Reads back the specific value back, see documentation for more info
        """
        reg_byte = ctypes.c_ubyte(reg)
        instruct_word = ctypes.c_uint64(ins_word)
        rec_word = ctypes.c_uint64()
        ans = self._dll.sc5511b_usb_reg_read(self._handle, reg_byte, instruct_word, ctypes.byref(rec_word))
        self._err_check(ans,'register read')
        return rec_word.value

    def get_idn(self) -> Dict[str, Optional[str]]:
        info = self._get_dev_info()

        return {'vendor':'SignalCore',
                'model':'SC5511B',
                'serial':info['product_serial_number'],
                'firmware':info['firmware_revision'],
                'hardware':info['hardware_revision'],
                'manufacture_date':'20{}-{}-{} at {}h'.format(info['man_date']['year'], 
                    info['man_date']['month'], info['man_date']['day'], info['man_date']['hour'])}


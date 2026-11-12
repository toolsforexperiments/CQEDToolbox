"""Structural typing interfaces for the shared Keysight P9374A/M5180 VNAR API.

These protocols describe the public driver API of a VNA. They
therefore work for QCoDeS parameters as well as InstrumentServer proxy
parameters, whose normal use is both 'parameter()' and 'parameter(value)'.
"""

from typing import Protocol, TypeVar, overload
import numpy as np
import numpy.typing as npt


T = TypeVar("T")

FrequencyArray = npt.NDArray[np.float64]
TraceArray = npt.NDArray[np.complex128]


class ReadOnlyParameter(Protocol[T]):
    def get(self) -> T:
        ...

    def __call__(self) -> T:
        ...


class Parameter(ReadOnlyParameter[T], Protocol[T]):
    def set(self, value: T) -> None:
        ...

    @overload
    def __call__(self) -> T:
        ...

    @overload
    def __call__(self, value: T, /) -> None:
        ...


class TraceType(Protocol):
    npts: ReadOnlyParameter[int]
    frequency: ReadOnlyParameter[FrequencyArray]
    data: ReadOnlyParameter[TraceArray]
    s_parameter: Parameter[str]


class VnaType(Protocol):
    fstart: Parameter[float]
    fstop: Parameter[float]
    fcenter: Parameter[float]
    fspan: Parameter[float]
    num_points: Parameter[int]

    rfout: Parameter[bool | int]
    power: Parameter[float]
    ifbw: Parameter[float]
    averaging: Parameter[bool | int]
    avg_num: Parameter[int]
    electrical_delay: Parameter[float]

    trigger_source: Parameter[str]
    sweep_mode: Parameter[str]
    trigger_mode: Parameter[str]

    trace_1: TraceType

    def clear_averages(self) -> None:
        ...
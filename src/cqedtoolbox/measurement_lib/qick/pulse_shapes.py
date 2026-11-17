"""Pi-pulse shape options shared by the QICK programs and the protocol parameters.

This module must not import ``qick`` so that it can be imported on machines
without QICK installed (e.g. OPX setups that use ``protocols.parameters``).
"""

PI_PULSE_SHAPES = ("gaussian", "flattop")
DEFAULT_PI_PULSE_SHAPE = "flattop"


def normalize_pi_pulse_shape(shape) -> str:
    """Return the pi-pulse shape as one of ``PI_PULSE_SHAPES``.

    ``None`` (shape not set) returns ``DEFAULT_PI_PULSE_SHAPE``. Other values are
    stripped and lower-cased before validation. Callers decide whether to warn
    about a missing shape.

    :param shape: The configured shape, or None if it is not set.
    :return: 'gaussian' or 'flattop'.
    :raises ValueError: If ``shape`` is not one of ``PI_PULSE_SHAPES``.
    """
    if shape is None:
        return DEFAULT_PI_PULSE_SHAPE
    shape = str(shape).strip().lower()
    if shape not in PI_PULSE_SHAPES:
        raise ValueError(
            f"Unknown pi pulse shape {shape!r}; expected one of {PI_PULSE_SHAPES}."
        )
    return shape

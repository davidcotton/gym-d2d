from math import log10, pow


def dB_to_linear(dB: float) -> float:
    """Convert decibel values to linear scale.

    Args:
        dB: The value in decibels.

    Returns:
        The value in a linear scale.
    """
    return pow(10, dB / 10)


def linear_to_dB(linear: float) -> float:
    """Converts linear values to decibel scale.

    Args:
        linear: The linear value to convert.

    Returns:
        The value in a decibel scale.
    """
    return 10 * log10(linear)


def dBm_to_W(dBm: float) -> float:
    return dB_to_linear(dBm) / 1000


def W_to_dBm(linear: float) -> float:
    return linear_to_dB(linear * 1000)

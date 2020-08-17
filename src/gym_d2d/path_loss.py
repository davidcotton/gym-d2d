from abc import ABC, abstractmethod
from math import log10, pi

from .device import Device


SPEED_OF_LIGHT = 299792458  # m/s


class PathLoss(ABC):
    def __init__(self, carrier_freq_GHz: float) -> None:
        super().__init__()
        self.carrier_freq_GHz: float = carrier_freq_GHz

    @abstractmethod
    def __call__(self, tx: Device, rx: Device) -> float:
        """Calculate the path loss between a transmitter and a receiver.

        :param tx: The transmitting device.
        :param rx: The receiving device.
        :return: The free space path loss in dB.
        """
        pass


def calc_fspl_constant_dB(carrier_freq_GHz: float) -> float:
    """Calculate the constant part of Free Space Path equation.

    We assume a fixed carrier frequency for all communications in our simulation.
    This means that only the distance and antenna gain parts of the FSPL equation will be changing,
    so we can memoize the freq + speed of light part to save computation.

    :param carrier_freq_GHz: The carrier frequencies in Ghz.
    :return: The free space path loss constant in dB.
    """
    return 20 * log10(carrier_freq_GHz * 1e9) + 20 * log10((4 * pi) / SPEED_OF_LIGHT)


class FreeSpacePathLoss(PathLoss):
    def __init__(self, carrier_freq_GHz: float) -> None:
        super().__init__(carrier_freq_GHz)
        self.fspl_constant_dB = calc_fspl_constant_dB(carrier_freq_GHz)

    def __call__(self, tx: Device, rx: Device) -> float:
        """Calculate the loss of signal strength in free space.

        FSPL = 20log10(d) + 20log10(f) + 20log10(4pi/c)

        Where:
            d: distance in metres
            f: the carrier frequency in Hz
            c: speed of light (m/s)

        :param tx: The transmitting device.
        :param rx: The receiving device.
        :return: The free space path loss in dB.
        """

        return 20 * log10(tx.position.distance(rx.position)) + self.fspl_constant_dB

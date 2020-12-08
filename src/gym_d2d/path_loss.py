from abc import ABC, abstractmethod
from math import log10, pi
from random import gauss

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


def pl_constant_dB(carrier_freq_GHz: float, ple: float) -> float:
    """Calculate the constant part of Log-Distance Path Loss equation.

    We assume a fixed carrier frequency for all communications in our simulation.
    This means that only the distance and antenna gain parts of the Log-Distance PL equation will be changing,
    so we can memoize the freq + speed of light part to save computation.

    :param carrier_freq_GHz: The carrier frequencies in Ghz.
    :param ple: The path loss exponent.
    :return: The log-distance path loss constant in dB.
    """
    return 10 * ple * log10(carrier_freq_GHz * 1e9) + 10 * ple * log10((4 * pi) / SPEED_OF_LIGHT)


class LogDistancePathLoss(PathLoss):
    def __init__(self, carrier_freq_GHz: float, ple: float = 2.0) -> None:
        super().__init__(carrier_freq_GHz)
        self.ple: float = ple  # path loss exponent (2.0 in free space, 3.5 in crowded env)
        self.pl_constant_dB = pl_constant_dB(carrier_freq_GHz, ple)

    def __call__(self, tx: Device, rx: Device) -> float:
        """Calculate the loss of signal strength in free space.

        LDPL = 10nlog_10(d) + 10nlog_10(f) + 10nlog_10(4pi/c)

        Where:
            d: distance in metres
            f: the carrier frequency in Hz
            c: speed of light (m/s)

        :param tx: The transmitting device.
        :param rx: The receiving device.
        :return: The log-distance path loss in dB.
        """

        return self._log_distance_path_loss(tx.position.distance(rx.position))

    def _log_distance_path_loss(self, dist_m: float) -> float:
        return 10 * self.ple * log10(dist_m) + self.pl_constant_dB


class ShadowingPathLoss(LogDistancePathLoss):
    def __init__(self, carrier_freq_GHz: float, ple: float = 2.0, d0_m: float = 100.0, chi_dB: float = 2.7) -> None:
        super().__init__(carrier_freq_GHz, ple)
        self.d0_m: float = d0_m  # shadowing close-in reference distance (metres)
        self.chi_dB: float = chi_dB  # shadowing standard deviation (dB), typically 2.7 to 3.5

    def __call__(self, tx: Device, rx: Device) -> float:
        d = tx.position.distance(rx.position)
        assert self.d0_m < d
        ldpl = self._log_distance_path_loss(self.d0_m)
        return ldpl + 10 * self.ple * log10(d / self.d0_m) + gauss(0, self.chi_dB)

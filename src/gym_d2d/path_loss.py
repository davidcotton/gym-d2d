from abc import ABC, abstractmethod
from enum import Enum
from math import log10, pi
from random import gauss

from .device import Device


SPEED_OF_LIGHT = 299792458  # m/s


class PathLoss(ABC):
    def __init__(self, carrier_freq_GHz: float) -> None:
        super().__init__()
        self.carrier_freq_GHz = float(carrier_freq_GHz)

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
    def __init__(self, carrier_freq_GHz: float, ple=2.0) -> None:
        super().__init__(carrier_freq_GHz)
        self.ple = float(ple)  # path loss exponent (2.0 in free space, 3.5 in crowded env)
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
    def __init__(self, carrier_freq_GHz: float, ple=2.0, d0_m=100.0, chi_dB=2.7) -> None:
        super().__init__(carrier_freq_GHz, ple)
        self.d0_m = float(d0_m)  # shadowing close-in reference distance (metres)
        self.chi_dB = float(chi_dB)  # shadowing standard deviation (dB), typically 2.7 to 3.5

    def __call__(self, tx: Device, rx: Device) -> float:
        d = tx.position.distance(rx.position)
        if d > self.d0_m:
            ldpl = self._log_distance_path_loss(self.d0_m)
            return ldpl + 10 * self.ple * log10(d / self.d0_m) + gauss(0, self.chi_dB)
        else:
            return self._log_distance_path_loss(d)


class AreaType(Enum):
    RURAL = 0
    SUBURBAN = 1
    URBAN = 2


class CostHataPathLoss(PathLoss):
    def __init__(self, carrier_freq_GHz: float, area_type=AreaType.SUBURBAN) -> None:
        super().__init__(carrier_freq_GHz)
        self.area_type: AreaType = area_type

    def __call__(self, tx: Device, rx: Device) -> float:
        """
        Lb = 46.3 + 33.9log_10(f) - 13.82log_10(h_tx) - a(h_rx,f) + (44.9 - 6.55log_10(h_tx)log_10(d) + C
        """
        f = self.carrier_freq_GHz * 1000  # transmission freq (MHz)
        d = tx.position.distance(rx.position) / 1000  # link distance (Km)
        h_tx = tx.antenna_height_m  # TX antenna height (m)
        h_rx = rx.antenna_height_m  # RX antenna height (m)
        a_hc = self._ms_h_correction(f, h_rx)  # MS antenna height correction factor
        # constant offset (dB), 0 for med cities & suburban, 3 for metropolitan areas
        c = 3 if self.area_type == AreaType.URBAN else 0
        pl = 46.3 + 33.9 * log10(f) - 13.82 * log10(h_tx) - a_hc + (44.9 - 6.55 * log10(h_tx)) * log10(d) + c
        return pl

    def _ms_h_correction(self, f: float, h_rx: float) -> float:
        """RX antenna height correction factor.

        :param f: Carrier freq (MHz)
        :param h_rx: Receiving station antenna height (m)
        :return: The height correction factor.
        """
        if self.area_type == AreaType.URBAN:
            if f >= 200:
                a_hc = 8.29 * (log10(1.54 * h_rx)) ** 2 - 1.1
            else:
                a_hc = 3.2 * (log10(11.75 * h_rx)) ** 2 - 4.97
        else:
            a_hc = (1.1 * log10(f) - 0.7) * h_rx - (1.56 * log10(f) - 0.8)
        return a_hc

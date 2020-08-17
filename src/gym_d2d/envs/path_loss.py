from math import log10, pi

from .device import Device


class PathLoss:
    def __init__(self, carrier_freq_GHz: float) -> None:
        super().__init__()
        self.carrier_freq_GHz: float = carrier_freq_GHz

    def __call__(self, tx: Device, rx: Device, d: float) -> float:
        """Calculate the path loss between communicating devices.

        :param tx: The transmitting device.
        :param rx: The receiving device.
        :param d: The transmission distance in metres.
        :return: The free space path loss in dB.
        """
        raise NotImplementedError


def calc_fspl_constant_dB(carrier_freq_GHz: float) -> float:
    """Calculate the constant part of Free Space Path equation.

    We assume a fixed carrier frequency for all communications in our simulation.
    This means that only the distance and antenna gain parts of the FSPL equation will be changing,
    so we can memoize the freq + speed of light part to save computation.

    :param carrier_freq_GHz: The carrier frequencies in Ghz.
    :return: The free space path loss constant in dB.
    """
    return 20 * log10(carrier_freq_GHz * 1e9) + 20 * log10((4 * pi) / 299792458)


class FreeSpacePathLoss(PathLoss):
    def __init__(self, carrier_freq_GHz: float) -> None:
        super().__init__(carrier_freq_GHz)
        self.fspl_constant_dB = calc_fspl_constant_dB(carrier_freq_GHz)

    def __call__(self, tx: Device, rx: Device, d: float) -> float:
        """Calculate the loss of signal strength in free space.

        FSPL = 20log10(d) + 20log10(f) + 20log10(4pi/c) - G_tx - G_rx

        Where:
            d: distance in metres
            f: the carrier frequency in Hz
            c: speed of light (m/s)
            G_tx: transmitting antenna gain
            G_rx: receiving antenna gain

        :param tx: The transmitting device.
        :param rx: The receiving device.
        :param d: The transmission distance in metres.
        :return: The free space path loss in dB.
        """

        return 20 * log10(d) + self.fspl_constant_dB - tx.tx_antenna_gain_dBi - rx.rx_antenna_gain_dBi

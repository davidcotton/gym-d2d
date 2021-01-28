from .conversion import dB_to_linear, dBm_to_W
from .id import Id
from .position import Position
from .utils import merge_dicts


# https://en.wikipedia.org/wiki/Johnson%E2%80%93Nyquist_noise#Noise_power_in_decibels
THERMAL_NOISE_POWER_dBm = -121.45  # 1x 180kHz LTE RB
THERMAL_NOISE_POWER_mW = dB_to_linear(THERMAL_NOISE_POWER_dBm)
THERMAL_NOISE_POWER_W = dBm_to_W(THERMAL_NOISE_POWER_dBm)

DEFAULT_DEVICE_CONFIG = {
    'num_PRB': 1,
    'num_subcarriers': 12,
    'subcarrier_spacing_kHz': 15.0,
}
DEFAULT_BASE_STATION_CONFIG = merge_dicts(dict(DEFAULT_DEVICE_CONFIG), {
    'max_tx_power_dBm': 46.0,
    # 'antenna_height_m': 10.0,
    'antenna_height_m': 23.0,
    'tx_antenna_gain_dBi': 17.5,
    'rx_antenna_gain_dBi': 17.5,
    'thermal_noise_dBm': -118.4,
    'noise_figure_dB': 2.0,
    'sinr_dB': -7.0,
    'ix_margin_dB': 2.0,  # accounts for the increase in noise from surrounding cells
    'cable_loss_dB': 2.0,
    'masthead_amplifier_gain_dB': 2.0,
})
DEFAULT_UE_CONFIG = merge_dicts(dict(DEFAULT_DEVICE_CONFIG), {
    'max_tx_power_dBm': 23.0,
    'antenna_height_m': 1.5,
    'tx_antenna_gain_dBi': 0.0,
    'rx_antenna_gain_dBi': 0.0,
    'thermal_noise_dBm': -104.5,
    'noise_figure_dB': 7.0,
    'sinr_dB': -10.0,
    'ix_margin_dB': 3.0,  # accounts for the increase in noise from surrounding cells
    'control_channel_overhead_dB': 1.0,
    'body_loss_dB': 3.0,  # loss from user's body blocking reception
})


class Device:
    def __init__(self, id_, config: dict) -> None:
        super().__init__()
        self.id = Id(id_)
        self.config: dict = config
        self.position: Position = Position(0, 0)

    def eirp_dBm(self, tx_pwr_dBm: float) -> float:
        """Calculate the Effective Isotropically Radiated Power (EIRP).

        How much power (in dBm) the device radiates when considering factors such as antenna gain.

        :param tx_pwr_dBm: The transmission power in dBm.
        :returns: The EIRP in dBm.
        """
        # - (10 * log10(self._subcarrier_quantity)) \
        return tx_pwr_dBm + self.tx_antenna_gain_dBi - self.ix_margin_dB

    def rx_signal_level_dBm(self, eirp_dBm: float, path_loss_dB: float) -> float:
        """Calculate the received signal strength.

        :param eirp_dBm: The transmitted EIRP in dBm.
        :param path_loss_dB: The path loss that has occurred in dB.
        :returns: The received signal level in dBm.
        """
        # - self._thermal_noise_dBm \
        # - self._noise_figure_dB \
        # - self._sinr_dB
        return eirp_dBm - path_loss_dB + self.rx_antenna_gain_dBi

    @property
    def rx_sensitivity_dBm(self) -> float:
        return self.rx_noise_floor_dBm + self.sinr_dB

    @property
    def rx_noise_floor_dBm(self) -> float:
        return self.noise_figure_dB + self.thermal_noise_dBm

    def set_position(self, pos: Position) -> None:
        self.position = pos

    @property
    def num_subcarriers(self) -> int:
        return int(self.config['num_subcarriers'])

    @property
    def subcarrier_spacing_kHz(self) -> int:
        return int(self.config['subcarrier_spacing_kHz'])

    @property
    def rb_bandwidth_kHz(self) -> int:
        return self.num_subcarriers * self.subcarrier_spacing_kHz

    @property
    def max_tx_power_dBm(self) -> int:
        return self.config['max_tx_power_dBm']

    @property
    def antenna_height_m(self) -> float:
        return self.config['antenna_height_m']

    @property
    def tx_antenna_gain_dBi(self) -> float:
        return self.config['tx_antenna_gain_dBi']

    @property
    def rx_antenna_gain_dBi(self) -> float:
        return self.config['rx_antenna_gain_dBi']

    @property
    def noise_figure_dB(self) -> float:
        return self.config['noise_figure_dB']

    @property
    def thermal_noise_dBm(self) -> float:
        return self.config['thermal_noise_dBm']

    @property
    def sinr_dB(self) -> float:
        return self.config['sinr_dB']

    @property
    def ix_margin_dB(self) -> float:
        return self.config['ix_margin_dB']


class BaseStation(Device):
    def __init__(self, id_, config: dict = None) -> None:
        super().__init__(id_, merge_dicts(dict(DEFAULT_BASE_STATION_CONFIG), config or {}))

    def eirp_dBm(self, tx_pwr_dBm: float) -> float:
        return super().eirp_dBm(tx_pwr_dBm) - self.cable_loss_dB + self.masthead_amplifier_gain_dB

    def rx_signal_level_dBm(self, eirp_dBm: float, path_loss_dB: float) -> float:
        return super().rx_signal_level_dBm(eirp_dBm, path_loss_dB) \
               - self.cable_loss_dB \
               + self.masthead_amplifier_gain_dB

    @property
    def cable_loss_dB(self) -> float:
        return self.config['cable_loss_dB']

    @property
    def masthead_amplifier_gain_dB(self) -> float:
        return self.config['masthead_amplifier_gain_dB']

    def __repr__(self) -> str:
        return f'<BS:{self.id}>'


class UserEquipment(Device):
    def __init__(self, id_, config: dict = None) -> None:
        super().__init__(id_, merge_dicts(dict(DEFAULT_UE_CONFIG), config or {}))

    def eirp_dBm(self, tx_pwr_dBm: float) -> float:
        return super().eirp_dBm(tx_pwr_dBm) - self.body_loss_dB

    def rx_signal_level_dBm(self, eirp_dBm: float, path_loss_dB: float) -> float:
        return super().rx_signal_level_dBm(eirp_dBm, path_loss_dB) - self.body_loss_dB

    @property
    def control_channel_overhead_dB(self) -> float:
        return self.config['control_channel_overhead_dB']

    @property
    def body_loss_dB(self) -> float:
        return self.config['body_loss_dB']

    def __repr__(self) -> str:
        return f'<UE:{self.id}>'

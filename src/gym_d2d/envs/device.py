from .conversion import dB_to_linear, dBm_to_W, linear_to_dB
from .id import Id
from .utils import merge_dicts


# https://en.wikipedia.org/wiki/Johnson%E2%80%93Nyquist_noise#Noise_power_in_decibels
THERMAL_NOISE_POWER_dBm = -121.45  # 1x 180kHz LTE RB
THERMAL_NOISE_POWER_mW = dB_to_linear(THERMAL_NOISE_POWER_dBm)
THERMAL_NOISE_POWER_W = dBm_to_W(THERMAL_NOISE_POWER_dBm)

DEFAULT_DEVICE_CONFIG = {
    'num_PRB': 1,
    'carrier_freq_GHz': 2.1,
    'subcarrier_quantity': 12,
    'subcarrier_spacing_kHz': 30.0,
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
    'body_loss_dB': 3.0,
})


class Device:
    def __init__(self, id_, config: dict) -> None:
        super().__init__()
        self.id = Id(id_)
        self.config: dict = config

    @property
    def max_tx_power_dBm(self) -> int:
        return self.config['max_tx_power_dBm']

    @property
    def carrier_freq_GHz(self) -> float:
        return self.config['carrier_freq_GHz']

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
    def rx_noise_floor_dBm(self) -> float:
        return self.noise_figure_dB + self.thermal_noise_dBm

    @property
    def sinr_dB(self) -> float:
        return self.config['sinr_dB']

    @property
    def ix_margin_dB(self) -> float:
        return self.config['ix_margin_dB']


class BaseStation(Device):
    def __init__(self, id_, config: dict = None) -> None:
        super().__init__(id_, merge_dicts(dict(DEFAULT_BASE_STATION_CONFIG), config or {}))


class UserEquipment(Device):
    def __init__(self, id_, config: dict = None) -> None:
        super().__init__(id_, merge_dicts(dict(DEFAULT_UE_CONFIG), config or {}))

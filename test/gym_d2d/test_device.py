import random

from pytest import approx

from gym_d2d.device import BaseStation, UserEquipment, DEFAULT_UE_CONFIG, DEFAULT_BASE_STATION_CONFIG
from gym_d2d.id import Id
from gym_d2d.position import Position


class TestDevice:
    def test_init_has_id(self):
        ue = UserEquipment('ue')
        bs = BaseStation('bs')
        assert ue.id == Id('ue')
        assert bs.id == Id('bs')

    def test_init_creates_default_position(self):
        ue = UserEquipment('ue')
        assert isinstance(ue.position, Position)
        assert ue.position.as_tuple() == (0, 0)

    def test_set_position(self):
        ue = UserEquipment('ue')
        ue.set_position(Position(-123.45, 78.9))
        assert ue.position.as_tuple() == (-123.45, 78.9)

    def test_init_has_default_config(self):
        ue = UserEquipment('ue')
        assert ue.max_tx_power_dBm == DEFAULT_UE_CONFIG['max_tx_power_dBm']
        assert ue.antenna_height_m == DEFAULT_UE_CONFIG['antenna_height_m']
        assert ue.tx_antenna_gain_dBi == DEFAULT_UE_CONFIG['tx_antenna_gain_dBi']
        assert ue.rx_antenna_gain_dBi == DEFAULT_UE_CONFIG['rx_antenna_gain_dBi']
        assert ue.thermal_noise_dBm == DEFAULT_UE_CONFIG['thermal_noise_dBm']
        assert ue.noise_figure_dB == DEFAULT_UE_CONFIG['noise_figure_dB']
        assert ue.sinr_dB == DEFAULT_UE_CONFIG['sinr_dB']
        assert ue.ix_margin_dB == DEFAULT_UE_CONFIG['ix_margin_dB']
        assert ue.control_channel_overhead_dB == DEFAULT_UE_CONFIG['control_channel_overhead_dB']
        assert ue.body_loss_dB == DEFAULT_UE_CONFIG['body_loss_dB']

        bs = BaseStation('bs')
        assert bs.max_tx_power_dBm == DEFAULT_BASE_STATION_CONFIG['max_tx_power_dBm']
        assert bs.antenna_height_m == DEFAULT_BASE_STATION_CONFIG['antenna_height_m']
        assert bs.tx_antenna_gain_dBi == DEFAULT_BASE_STATION_CONFIG['tx_antenna_gain_dBi']
        assert bs.rx_antenna_gain_dBi == DEFAULT_BASE_STATION_CONFIG['rx_antenna_gain_dBi']
        assert bs.thermal_noise_dBm == DEFAULT_BASE_STATION_CONFIG['thermal_noise_dBm']
        assert bs.noise_figure_dB == DEFAULT_BASE_STATION_CONFIG['noise_figure_dB']
        assert bs.sinr_dB == DEFAULT_BASE_STATION_CONFIG['sinr_dB']
        assert bs.ix_margin_dB == DEFAULT_BASE_STATION_CONFIG['ix_margin_dB']
        assert bs.cable_loss_dB == DEFAULT_BASE_STATION_CONFIG['cable_loss_dB']
        assert bs.masthead_amplifier_gain_dB == DEFAULT_BASE_STATION_CONFIG['masthead_amplifier_gain_dB']

    def test_init_merges_override_config(self):
        ue_pwr = random.uniform(0, 23)
        ue_ant_height = random.uniform(0, 200)
        ue = UserEquipment('ue', {
            'antenna_height_m': ue_ant_height,
            'max_tx_power_dBm': ue_pwr,
        })
        assert ue.max_tx_power_dBm == ue_pwr
        assert ue.antenna_height_m == ue_ant_height

        bs_pwr = random.uniform(0, 50)
        bs_tx_ant_gain = random.uniform(0, 50)
        bs = BaseStation('bs', {
            'max_tx_power_dBm': bs_pwr,
            'tx_antenna_gain_dBi': bs_tx_ant_gain,
        })
        assert bs.max_tx_power_dBm == bs_pwr
        assert bs.tx_antenna_gain_dBi == bs_tx_ant_gain

    def test_eirp_dBm(self):
        ue = UserEquipment('ue')
        assert ue.eirp_dBm(12) == approx(
            12
            + DEFAULT_UE_CONFIG['tx_antenna_gain_dBi']
            - DEFAULT_UE_CONFIG['ix_margin_dB']
            - DEFAULT_UE_CONFIG['body_loss_dB'])

        bs = BaseStation('bs')
        assert bs.eirp_dBm(46) == approx(
            46
            + DEFAULT_BASE_STATION_CONFIG['tx_antenna_gain_dBi']
            - DEFAULT_BASE_STATION_CONFIG['ix_margin_dB']
            - DEFAULT_BASE_STATION_CONFIG['cable_loss_dB']
            + DEFAULT_BASE_STATION_CONFIG['masthead_amplifier_gain_dB'])

    def test_rx_sensitivity_dBm(self):
        ue = UserEquipment('ue')
        bs = BaseStation('bs')
        assert ue.rx_sensitivity_dBm == approx(ue.rx_noise_floor_dBm + DEFAULT_UE_CONFIG['sinr_dB'])
        assert bs.rx_sensitivity_dBm == approx(bs.rx_noise_floor_dBm + DEFAULT_BASE_STATION_CONFIG['sinr_dB'])

    def test_rx_noise_floor_dBm(self):
        ue = UserEquipment('ue')
        bs = BaseStation('bs')
        assert ue.rx_noise_floor_dBm == approx(
            DEFAULT_UE_CONFIG['noise_figure_dB'] + DEFAULT_UE_CONFIG['thermal_noise_dBm'])
        assert bs.rx_noise_floor_dBm == approx(
            DEFAULT_BASE_STATION_CONFIG['noise_figure_dB'] + DEFAULT_BASE_STATION_CONFIG['thermal_noise_dBm'])

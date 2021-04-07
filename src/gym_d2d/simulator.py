from collections import defaultdict
from math import log2
from typing import Dict, Tuple

from gym_d2d.link_type import LinkType
from gym_d2d.utils import SurjectiveDict, BidirectionalDict

from .actions import Actions
from .conversion import dB_to_linear, linear_to_dB
from .device import BaseStation, UserEquipment
from .devices import Devices, DeviceMap
from .envs.env_config import EnvConfig
from .id import Id
from .path_loss import PathLoss
from .position import get_random_position_nearby, get_random_position, Position
from .traffic_model import TrafficModel


BASE_STATION_ID = Id('mbs')


def create_devices(config: EnvConfig) -> Devices:
    """Initialise devices: BSs, CUEs & DUE pairs as per the env config.

    :param config: The environment's configuration.
    :returns: A dataclass containing BSs, CUEs, and DUE pairs.
    """

    base_cfg = {
        'num_subcarriers': config.num_subcarriers,
        'subcarrier_spacing_kHz': config.subcarrier_spacing_kHz,
    }
    default_cue_cfg = {**base_cfg, **{'max_tx_power_dBm': config.cue_max_tx_power_dBm}}
    default_due_cfg = {**base_cfg, **{'max_tx_power_dBm': config.due_max_tx_power_dBm}}
    get_config = lambda id_, default_cfg: config.devices.get(id_, {}).get('config', default_cfg)

    devices = {
        BASE_STATION_ID: BaseStation(BASE_STATION_ID, get_config(BASE_STATION_ID, base_cfg)),
    }
    for i in range(config.num_cues):
        cue_id = Id(f'cue{i:02d}')
        devices[cue_id] = UserEquipment(cue_id, get_config(cue_id, default_cue_cfg))
    for i in range(0, config.num_due_pairs * 2):
        due_id = Id(f'due{i:02d}')
        devices[due_id] = UserEquipment(due_id, get_config(due_id, default_due_cfg))

    return Devices(**devices)


class Simulator:
    def __init__(self, env_config: dict) -> None:
        super().__init__()
        self.config = EnvConfig(**env_config)
        self.devices: Devices = create_devices(self.config)
        device_map = self.config.device_map
        # self.device_map = DeviceMap(device_map)
        # self.device_map = DeviceMap()
        # self.device_map = SurjectiveDict()
        self.txs = defaultdict(list)
        self.rxs = defaultdict(list)
        self.linktype_map = BidirectionalDict()
        for tx_id, rx_id, linktype in device_map:
            # self.device_map[tx_id] = rx_id
            self.txs[tx_id].append(rx_id)
            self.rxs[rx_id].append(tx_id)
            self.linktype_map[(tx_id, rx_id)] = linktype
        self.traffic_model: TrafficModel = self.config.traffic_model(self.config.num_rbs)
        self.path_loss: PathLoss = self.config.path_loss_model(self.config.carrier_freq_GHz)
        self.default_sinr = -100.0
        self.default_snr = -100.0

    def reset(self) -> None:
        self._reposition_devices()

    def _reposition_devices(self) -> None:
        for device in self.devices.values():
            if device.id == BASE_STATION_ID:
                pos = Position(0, 0)  # assume MBS fixed at (0,0) and everything else builds around it
            elif device.id in self.config.devices:
                pos = Position(*self.config.devices[device.id]['position'])
            elif device.id in self.txs:
                pos = get_random_position(self.config.cell_radius_m)
            elif device.id in self.rxs:
                tx_ids = self.rxs[device.id]
                tx_id = tx_ids[0]  # @todo need to fix this
                if self.linktype_map[(tx_id, device.id)] == LinkType.SIDELINK:
                    tx_pos = self.devices[tx_id].position
                    pos = get_random_position_nearby(self.config.cell_radius_m, tx_pos, self.config.d2d_radius_m)
                else:
                    pos = get_random_position(self.config.cell_radius_m)
            else:
                raise ValueError(f'Invalid configuration for device "{device.id}".')
            device.set_position(pos)

    def step(self, actions: Actions) -> dict:
        # self.channels = self.traffic_model.get_traffic(self.devices)
        sinrs_db = self._calculate_sinrs(actions)
        capacities = self._calculate_network_capacity(sinrs_db)

        return {
            'sinrs_db': sinrs_db,
            'snrs_db': self._calculate_snrs(actions),
            'rate_bps': self._calculate_rates(sinrs_db),
            'capacity_mbps': capacities,
        }

    def _calculate_sinrs(self, actions: Actions) -> Dict[Tuple[Id, Id], float]:
        sinrs_db = {}
        for id_pair in self.linktype_map.keys():
            if id_pair in actions:
                action = actions[id_pair]
                tx, rx = action.tx, action.rx
                rx_pwr_dBm = rx.rx_signal_level_dBm(tx.eirp_dBm(action.tx_pwr_dBm), self.path_loss(tx, rx))

                ix_actions = actions.get_actions_by_rb(action.rb).difference({action})
                sum_ix_pwr_mW = 0.0
                for ix_action in ix_actions:
                    ix_tx = ix_action.tx
                    ix_eirp_dBm = ix_tx.eirp_dBm(ix_action.tx_pwr_dBm)
                    ix_path_loss_dB = self.path_loss(ix_tx, rx)
                    sum_ix_pwr_mW += dB_to_linear(ix_eirp_dBm - ix_path_loss_dB)

                # noise_mW = dB_to_linear(rx.thermal_noise_dBm)  # @todo this can be memoized
                # ix_and_noise_mW = sum_ix_pwr_mW + noise_mW
                # sinrs_db[(tx_id, rx_id)] = rx_pwr_dBm - linear_to_dB(ix_and_noise_mW)
                # sinrs_db[(tx_id, rx_id)] = \
                #     float(rx_pwr_dBm - linear_to_dB(sum_ix_pwr_mW + dB_to_linear(rx.thermal_noise_dBm)))
                sinr = float(rx_pwr_dBm - linear_to_dB(sum_ix_pwr_mW + dB_to_linear(rx.thermal_noise_dBm)))
            else:
                sinr = self.default_sinr
            sinrs_db[id_pair] = sinr
        return sinrs_db

    def _calculate_snrs(self, actions: Actions) -> Dict[Tuple[Id, Id], float]:
        SNRs_dB = {}
        for id_pair in self.linktype_map.keys():
            if id_pair in actions:
                action = actions[id_pair]
                tx, rx = action.tx, action.rx
                rx_pwr_dBm = rx.rx_signal_level_dBm(tx.eirp_dBm(action.tx_pwr_dBm), self.path_loss(tx, rx))
                snr = float(rx_pwr_dBm - rx.thermal_noise_dBm)
            else:
                snr = self.default_snr
            SNRs_dB[id_pair] = snr

        return SNRs_dB

    def _calculate_rates(self, sinrs_db: Dict[Tuple[Id, Id], float]) -> Dict[Tuple[Id, Id], float]:
        rates_bps = {}
        for (tx_id, rx_id), sinr_db in sinrs_db.items():
            _, rx = self.devices[tx_id], self.devices[rx_id]
            # max_path_loss_dB = rx.max_path_loss_dB(tx.eirp_dBm())
            if sinr_db > rx.rx_sensitivity_dBm:
                rates_bps[(tx_id, rx_id)] = float(log2(1 + dB_to_linear(sinr_db)))
            else:
                rates_bps[(tx_id, rx_id)] = 0.0
        return rates_bps

    def _calculate_throughput_lte(self, sinrs_db: Dict[Tuple[Id, Id], float]) -> Dict[Tuple[Id, Id], float]:
        capacities = {}
        num_rbs = 100  # 100RBs @ 20MHz channel bandwidth
        num_re = 12 * 7 * 2  # num_subcarriers * num_symbols (short CP) * num_slots/subframe
        for (tx_id, rx_id), sinr_dB in sinrs_db.items():
            _, rx = self.devices[tx_id], self.devices[rx_id]
            if sinr_dB > rx.rx_sensitivity_dBm:
                num_bits = 6  # 64QAM
                capacity_b_per_ms = num_rbs * num_re * num_bits
                capacity_mbps = capacity_b_per_ms / 1000
                capacities[(tx_id, rx_id)] = capacity_mbps
            else:
                capacities[(tx_id, rx_id)] = 0
        return capacities

    def _calculate_network_capacity(self, sinrs_db: Dict[Tuple[Id, Id], float]) -> Dict[Tuple[Id, Id], float]:
        capacities_mbps = {}
        for (tx_id, rx_id), sinr_db in sinrs_db.items():
            tx, rx = self.devices[tx_id], self.devices[rx_id]
            # max_path_loss_dB = rx.max_path_loss_dB(tx.eirp_dBm())
            if sinr_db > rx.rx_sensitivity_dBm:
                b = tx.rb_bandwidth_kHz * 1000
                capacities_mbps[(tx_id, rx_id)] = float(1e-6 * b * log2(1 + dB_to_linear(sinr_db)))
            else:
                capacities_mbps[(tx_id, rx_id)] = 0.0
        return capacities_mbps

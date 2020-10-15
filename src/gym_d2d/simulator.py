from collections import defaultdict
from math import log2
from typing import Dict, Tuple

from .action import Action
from .channel import Channel
from .conversion import dB_to_linear, linear_to_dB
from .device import Device
from .id import Id
from .path_loss import PathLoss
from .traffic_model import TrafficModel


class D2DSimulator:
    def __init__(self, devices: Dict[Id, Device], traffic_model: TrafficModel, path_loss: PathLoss) -> None:
        super().__init__()
        self.devices: Dict[Id, Device] = devices
        self.traffic_model: TrafficModel = traffic_model
        self.path_loss: PathLoss = path_loss
        self.channels: Dict[Tuple[Id, Id], Channel] = {}

    def reset(self):
        self.channels.clear()

    def step(self, actions: Dict[Id, Action]) -> dict:
        self._generate_traffic(actions)
        sinrs_db = self._calculate_sinrs()
        capacities = self._calculate_network_capacity(sinrs_db)

        return {
            'sinrs_db': sinrs_db,
            # 'snrs_db': self._calculate_snrs(),
            'rate_bps': self._calculate_rates(sinrs_db),
            'capacity_mbps': capacities,
        }

    def _generate_traffic(self, actions: Dict[Id, Action]) -> None:
        # automated traffic
        self.channels = self.traffic_model.get_traffic()
        # supplied actions
        for action in actions.values():
            tx, rx = self.devices[action.tx_id], self.devices[action.rx_id]
            self.channels[(tx.id, rx.id)] = Channel(tx, rx, action.mode, action.rb, action.tx_pwr_dBm)

    def _calculate_sinrs(self) -> Dict[Tuple[Id, Id], float]:
        # group channels by RB
        rbs = defaultdict(set)
        for channel in self.channels.values():
            rbs[channel.rb].add(channel)

        sinrs_db = {}
        for (tx_id, rx_id), channel in self.channels.items():
            tx, rx = channel.tx, channel.rx
            rx_pwr_dBm = rx.rx_signal_level_dBm(tx.eirp_dBm(channel.tx_pwr_dBm), self.path_loss(tx, rx))

            ix_channels = rbs[channel.rb].difference({channel})
            sum_ix_pwr_mW = 0.0
            for ix_channel in ix_channels:
                ix_tx = ix_channel.tx
                ix_eirp_dBm = ix_tx.eirp_dBm(ix_channel.tx_pwr_dBm)
                ix_path_loss_dB = self.path_loss(ix_tx, rx)
                sum_ix_pwr_mW += dB_to_linear(ix_eirp_dBm - ix_path_loss_dB)

            # noise_mW = dB_to_linear(rx.thermal_noise_dBm)  # @todo this can be memoized
            # ix_and_noise_mW = sum_ix_pwr_mW + noise_mW
            # sinrs_db[(tx_id, rx_id)] = rx_pwr_dBm - linear_to_dB(ix_and_noise_mW)
            sinrs_db[(tx_id, rx_id)] = float(rx_pwr_dBm - linear_to_dB(sum_ix_pwr_mW + dB_to_linear(rx.thermal_noise_dBm)))
        return sinrs_db

    def _calculate_snrs(self) -> Dict[Tuple[Id, Id], float]:
        # group channels by RB
        rbs = defaultdict(set)
        for channel in self.channels.values():
            rbs[channel.rb].add(channel)

        SNRs_dB = {}
        for ids, channel in self.channels.items():
            tx, rx = channel.tx, channel.rx
            rx_pwr_dBm = rx.rx_signal_level_dBm(tx.eirp_dBm(channel.tx_pwr_dBm), self.path_loss(tx, rx))
            SNRs_dB[ids] = float(rx_pwr_dBm - rx.thermal_noise_dBm)
        return SNRs_dB

    def _calculate_rates(self, sinrs_db: Dict[Tuple[Id, Id], float]) -> Dict[Tuple[Id, Id], float]:
        rates_bps = {}
        for (tx_id, rx_id), sinr_db in sinrs_db.items():
            tx, rx = self.devices[tx_id], self.devices[rx_id]
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
            tx, rx = self.devices[tx_id], self.devices[rx_id]
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

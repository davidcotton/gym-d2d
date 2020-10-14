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
        SINRs_dB = self._calculate_SINRs()
        capacities = self._calculate_network_capacity(SINRs_dB)

        return {
            'SINRs_dB': SINRs_dB,
            'SNRs_dB': self._calculate_SNRs(),
            'sum_rate_bps': self._calculate_sum_rate(SINRs_dB),
            'capacity_Mbps': capacities,
        }

    def _generate_traffic(self, actions: Dict[Id, Action]) -> None:
        # automated traffic
        self.channels = self.traffic_model.get_traffic()
        # supplied actions
        for action in actions.values():
            tx, rx = self.devices[action.tx_id], self.devices[action.rx_id]
            self.channels[(tx.id, rx.id)] = Channel(tx, rx, action.mode, action.rb, action.tx_pwr_dBm)

    def _calculate_SINRs(self) -> Dict[Tuple[Id, Id], float]:
        # group channels by RB
        rbs = defaultdict(set)
        for channel in self.channels.values():
            rbs[channel.rb].add(channel)

        SINRs_dB = {}
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
            # SINRs_dB[(tx_id, rx_id)] = rx_pwr_dBm - linear_to_dB(ix_and_noise_mW)
            SINRs_dB[(tx_id, rx_id)] = rx_pwr_dBm - linear_to_dB(sum_ix_pwr_mW + dB_to_linear(rx.thermal_noise_dBm))
        return SINRs_dB

    def _calculate_SNRs(self) -> Dict[Tuple[Id, Id], float]:
        # group channels by RB
        rbs = defaultdict(set)
        for channel in self.channels.values():
            rbs[channel.rb].add(channel)

        SNRs_dB = {}
        for ids, channel in self.channels.items():
            tx, rx = channel.tx, channel.rx
            rx_pwr_dBm = rx.rx_signal_level_dBm(tx.eirp_dBm(channel.tx_pwr_dBm), self.path_loss(tx, rx))
            SNRs_dB[ids] = rx_pwr_dBm - rx.thermal_noise_dBm
        return SNRs_dB

    def _calculate_sum_rate(self, SINRs_dB: Dict[Tuple[Id, Id], float]) -> Dict[Tuple[Id, Id], float]:
        sum_rate_bps = {}
        for (tx_id, rx_id), SINR_dB in SINRs_dB.items():
            tx, rx = self.devices[tx_id], self.devices[rx_id]
            # max_path_loss_dB = rx.max_path_loss_dB(tx.eirp_dBm())
            if SINR_dB > rx.rx_sensitivity_dBm:
                sum_rate_bps[(tx_id, rx_id)] = log2(1 + dB_to_linear(SINR_dB))
            else:
                sum_rate_bps[(tx_id, rx_id)] = 0
        return sum_rate_bps

    def _calculate_throughput_lte(self, sinrs: Dict[Tuple[Id, Id], float]) -> Dict[Tuple[Id, Id], float]:
        capacities = {}
        num_rbs = 100  # 100RBs @ 20MHz channel bandwidth
        num_re = 12 * 7 * 2  # num_subcarriers * num_symbols (short CP) * num_slots/subframe
        for (tx_id, rx_id), sinr_dB in sinrs.items():
            tx, rx = self.devices[tx_id], self.devices[rx_id]
            if sinr_dB > rx.rx_sensitivity_dBm:
                num_bits = 6  # 64QAM
                capacity_b_per_ms = num_rbs * num_re * num_bits
                capacity_Mbps = capacity_b_per_ms / 1000
                capacities[(tx_id, rx_id)] = capacity_Mbps
            else:
                capacities[(tx_id, rx_id)] = 0
        return capacities

    def _calculate_network_capacity(self, SINRs_dB: Dict[Tuple[Id, Id], float]) -> Dict[Tuple[Id, Id], float]:
        capacities_Mbps = {}
        for (tx_id, rx_id), SINR_dB in SINRs_dB.items():
            tx, rx = self.devices[tx_id], self.devices[rx_id]
            # max_path_loss_dB = rx.max_path_loss_dB(tx.eirp_dBm())
            if SINR_dB > rx.rx_sensitivity_dBm:
                B = tx.rb_bandwidth_kHz * 1000
                capacities_Mbps[(tx_id, rx_id)] = 1e-6 * B * log2(1 + dB_to_linear(SINR_dB))
            else:
                capacities_Mbps[(tx_id, rx_id)] = 0
        return capacities_Mbps

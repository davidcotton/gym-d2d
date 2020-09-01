from collections import defaultdict
from math import log2
from typing import Dict, Tuple, Union

from .action import Action
from .channel import Channel
from .conversion import dB_to_linear, linear_to_dB
from .device import Device, BaseStation, UserEquipment
from .id import Id
from .path_loss import PathLoss
from .traffic_model import TrafficModel


class D2DSimulator:
    def __init__(self, path_loss: PathLoss, channel_bandwidth_MHz: float, num_rbs: int) -> None:
        super().__init__()
        self.path_loss: PathLoss = path_loss
        self.channel_bandwidth_MHz: float = channel_bandwidth_MHz
        self.num_rbs: int = num_rbs
        self.devices: Dict[Id, Device] = {}
        self.base_stations: Dict[Id, BaseStation] = {}
        self.ues: Dict[Id, UserEquipment] = {}
        self.channels: Dict[Tuple[Id, Id], Channel] = {}
        self.traffic_model: TrafficModel = None

    def reset(self):
        self.channels.clear()

    def step(self, actions: Dict[Id, Action]) -> dict:
        self._generate_traffic(actions)
        sinrs_dB, snrs_dB = self._calculate_sinrs()
        capacities, sum_rate_bps = self._calculate_network_capacity(sinrs_dB)

        return {
            'SINRs_dB': sinrs_dB,
            'SNRs_dB': snrs_dB,
            'capacity_Mbps': capacities,
            'sum_rate_bps': sum_rate_bps,
        }

    def _generate_traffic(self, actions: Dict[Id, Action]) -> None:
        # automated traffic
        self.channels = self.traffic_model.get_traffic()
        # supplied actions
        for action in actions.values():
            tx, rx = self.devices[action.tx_id], self.devices[action.rx_id]
            self.channels[(tx.id, rx.id)] = Channel(tx, rx, action.mode, action.rb, action.tx_pwr_dBm)

    # def _calculate_sinrs(self) -> Dict[Tuple[Id, Id], float]:
    def _calculate_sinrs(self):
        # group channels by RB
        rbs = defaultdict(set)
        for channel in self.channels.values():
            rbs[channel.rb].add(channel)

        sinrs_dB = {}
        snrs_dB = {}
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
            # sinrs_dB[(tx_id, rx_id)] = rx_pwr_dBm - linear_to_dB(ix_and_noise_mW)
            sinrs_dB[(tx_id, rx_id)] = rx_pwr_dBm - linear_to_dB(sum_ix_pwr_mW + dB_to_linear(rx.thermal_noise_dBm))
            snrs_dB[(tx_id, rx_id)] = rx_pwr_dBm - rx.thermal_noise_dBm
        return sinrs_dB, snrs_dB

    # def _calculate_network_capacity(self, sinrs_dB: Dict[Tuple[Id, Id], float]) -> Dict[Tuple[Id, Id], float]:
    def _calculate_network_capacity(self, sinrs_dB: Dict[Tuple[Id, Id], float]):
        capacities_Mbps = {}
        sum_rate_bps = {}
        for (tx_id, rx_id), sinr_dB in sinrs_dB.items():
            tx, rx = self.devices[tx_id], self.devices[rx_id]
            # max_path_loss_dB = rx.max_path_loss_dB(tx.eirp_dBm())
            if sinr_dB > rx.rx_sensitivity_dBm:
                B = tx.rb_bandwidth_kHz * 1000
                capacities_Mbps[(tx_id, rx_id)] = 1e-6 * B * log2(1 + dB_to_linear(sinr_dB))
                sum_rate_bps[(tx_id, rx_id)] = log2(1 + dB_to_linear(sinr_dB))
            else:
                capacities_Mbps[(tx_id, rx_id)] = 0
                sum_rate_bps[(tx_id, rx_id)] = 0
        return capacities_Mbps, sum_rate_bps

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

    def add_base_station(self, bs_id: Union[Id, str], config: dict) -> BaseStation:
        bs_id = bs_id if isinstance(bs_id, Id) else Id(bs_id)
        bs = BaseStation(bs_id, config)
        self.base_stations[bs_id] = bs
        self.devices[bs_id] = bs
        return bs

    def add_ue(self, ue_id: Union[Id, str], config: dict) -> UserEquipment:
        ue_id = ue_id if isinstance(ue_id, Id) else Id(ue_id)
        ue = UserEquipment(ue_id, config)
        self.ues[ue_id] = ue
        self.devices[ue_id] = ue
        return ue

    def add_traffic_model(self, traffic_model: TrafficModel):
        self.traffic_model = traffic_model

from collections import defaultdict
from math import log2
from typing import Any, Dict, Tuple, Union

from .action import Action
from .channel import Channel
from .conversion import dB_to_linear, linear_to_dB
from .device import Device, BaseStation, UserEquipment
from .id import Id
from .path_loss import PathLoss
from .traffic_model import TrafficModel


class D2DSimulator:
    def __init__(self, path_loss: PathLoss, num_rbs: int) -> None:
        super().__init__()
        self.path_loss: PathLoss = path_loss
        self.num_rbs: int = num_rbs
        self.devices: Dict[Id, Device] = {}
        self.base_stations: Dict[Id, BaseStation] = {}
        self.ues: Dict[Id, UserEquipment] = {}
        self.channels: Dict[Tuple[Id, Id], Channel] = {}
        self.traffic_models: Dict[Tuple[Any], TrafficModel] = {}

    def reset(self):
        self.channels.clear()

    def step(self, actions: Dict[Id, Action]) -> dict:
        self._generate_traffic(actions)
        sinrs = self._calculate_sinrs()
        capacities = self._calculate_network_capacity(sinrs)

        return {
            'sinrs': sinrs,
            'capacity': capacities,
        }

    def _generate_traffic(self, actions: Dict[Id, Action]) -> None:
        # automated traffic
        rb = 0
        for ids, traffic_model in self.traffic_models.items():
            channel = traffic_model.get_channel(rb)
            self.channels[(channel.tx.id, channel.rx.id)] = channel
            rb = (rb + 1) % self.num_rbs
        # supplied actions
        for due_id, action in actions.items():
            # self.channels[(action.tx_id, action.rx_id)] = Channel.from_action(action)
            tx, rx = self.devices[action.tx_id], self.devices[action.rx_id]
            self.channels[(tx.id, rx.id)] = Channel(tx, rx, action.mode, action.rb, action.tx_pwr)

    def _calculate_sinrs(self) -> Dict[Tuple[Id, Id], float]:
        # group by RB
        rbs = defaultdict(set)
        for ids, channel in self.channels.items():
            rbs[channel.rb].add(ids)

        sinrs_dB = {}
        for (tx_id, rx_id), channel in self.channels.items():
            tx, rx = self.devices[tx_id], self.devices[rx_id]
            tx_eirp_dBm = tx.eirp_dBm(channel.tx_pwr_dBm)
            path_loss_dB = self.path_loss(tx, rx)
            rx_pwr_dBm = rx.rx_signal_level_dBm(tx_eirp_dBm, path_loss_dB)

            ix_channels = rbs[channel.rb].difference({(tx_id, rx_id)})
            sum_ix_pwr_mW = 0
            for ix_tx_id, ix_rx_id in ix_channels:
                ix_tx = self.devices[ix_tx_id]
                ix_channel = self.channels[(ix_tx_id, ix_rx_id)]
                ix_eirp_dBm = ix_tx.eirp_dBm(ix_channel.tx_pwr_dBm)
                ix_path_loss_dB = self.path_loss(ix_tx, rx)
                sum_ix_pwr_mW += dB_to_linear(ix_eirp_dBm - ix_path_loss_dB)

            # noise_mW = dB_to_linear(rx.thermal_noise_dBm)  # @todo this can be memoized
            # ix_and_noise_mW = sum_ix_pwr_mW + noise_mW
            # sinrs_dB[(tx_id, rx_id)] = rx_pwr_dBm - linear_to_dB(ix_and_noise_mW)
            sinrs_dB[(tx_id, rx_id)] = rx_pwr_dBm - linear_to_dB(sum_ix_pwr_mW + dB_to_linear(rx.thermal_noise_dBm))
        return sinrs_dB

    def _calculate_network_capacity(self, sinrs: Dict[Tuple[Id, Id], float]) -> Dict[Tuple[Id, Id], float]:
        capacities = {}
        for (tx_id, rx_id), sinr_dB in sinrs.items():
            tx, rx = self.devices[tx_id], self.devices[rx_id]
            # max_path_loss_dB = rx.max_path_loss_dB(tx.eirp_dBm())
            if sinr_dB > rx.rx_sensitivity_dBm:
                capacities[(tx_id, rx_id)] = tx.rb_bandwidth_kHz * log2(1 + dB_to_linear(sinr_dB))
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
        ids = tuple(d.id for d in traffic_model.devices)
        self.traffic_models[ids] = traffic_model

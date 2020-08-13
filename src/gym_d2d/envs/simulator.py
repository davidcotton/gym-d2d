from collections import defaultdict
from typing import Any, Dict, Tuple

from .action import Action
from .channel import Channel
from .conversion import dB_to_linear, dBm_to_W, linear_to_dB
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
        # create automated actions
        rb = 0
        for ids, traffic_model in self.traffic_models.items():
            channel = traffic_model.get_channel(rb)
            self.channels[(channel.tx.id, channel.rx.id)] = channel
            rb = (rb + 1) % self.num_rbs
        # execute supplied actions
        for due_id, action in actions.items():
            # self.channels[(action.tx_id, action.rx_id)] = Channel.from_action(action)
            tx, rx = self.devices[action.tx_id], self.devices[action.rx_id]
            self.channels[(tx.id, rx.id)] = Channel(tx, rx, action.mode, action.rb, action.tx_pwr)

        # CALCULATE SINR
        # group by RB
        # rbs = defaultdict(list)
        rbs = defaultdict(set)
        for ids, channel in self.channels.items():
            # rbs[channel.rb].append(ids)
            rbs[channel.rb].add(ids)
        sinrs_dB = {}
        for (tx_id, rx_id), channel in self.channels.items():
            tx, rx = self.devices[tx_id], self.devices[rx_id]
            tx_pwr_dBm = tx.eirp_dBm(channel.tx_pwr_dBm)
            d = channel.distance
            path_loss_dB = self.path_loss(tx, rx, d)
            rx_pwr_dBm = tx_pwr_dBm - path_loss_dB

            ix_channels = rbs[channel.rb].difference({(tx_id, rx_id)})
            sum_ix_pwr_mW = 0
            for ix_tx_id, ix_rx_id in ix_channels:
                ix_tx, ix_rx = self.devices[ix_tx_id], self.devices[ix_rx_id]
                ix_channel = self.channels[(ix_tx_id, ix_rx_id)]
                ix_tx_pwr_dBm = ix_tx.eirp_dBm(ix_channel.tx_pwr_dBm)
                ix_path_loss_dB = self.path_loss(ix_tx, ix_rx, channel.distance)
                sum_ix_pwr_mW += dB_to_linear(ix_tx_pwr_dBm - ix_path_loss_dB)

            nx_pwr_mW = dB_to_linear(tx.thermal_noise_dBm)
            ixnx_pwr_mW = sum_ix_pwr_mW + nx_pwr_mW
            sinrs_dB[(tx_id, rx_id)] = rx_pwr_dBm - linear_to_dB(ixnx_pwr_mW)

        return {
            'sinrs': sinrs_dB,
        }

    def add_base_station(self, bs_id, config: dict) -> BaseStation:
        bs = BaseStation(bs_id, config)
        self.base_stations[bs_id] = bs
        self.devices[bs_id] = bs
        return bs

    def add_ue(self, ue_id, config: dict) -> UserEquipment:
        ue = UserEquipment(ue_id, config)
        self.ues[ue_id] = ue
        self.devices[ue_id] = ue
        return ue

    def add_traffic_model(self, traffic_model: TrafficModel):
        ids = tuple(d.id for d in traffic_model.devices)
        self.traffic_models[ids] = traffic_model

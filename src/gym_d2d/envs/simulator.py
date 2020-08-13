from typing import Any, Dict, Tuple

from .action import Action
from .channel import Channel
from .device import Device, BaseStation, UserEquipment
from .id import Id
from .traffic_model import TrafficModel


class D2DSimulator:
    def __init__(self, num_rbs: int) -> None:
        super().__init__()
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
            self.channels[(action.tx_id, action.rx_id)] = Channel.from_action(action)

        return {}

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

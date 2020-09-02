from typing import Dict, List, Tuple

from .channel import Channel
from .device import BaseStation, UserEquipment
from .id import Id
from .link_type import LinkType


class TrafficModel:
    def __init__(self, bs: BaseStation, ues: List[UserEquipment], num_rbs: int) -> None:
        super().__init__()
        self.bs: BaseStation = bs
        self.ues: List[UserEquipment] = ues
        self.num_rbs: int = num_rbs

    def get_traffic(self) -> Dict[Tuple[Id, Id], Channel]:
        pass


class UplinkTrafficModel(TrafficModel):
    def get_traffic(self) -> Dict[Tuple[Id, Id], Channel]:
        rb = 0
        traffic = {}
        for ue in self.ues:
            traffic[(ue.id, self.bs.id)] = Channel(ue, self.bs, LinkType.UPLINK, rb, ue.max_tx_power_dBm)
            rb = (rb + 1) % self.num_rbs
        return traffic


class DownlinkTrafficModel(TrafficModel):
    def get_traffic(self) -> Dict[Tuple[Id, Id], Channel]:
        rb = 0
        traffic = {}
        for ue in self.ues:
            traffic[(self.bs.id, ue.id)] = Channel(self.bs, ue, LinkType.DOWNLINK, rb, ue.max_tx_power_dBm)
            rb = (rb + 1) % self.num_rbs
        return traffic

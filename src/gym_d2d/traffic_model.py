from typing import Dict, Tuple

from .channel import Channel
from .devices import Devices
from .id import Id
from .link_type import LinkType


class TrafficModel:
    def __init__(self, num_rbs: int) -> None:
        super().__init__()
        self.num_rbs: int = num_rbs

    def get_traffic(self, devices: Devices) -> Dict[Tuple[Id, Id], Channel]:
        pass


class UplinkTrafficModel(TrafficModel):
    def get_traffic(self, devices: Devices) -> Dict[Tuple[Id, Id], Channel]:
        rb = 0
        traffic = {}
        for cue_id, cue in devices.cues.items():
            traffic[(cue_id, devices.bs.id)] = Channel(cue, devices.bs, LinkType.UPLINK, rb, cue.max_tx_power_dBm)
            rb = (rb + 1) % self.num_rbs
        return traffic


class DownlinkTrafficModel(TrafficModel):
    def get_traffic(self, devices: Devices) -> Dict[Tuple[Id, Id], Channel]:
        rb = 0
        traffic = {}
        for cue_id, cue in devices.cues.items():
            traffic[(devices.bs.id, cue_id)] = Channel(devices.bs, cue, LinkType.DOWNLINK, rb, cue.max_tx_power_dBm)
            rb = (rb + 1) % self.num_rbs
        return traffic

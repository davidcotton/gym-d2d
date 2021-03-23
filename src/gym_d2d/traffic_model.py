from gym_d2d.actions import Actions, Action
from .devices import Devices
from .link_type import LinkType


class TrafficModel:
    def __init__(self, num_rbs: int) -> None:
        super().__init__()
        self.num_rbs: int = num_rbs

    def get_traffic(self, devices: Devices) -> Actions:
        pass


class UplinkTrafficModel(TrafficModel):
    def get_traffic(self, devices: Devices) -> Actions:
        rb = 0
        traffic = Actions()
        for cue_id, cue in devices.cues.items():
            traffic[(cue_id, devices.bs.id)] = Action(cue, devices.bs, LinkType.UPLINK, rb, cue.max_tx_power_dBm)
            rb = (rb + 1) % self.num_rbs
        return traffic


class DownlinkTrafficModel(TrafficModel):
    def get_traffic(self, devices: Devices) -> Actions:
        rb = 0
        traffic = Actions()
        for cue_id, cue in devices.cues.items():
            traffic[(devices.bs.id, cue_id)] = Action(devices.bs, cue, LinkType.DOWNLINK, rb, cue.max_tx_power_dBm)
            rb = (rb + 1) % self.num_rbs
        return traffic

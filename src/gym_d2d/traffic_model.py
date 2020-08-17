from typing import List, Tuple

from .channel import Channel
from .device import Device
from .mode import Mode


class TrafficModel:
    def __init__(self, devices: List[Device]) -> None:
        super().__init__()
        self.devices: List[Device] = devices

    def get_traffic(self) -> Tuple[Device, Device]:
        pass

    def get_channel(self, rb: int) -> Channel:
        pass


class SimplexTrafficModel(TrafficModel):
    def get_traffic(self) -> Tuple[Device, Device]:
        return self.devices[0], self.devices[1]

    def get_channel(self, rb: int) -> Channel:
        tx_pwr = self.devices[0].max_tx_power_dBm
        return Channel(self.devices[0], self.devices[1], Mode.CELLULAR, rb, tx_pwr)

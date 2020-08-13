from typing import Dict

from .device import Device, BaseStation, UserEquipment
from .id import Id


class D2DSimulator:
    def __init__(self, num_rbs: int) -> None:
        super().__init__()
        self.num_rbs: int = num_rbs
        self.devices: Dict[Id, Device] = {}
        self.base_stations: Dict[Id, BaseStation] = {}
        self.ues: Dict[Id, UserEquipment] = {}
        # self.links: Dict[Id, Link] = {}
        self.links: Dict[Tuple[Id, Id], Link] = {}
        self.traffic_models: Dict[Tuple[Any], TrafficModel] = {}


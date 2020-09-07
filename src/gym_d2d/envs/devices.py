from dataclasses import dataclass
from typing import Dict, Tuple

from gym_d2d.device import Device, BaseStation, UserEquipment
from gym_d2d.id import Id


@dataclass
class Devices:
    bs: BaseStation
    cues: Dict[Id, UserEquipment]
    dues: Dict[Tuple[Id, Id], Tuple[UserEquipment, UserEquipment]]

    def __post_init__(self):
        self.due_pairs = {tx_id: rx_id for tx_id, rx_id in self.dues.keys()}
        self.due_pairs_inv = {rx_id: tx_id for tx_id, rx_id in self.dues.keys()}

    def to_dict(self) -> Dict[Id, Device]:
        dues = {}
        for (tx_id, rx_id), (tx, rx) in self.dues.items():
            dues[tx_id] = tx
            dues[rx_id] = rx
        return {
            self.bs.id: self.bs,
            **self.cues,
            **dues
        }

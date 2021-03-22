from collections.abc import Mapping
from typing import Dict, Tuple

from gym_d2d.device import Device, BaseStation, UserEquipment
from gym_d2d.id import Id


class Devices(Mapping):
    def __init__(self,
                 bs: BaseStation,
                 cues: Dict[Id, UserEquipment],
                 due_pairs: Dict[Tuple[Id, Id], Tuple[UserEquipment, UserEquipment]]
                 ) -> None:
        super().__init__()
        self.bs: BaseStation = bs
        self.cues: Dict[Id, UserEquipment] = cues
        self.dues: Dict[Tuple[Id, Id], Tuple[UserEquipment, UserEquipment]] = due_pairs
        self.due_pairs = {}
        self.due_pairs_inv = {}
        self._devices: Dict[Id, Device] = {bs.id: bs, **cues}
        for (tx_id, rx_id), (tx, rx) in due_pairs.items():
            self._devices[tx_id] = tx
            self._devices[rx_id] = rx
            self.due_pairs[tx_id] = rx_id
            self.due_pairs_inv[rx_id] = tx_id

    def __getitem__(self, key: Id) -> Device:
        return self._devices[key]

    def __len__(self) -> int:
        return len(self._devices)

    def __iter__(self):
        return iter(self._devices)

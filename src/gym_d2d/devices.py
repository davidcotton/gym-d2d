from collections import UserDict, defaultdict
from collections.abc import Mapping, MutableMapping
from typing import Dict, Iterator, List, Tuple

from gym_d2d.device import Device, BaseStation, UserEquipment
from gym_d2d.id import Id
from gym_d2d.link_type import LinkType
from gym_d2d.utils import BijectiveDict, SurjectiveDict

Devices = dict


# class Devices(Mapping):
#     def __init__(self,
#                  bs: BaseStation,
#                  cues: Dict[Id, UserEquipment],
#                  due_pairs: Dict[Tuple[Id, Id], Tuple[UserEquipment, UserEquipment]]
#                  ) -> None:
#         super().__init__()
#         self.bs: BaseStation = bs
#         self.cues: Dict[Id, UserEquipment] = cues
#         self.dues: Dict[Tuple[Id, Id], Tuple[UserEquipment, UserEquipment]] = due_pairs
#         self.due_pairs = {}
#         self.due_pairs_inv = {}
#         self._devices: Dict[Id, Device] = {bs.id: bs, **cues}
#         for (tx_id, rx_id), (tx, rx) in due_pairs.items():
#             self._devices[tx_id] = tx
#             self._devices[rx_id] = rx
#             self.due_pairs[tx_id] = rx_id
#             self.due_pairs_inv[rx_id] = tx_id
#
#     def __getitem__(self, key: Id) -> Device:
#         return self._devices[key]
#
#     def __len__(self) -> int:
#         return len(self._devices)
#
#     def __iter__(self):
#         return iter(self._devices)


# class DeviceMap(UserDict):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(**kwargs)
#         self._linktypes = BidirectionalDict()
#         for tx_id, rx_id, linktype in args[0]:
#             self.__setitem__(tx_id, rx_id)
#             self._linktypes[linktype] = (tx_id, rx_id)
#             foo = 1
#         foo = 1
#
#     def __setitem__(self, key, value):
#         if key in self:
#             del self[self[key]]
#         if value in self:
#             del self[value]
#         super().__setitem__(key, value)
#         super().__setitem__(value, key)
#         self._linktypes[key] = value
#
#     def __delitem__(self, key):
#         value = self[key]
#         super().__delitem__(key)
#         self.pop(value, None)
#         self._linktypes.pop(key)
#
#     def get_by_linktype(self, key):
#         return self._linktypes[key]


# class DeviceMap(dict):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(**kwargs)
#         # super().__init__(*args, **kwargs)
#         # self.inverse = {}
#         self.linktypes = SurjectiveDict()
#         # for key, value in self.items():
#         #     self.inverse.setdefault(value, []).append(key)
#
#         self.devices = SurjectiveDict()
#
#         # for tx_id, rx_id, linktype in args[0]:
#         #     self.__setitem__(tx_id, rx_id)
#         #     self.linktypes[linktype] = (tx_id, rx_id)
#         #     foo = 1
#         foo = 1
#
#     def __setitem__(self, key: Tuple[Id, Id], linktype: LinkType) -> None:
#         tx_id, rx_id = key
#         # if tx_id in self:
#         #     self.inverse[self[tx_id]].remove(tx_id)
#         # super().__setitem__(tx_id, rx_id)
#         # self.inverse.setdefault(rx_id, []).append(tx_id)
#         self.linktypes[key] = linktype
#         self.devices[tx_id] = rx_id
#         foo = 1
#
#     def __delitem__(self, key: Tuple[Id, Id]):
#         # self.inverse.setdefault(self[key], []).remove(key)
#         # if self[key] in self.inverse and not self.inverse[self[key]]:
#         #     del self.inverse[self[key]]
#         # super().__delitem__(key)
#
#         tx_id, rx_id = key
#         self.devices[tx_id].__delitem__(rx_id)
#         self.linktypes[key].__delitem__(key)
#         foo = 1


class DeviceMap(MutableMapping):
    def __init__(self) -> None:
        super().__init__()
        self.devices = {}
        self.inverse = {}
        self.linktypes = SurjectiveDict()

    def __getitem__(self, key: Tuple[Id, Id]) -> LinkType:
        tx_id, rx_id = key
        return self.devices[tx_id]

    def __setitem__(self, key: Tuple[Id, Id], linktype: LinkType) -> None:
        tx_id, rx_id = key
        if tx_id not in self.devices:
            self.devices[tx_id] = [rx_id]
            self.inverse[rx_id] = [tx_id]
        else:
            self.devices[tx_id].append(rx_id)
            self.devices[rx_id].append(tx_id)
        self.linktypes[key] = linktype
        foo = 1

    def __delitem__(self, key: Tuple[Id, Id]) -> None:
        raise NotImplementedError
        # self.linktypes.__delitem__(key)
        # tx_id, rx_id = key

    def __len__(self) -> int:
        return len(self.devices)

    def __iter__(self) -> Iterator:
        return iter(self.devices)

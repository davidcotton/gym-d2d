from collections import UserDict, defaultdict
from dataclasses import dataclass
from typing import Dict, Set

from gym_d2d.device import Device
from gym_d2d.link_type import LinkType


@dataclass(frozen=True)
class Action:
    tx: Device
    rx: Device
    link_type: LinkType
    rb: int
    tx_pwr_dBm: float


class Actions(UserDict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rbs: Dict[int, Set[Action]] = defaultdict(set)

    def clear(self) -> None:
        super().clear()
        self._rbs.clear()

    def get_actions_by_rb(self, rb: int) -> Set[Action]:
        if not self._rbs:
            for action in self.data.values():
                self._rbs[action.rb].add(action)
        return self._rbs[rb]

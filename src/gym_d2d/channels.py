from collections import defaultdict, UserDict
from typing import Dict, Set

from gym_d2d.channel import Channel


class Channels(UserDict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rbs: Dict[int, Set[Channel]] = defaultdict(set)

    def clear(self) -> None:
        super().clear()
        self._rbs.clear()

    def get_channels_by_rb(self, rb: int) -> Set[Channel]:
        if not self._rbs:
            for channel in self.data.values():
                self._rbs[channel.rb].add(channel)
        return self._rbs[rb]

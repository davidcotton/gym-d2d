from __future__ import annotations

# from .action import Action
from .device import Device
from .mode import Mode


UNINITIALISED_DISTANCE = -1.0


class Channel:
    def __init__(self, tx: Device, rx: Device, mode: Mode, rb: int, tx_pwr_dBm: float) -> None:
        super().__init__()
        self.tx: Device = tx
        self.rx: Device = rx
        self.mode: Mode = mode
        self.rb: int = rb
        self.tx_pwr_dBm: float = tx_pwr_dBm
        self._distance: float = UNINITIALISED_DISTANCE

    # @staticmethod
    # def from_action(action: Action) -> Channel:
    #     return Channel(*action)

    @property
    def distance(self) -> float:
        if self._distance == UNINITIALISED_DISTANCE:
            self._distance = self.tx.position.distance(self.rx.position)
        return self._distance

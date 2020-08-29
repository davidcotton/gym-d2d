from __future__ import annotations

# from .action import Action
from .device import Device
from .link_type import LinkType


class Channel:
    def __init__(self, tx: Device, rx: Device, link_type: LinkType, rb: int, tx_pwr_dBm: float) -> None:
        super().__init__()
        self.tx: Device = tx
        self.rx: Device = rx
        self.link_type: LinkType = link_type
        self.rb: int = rb
        self.tx_pwr_dBm: float = tx_pwr_dBm

    # @staticmethod
    # def from_action(action: Action) -> Channel:
    #     return Channel(*action)

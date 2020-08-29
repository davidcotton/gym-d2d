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

    def __repr__(self) -> str:
        return f'<{self.tx.id}>:<{self.rx.id}>'

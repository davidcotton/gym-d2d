from .id import Id


class Device:
    def __init__(self, id_, config: dict) -> None:
        super().__init__()
        self.id = Id(id_)
        self.config: dict = config


class BaseStation(Device):
    pass


class UserEquipment(Device):
    pass

from dataclasses import dataclass


@dataclass
class Position:
    x: float
    y: float

    def distance(self, other) -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def as_tuple(self) -> tuple:
        return self.x, self.y

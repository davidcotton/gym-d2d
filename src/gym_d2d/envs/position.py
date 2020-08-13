from dataclasses import dataclass
from math import pi, sin, cos, sqrt
import random


@dataclass
class Position:
    x: float
    y: float

    def distance(self, other) -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def as_tuple(self) -> tuple:
        return self.x, self.y


def get_random_position(radius: float) -> Position:
    """Generate a random position somewhere within a circle.

    :param radius: The radius within which to generate the random position
    :return: The random position object.
    """
    theta = 2 * pi * random.random()
    r = radius * sqrt(random.random())
    x = r * cos(theta)
    y = r * sin(theta)
    return Position(x, y)

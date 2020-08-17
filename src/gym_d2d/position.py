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


def get_random_position_nearby(radius: float, anchor_pos: Position, anchor_radius: float) -> Position:
    """Generate a random position within range of an anchor position.

    :param radius: The radius within which to generate the random position
    :param anchor_pos: The position to which the generated position should appear near.
    :param anchor_radius: The maximum range the random position can be from the anchor.
    :return: A random position.
    """
    x, y = radius, radius
    while x**2 + y**2 > radius**2:
        theta = 2 * pi * random.random()
        r = anchor_radius * sqrt(random.random())
        x = anchor_pos.x + r * cos(theta)
        y = anchor_pos.y + r * sin(theta)
    return Position(x, y)

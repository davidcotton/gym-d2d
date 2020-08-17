from math import sqrt
import random

from pytest import approx

from gym_d2d.position import Position, get_random_position, get_random_position_nearby


NUM_TEST_REPEATS = 10


class TestPosition:
    def test_distance(self):
        for _ in range(NUM_TEST_REPEATS):
            a_x, a_y = random.uniform(0, 500), random.uniform(0, 500)
            b_x, b_y = random.uniform(0, 500), random.uniform(0, 500)
            pos_a = Position(a_x, a_y)
            pos_b = Position(b_x, b_y)
            dist = sqrt((a_x - b_x) ** 2 + (a_y - b_y) ** 2)
            assert dist >= 0.0
            assert dist == approx(pos_a.distance(pos_b))
            assert dist == approx(pos_b.distance(pos_a))

    def test_as_tuple(self):
        x, y = random.uniform(0, 500), random.uniform(0, 500)
        pos = Position(x, y)
        assert pos.as_tuple() == (x, y)


def test_get_random_position_in_radius():
    for _ in range(NUM_TEST_REPEATS):
        radius = random.uniform(0, 500)
        pos = get_random_position(radius)
        assert sqrt(pos.x ** 2 + pos.y ** 2) <= radius


def test_get_random_position_nearby_in_radius():
    for _ in range(NUM_TEST_REPEATS):
        radius = random.uniform(0, 1000)
        anchor_pos = get_random_position(radius)
        anchor_radius = random.uniform(0, 50)
        pos = get_random_position_nearby(radius, anchor_pos, anchor_radius)
        assert sqrt(pos.x ** 2 + pos.y ** 2) <= radius
        assert pos.distance(anchor_pos) <= anchor_radius

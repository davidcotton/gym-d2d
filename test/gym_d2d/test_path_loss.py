from pytest import approx

from gym_d2d.path_loss import pl_constant_dB, LogDistancePathLoss, ShadowingPathLoss, AreaType, CostHataPathLoss
from gym_d2d.device import BaseStation, UserEquipment
from gym_d2d.position import Position


def test_pl_constant_dB():
    ple = 2.0
    assert pl_constant_dB(2.0, ple) == approx(38.46838313516298)
    assert pl_constant_dB(2.1, ple) == approx(38.892169116561746)
    assert pl_constant_dB(2.2, ple) == approx(39.2962368383275)


class TestLogDistancePathLoss:
    def test_init_calculates_pl_constant(self):
        pl = LogDistancePathLoss(2.1)
        assert pl.pl_constant_dB == approx(38.892169116561746)

    def test_call(self):
        pl = LogDistancePathLoss(2.1)
        bs = BaseStation('bs')
        ue = UserEquipment('ue')
        ue.set_position(Position(250, 0))
        assert pl(ue, bs) == approx(86.85097)
        ue.set_position(Position(0, 500))
        assert pl(ue, bs) == approx(92.87156)


# @todo work out how to test random values
# class TestShadowingPathLoss:
#     def test_call(self):
#         pl = ShadowingPathLoss(2.1, 2.0, 100.0, 2.7)
#         bs = BaseStation('bs')
#         ue = UserEquipment('ue')
#         ue.set_position(Position(250, 0))
#         assert pl(ue, bs) == approx(86.85097)
#         ue.set_position(Position(0, 500))
#         assert pl(ue, bs) == approx(92.87156)


class TestCostHataPathLoss:
    def test_call(self):
        pl = CostHataPathLoss(2.1, AreaType.URBAN)
        bs = BaseStation('bs')
        ue = UserEquipment('ue')
        ue.set_position(Position(250, 0))
        assert pl(bs, ue) == approx(121.44557455875727)
        assert pl(ue, bs) == approx(114.35415557446962)
        ue.set_position(Position(0, 500))
        assert pl(bs, ue) == approx(132.2768393081241)
        assert pl(ue, bs) == approx(127.5231950610599)

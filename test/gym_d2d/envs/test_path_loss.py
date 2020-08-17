from pytest import approx

from gym_d2d.path_loss import calc_fspl_constant_dB, FreeSpacePathLoss
from gym_d2d.device import BaseStation, UserEquipment
from gym_d2d.position import Position


def test_calc_fspl_constant_dB():
    assert calc_fspl_constant_dB(2.0) == approx(38.46838313516298)
    assert calc_fspl_constant_dB(2.1) == approx(38.892169116561746)
    assert calc_fspl_constant_dB(2.2) == approx(39.2962368383275)


class TestFreeSpacePathLoss:
    def test_init_calculates_fspl_constant(self):
        pl = FreeSpacePathLoss(2.1)
        assert pl.fspl_constant_dB == approx(38.892169116561746)

    def test_call(self):
        pl = FreeSpacePathLoss(2.1)
        bs = BaseStation('bs')
        ue = UserEquipment('ue')
        ue.set_position(Position(250, 0))
        assert pl(ue, bs) == approx(86.85097)
        ue.set_position(Position(0, 500))
        assert pl(ue, bs) == approx(92.87156)

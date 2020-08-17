from pytest import approx

from gym_d2d.path_loss import calc_fspl_constant_dB, FreeSpacePathLoss
from gym_d2d.device import BaseStation, UserEquipment


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
        bs = BaseStation('bs', {
            'tx_antenna_gain_dBi': 17.5,
            'rx_antenna_gain_dBi': 17.5,
        })
        ue = UserEquipment('ue', {
            'tx_antenna_gain_dBi': 0.0,
            'rx_antenna_gain_dBi': 1.0,
        })
        assert pl(ue, bs, 250) == approx(69.3509692900025)
        assert pl(bs, ue, 250) == approx(68.3509692900025)
        assert pl(ue, bs, 500) == approx(75.37156920328212)
        assert pl(bs, ue, 500) == approx(74.37156920328212)

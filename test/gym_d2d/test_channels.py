import pytest

from gym_d2d.channel import Channel
from gym_d2d.channels import Channels
from gym_d2d.device import UserEquipment, BaseStation
from gym_d2d.link_type import LinkType


@pytest.fixture
def channels():
    due00 = UserEquipment('due00')
    due01 = UserEquipment('due01')
    due02 = UserEquipment('due02')
    due03 = UserEquipment('due03')
    cue = UserEquipment('cue')
    bs = BaseStation('bs')

    return Channels({
        (due00.id, due01.id): Channel(due00, due01, LinkType.SIDELINK, 0, 17),
        (due02.id, due03.id): Channel(due02, due03, LinkType.SIDELINK, 1, 15),
        (cue.id, bs.id): Channel(cue, bs, LinkType.UPLINK, 0, 23),
    })


def test_get_channels_by_rb(channels):
    # ensure Channels can group channels by RB
    assert channels.get_channels_by_rb(0) == {channels[('due00', 'due01')], channels[('cue', 'bs')]}
    assert channels.get_channels_by_rb(1) == {channels[('due02', 'due03')]}
    # ensure KeyError is raised if RB not present
    with pytest.raises(KeyError):
        assert channels.get_channels_by_rb(2)


def test_clear(channels):
    # ensure clearing, also clears dict of RBs
    channels.clear()
    assert channels == {}
    with pytest.raises(KeyError):
        assert channels.get_channels_by_rb(0)
        assert channels.get_channels_by_rb(1)


def test_get_channels_by_rb_after_clear(channels):
    # ensure that after clearing, still groups by channel as expected
    _ = channels.get_channels_by_rb(0)
    channels.clear()
    test_channel = Channel(UserEquipment('due00'), UserEquipment('due01'), LinkType.SIDELINK, 0, 17)
    channels[('due00', 'due01')] = test_channel
    assert channels.get_channels_by_rb(0) == {test_channel}
    with pytest.raises(KeyError):
        assert channels.get_channels_by_rb(1)

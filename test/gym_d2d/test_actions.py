import pytest
from gym_d2d.actions import Actions, Action

from gym_d2d.device import UserEquipment, BaseStation
from gym_d2d.link_type import LinkType


@pytest.fixture
def actions():
    due00 = UserEquipment('due00')
    due01 = UserEquipment('due01')
    due02 = UserEquipment('due02')
    due03 = UserEquipment('due03')
    cue = UserEquipment('cue')
    bs = BaseStation('bs')

    return Actions({
        (due00.id, due01.id): Action(due00, due01, LinkType.SIDELINK, 0, 17),
        (due02.id, due03.id): Action(due02, due03, LinkType.SIDELINK, 1, 15),
        (cue.id, bs.id): Action(cue, bs, LinkType.UPLINK, 0, 23),
    })


def test_get_actions_by_rb(actions):
    # ensure Actions can group by RB
    assert actions.get_actions_by_rb(0) == {actions[('due00', 'due01')], actions[('cue', 'bs')]}
    assert actions.get_actions_by_rb(1) == {actions[('due02', 'due03')]}
    assert actions.get_actions_by_rb(2) == set()


def test_clear(actions):
    # ensure clearing, also clears dict of RBs
    actions.clear()
    assert actions == {}
    assert actions.get_actions_by_rb(0) == set()
    assert actions.get_actions_by_rb(1) == set()
    assert actions.get_actions_by_rb(2) == set()


def test_get_actions_by_rb_after_clear(actions):
    # ensure that after clearing, still groups by action as expected
    _ = actions.get_actions_by_rb(0)
    actions.clear()
    test_action = Action(UserEquipment('due00'), UserEquipment('due01'), LinkType.SIDELINK, 0, 17)
    actions[('due00', 'due01')] = test_action
    assert actions.get_actions_by_rb(0) == {test_action}
    assert actions.get_actions_by_rb(1) == set()
    assert actions.get_actions_by_rb(2) == set()

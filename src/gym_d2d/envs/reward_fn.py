from abc import ABC, abstractmethod
from math import log2
from typing import Dict

from gym_d2d.action import Actions
from gym_d2d.channels import Channels
from gym_d2d.devices import Devices
from gym_d2d.conversion import dB_to_linear
from gym_d2d.link_type import LinkType


class RewardFunction(ABC):
    @abstractmethod
    def __call__(self, actions: Actions, state: dict, channels: Channels, devices: Devices) -> Dict[str, float]:
        """Calculate rewards for each action.

        :param actions: Dict of actions to calculate rewards for.
        :param state: Dict of SINRs, etc. representing the simulation state after taking the actions.
        :param channels: Dict of channels generated by the actions.
        :param devices: Dict of devices in the simulation.
        :returns: A dict mapping tx-rx ID pairs to scalar rewards.
        """
        pass


class SystemCapacityRewardFunction(RewardFunction):
    def __call__(self, actions: Actions, state: dict, channels: Channels, devices: Devices) -> Dict[str, float]:
        reward = -1
        _break = False
        for tx_id, rx_id in devices.due_pairs.items():
            if _break:
                break
            channel = channels[(tx_id, rx_id)]
            ix_channels = channels.get_channels_by_rb(channel.rb).difference({channel})
            for ix_channel in ix_channels:
                if ix_channel.tx.id in devices.due_pairs:
                    continue
                if state['capacity_mbps'][(ix_channel.tx.id, ix_channel.rx.id)] <= 0:
                    _break = True
                    break
        else:
            sum_capacity = sum(state['capacity_mbps'].values())
            reward = sum_capacity / len(devices.due_pairs)

        return {':'.join(tx_rx_id): reward for tx_rx_id in actions.keys()}


class DueShannonRewardFunction(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self.min_sinr = -70.0

    def __call__(self, actions: Actions, state: dict, channels: Channels, devices: Devices) -> Dict[str, float]:
        rewards = {}
        for tx_id, rx_id in devices.due_pairs.items():
            sinr = state['sinrs_db'][(tx_id, rx_id)]
            if sinr >= self.min_sinr:
                rewards[tx_id] = log2(1 + dB_to_linear(sinr))
            else:
                rewards[tx_id] = -1
        return rewards


class CueSinrShannonRewardFunction(RewardFunction):
    def __init__(self, sinr_threshold_dB=0.0) -> None:
        super().__init__()
        self.sinr_threshold_dB: float = float(sinr_threshold_dB)

    def __call__(self, actions: Actions, state: dict, channels: Channels, devices: Devices) -> Dict[str, float]:
        rewards = {}
        for tx_id, rx_id in devices.due_pairs.items():
            channel = channels[(tx_id, rx_id)]
            ix_channels = channels.get_channels_by_rb(channel.rb).difference({channel})
            rewards[tx_id] = -1
            for ix_channel in ix_channels:
                if ix_channel.link_type != LinkType.SIDELINK:
                    cue_sinr_dB = state['sinrs_db'][(ix_channel.tx.id, ix_channel.rx.id)]
                    if cue_sinr_dB < self.sinr_threshold_dB:
                        break
            else:
                rewards[tx_id] = log2(1 + dB_to_linear(state['sinrs_db'][(tx_id, rx_id)]))
        return rewards

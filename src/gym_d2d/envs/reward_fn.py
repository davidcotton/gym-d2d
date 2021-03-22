from abc import ABC, abstractmethod
from collections import defaultdict
from math import log2
from typing import Dict, Tuple

from gym_d2d.channel import Channel
from gym_d2d.devices import Devices
from gym_d2d.id import Id
from gym_d2d.conversion import dB_to_linear
from gym_d2d.link_type import LinkType


class RewardFunction(ABC):
    @abstractmethod
    def __call__(self, state: dict, channels: Dict[Tuple[Id, Id], Channel], devices: Devices) -> Dict[Id, float]:
        pass


class SystemCapacityRewardFunction(RewardFunction):
    def __call__(self, state: dict, channels: Dict[Tuple[Id, Id], Channel], devices: Devices) -> Dict[Id, float]:
        # group by RB
        rbs = defaultdict(set)
        for ids, channel in channels.items():
            rbs[channel.rb].add(ids)

        reward = -1
        _break = False
        for tx_id, rx_id in devices.due_pairs.items():
            if _break:
                break
            rb = channels[(tx_id, rx_id)].rb
            ix_channels = rbs[rb].difference({(tx_id, rx_id)})
            for ix_tx_id, ix_rx_id in ix_channels:
                if ix_tx_id in devices.due_pairs:
                    continue
                if state['capacity_mbps'][(ix_tx_id, ix_rx_id)] <= 0:
                    _break = True
                    break
        else:
            sum_capacity = sum(state['capacity_mbps'].values())
            reward = sum_capacity / len(devices.due_pairs)

        return {tx_id: reward for tx_id in devices.due_pairs.keys()}


class DueShannonRewardFunction(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self.min_sinr = -70.0

    def __call__(self, state: dict, channels: Dict[Tuple[Id, Id], Channel], devices: Devices) -> Dict[Id, float]:
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

    def __call__(self, state: dict, channels: Dict[Tuple[Id, Id], Channel], devices: Devices) -> Dict[Id, float]:
        # group channels by RB
        rbs = defaultdict(set)
        for ids, channel in channels.items():
            rbs[channel.rb].add(channel)

        rewards = {}
        for tx_id, rx_id in devices.due_pairs.items():
            channel = channels[(tx_id, rx_id)]
            ix_channels = rbs[channel.rb].difference({channel})
            rewards[tx_id] = -1
            for ix_channel in ix_channels:
                if ix_channel.link_type != LinkType.SIDELINK:
                    cue_sinr_dB = state['sinrs_db'][(ix_channel.tx.id, ix_channel.rx.id)]
                    if cue_sinr_dB < self.sinr_threshold_dB:
                        break
            else:
                rewards[tx_id] = log2(1 + dB_to_linear(state['sinrs_db'][(tx_id, rx_id)]))
        return rewards

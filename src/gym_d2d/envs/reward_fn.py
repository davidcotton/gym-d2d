from abc import ABC, abstractmethod
from collections import defaultdict
from math import log2
from typing import Dict

from .devices import Devices
from gym_d2d.id import Id
from gym_d2d.simulator import D2DSimulator
from gym_d2d.conversion import dB_to_linear
from gym_d2d.link_type import LinkType


class RewardFunction(ABC):
    def __init__(self, simulator: D2DSimulator, devices: Devices) -> None:
        super().__init__()
        self.simulator: D2DSimulator = simulator
        self.devices: Devices = devices

    @abstractmethod
    def __call__(self, results: dict) -> Dict[Id, float]:
        pass


class SystemCapacityRewardFunction(RewardFunction):
    def __call__(self, results: dict) -> Dict[Id, float]:
        # group by RB
        rbs = defaultdict(set)
        for ids, channel in self.simulator.channels.items():
            rbs[channel.rb].add(ids)

        reward = -1
        _break = False
        for tx_id, rx_id in self.devices.due_pairs.items():
            if _break:
                break
            rb = self.simulator.channels[(tx_id, rx_id)].rb
            ix_channels = rbs[rb].difference({(tx_id, rx_id)})
            for ix_tx_id, ix_rx_id in ix_channels:
                if ix_tx_id in self.devices.due_pairs:
                    continue
                if results['capacity_mbps'][(ix_tx_id, ix_rx_id)] <= 0:
                    _break = True
                    break
        else:
            sum_capacity = sum(results['capacity_mbps'].values())
            reward = sum_capacity / len(self.devices.due_pairs)

        return {tx_id: reward for tx_id in self.devices.due_pairs.keys()}


class DueShannonRewardFunction(RewardFunction):
    def __init__(self, simulator: D2DSimulator, devices: Devices) -> None:
        super().__init__(simulator, devices)
        self.min_sinr = -70.0

    def __call__(self, results: dict) -> Dict[Id, float]:
        rewards = {}
        for tx_id, rx_id in self.devices.due_pairs.items():
            sinr = results['sinrs_db'][(tx_id, rx_id)]
            if sinr >= self.min_sinr:
                rewards[tx_id] = log2(1 + dB_to_linear(sinr))
            else:
                rewards[tx_id] = -1
        return rewards


class CueSinrShannonRewardFunction(RewardFunction):
    def __init__(self, simulator: D2DSimulator, devices: Devices, sinr_threshold_dB=0.0) -> None:
        super().__init__(simulator, devices)
        self.sinr_threshold_dB: float = float(sinr_threshold_dB)

    def __call__(self, results: dict) -> Dict[Id, float]:
        # group channels by RB
        rbs = defaultdict(set)
        for ids, channel in self.simulator.channels.items():
            rbs[channel.rb].add(channel)

        rewards = {}
        for tx_id, rx_id in self.devices.due_pairs.items():
            channel = self.simulator.channels[(tx_id, rx_id)]
            ix_channels = rbs[channel.rb].difference({channel})
            rewards[tx_id] = -1
            for ix_channel in ix_channels:
                if ix_channel.link_type != LinkType.SIDELINK:
                    cue_sinr_dB = results['sinrs_db'][(ix_channel.tx.id, ix_channel.rx.id)]
                    if cue_sinr_dB < self.sinr_threshold_dB:
                        break
            else:
                rewards[tx_id] = log2(1 + dB_to_linear(results['sinrs_db'][(tx_id, rx_id)]))
        return rewards


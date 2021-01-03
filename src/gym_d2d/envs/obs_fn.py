from abc import ABC, abstractmethod
from typing import Dict

from gym import Space, spaces
import numpy as np

from .devices import Devices
from gym_d2d.id import Id
from gym_d2d.simulator import D2DSimulator


class ObsFunction(ABC):
    def __init__(self, simulator: D2DSimulator, devices: Devices) -> None:
        super().__init__()
        self.simulator: D2DSimulator = simulator
        self.devices: Devices = devices

    @abstractmethod
    def get_obs_space(self, env_config: dict) -> Space:
        pass

    @abstractmethod
    def get_state(self, results: dict) -> Dict[Id, np.array]:
        pass


class Linear1ObsFunction(ObsFunction):
    def get_obs_space(self, env_config: dict) -> Space:
        num_txs = env_config['num_cellular_users'] + env_config['num_d2d_pairs']
        # num_rxs = 1 + self.num_d2d_pairs  # basestation + num D2D rxs
        num_tx_obs = 5  # sinrs, tx_pwrs, rbs, xs, ys
        num_rx_obs = 2  # xs, ys
        # obs_shape = ((num_txs * num_tx_obs) + (num_rxs * num_rx_obs),)
        obs_shape = ((num_txs * num_tx_obs) + (num_txs * num_rx_obs),)
        return spaces.Box(low=-env_config['cell_radius_m'], high=env_config['cell_radius_m'], shape=obs_shape)

    def get_state(self, results: dict) -> Dict[Id, np.array]:
        tx_pwrs_dBm = []
        rbs = []
        positions = []
        for channel in self.simulator.channels.values():
            tx_pwrs_dBm.append(channel.tx_pwr_dBm)
            rbs.append(channel.rb)
            positions.extend(list(channel.tx.position.as_tuple()))
        for channel in self.simulator.channels.values():
            positions.extend(list(channel.rx.position.as_tuple()))
        common_obs = []
        common_obs.extend(list(results['sinrs_db'].values()))
        common_obs.extend(tx_pwrs_dBm)
        common_obs.extend(rbs)
        common_obs.extend(positions)

        return {due_id: np.array(common_obs) for due_id in self.devices.due_pairs.keys()}


class Linear2ObsFunction(ObsFunction):
    def get_obs_space(self, env_config: dict) -> Space:
        num_txs = env_config['num_cellular_users'] + env_config['num_d2d_pairs']
        num_due_obs = 2  # x, y
        num_common_obs = 7  # tx_x, tx_y, rx_x, rx_y, tx_pwr, rb, sinr
        obs_shape = (num_due_obs + (num_common_obs * num_txs),)
        return spaces.Box(low=-env_config['cell_radius_m'], high=env_config['cell_radius_m'], shape=obs_shape)

    def get_state(self, results: dict) -> Dict[Id, np.array]:
        common_obs = []
        for channel in self.simulator.channels.values():
            common_obs.extend(list(channel.tx.position.as_tuple()))
            common_obs.extend(list(channel.rx.position.as_tuple()))
            common_obs.append(channel.tx_pwr_dBm)
            common_obs.append(channel.rb)
            common_obs.append(results['sinrs_db'][(channel.tx.id, channel.rx.id)])

        obs_dict = {}
        for due_id in self.devices.due_pairs.keys():
            due_pos = list(self.simulator.devices[due_id].position.as_tuple())
            due_obs = []
            due_obs.extend(due_pos)
            due_obs.extend(common_obs)
            obs_dict[due_id] = np.array(due_obs)

        return obs_dict

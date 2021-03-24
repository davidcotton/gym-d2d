from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict

from gym import Space, spaces
import numpy as np

from gym_d2d.actions import Actions
from gym_d2d.devices import Devices
from gym_d2d.envs.env_config import EnvConfig


class ObsFunction(ABC):
    @abstractmethod
    def get_obs_space(self, env_config: EnvConfig) -> Space:
        """Calculate the observation space generated by this observation function.

        :param env_config: The environment's configuration.
        :returns: An OpenAI Gym Space.
        """
        pass

    @abstractmethod
    def get_state(self, actions: Actions, state: dict, devices: Devices, device_map) -> Dict[str, np.array]:
        """Calculate the next observations for each agent.

        :param actions: Dict of previous actions (empty 1st step).
        :param state: Dict of SINRs, etc. representing the simulation state after taking the actions.
        :param devices: Dict of devices in the simulation.
        :param device_mapping_fn:
        :returns: A dict mapping tx-rx ID pairs to observations in numpy arrays.
        """
        pass


class LinearObsFunction(ObsFunction):
    def get_obs_space(self, env_config: EnvConfig) -> Space:
        r = env_config.cell_radius_m
        num_txs = env_config.num_cues + env_config.num_due_pairs
        self.num_obs = 6  # tx_x, tx_y, rx_x, rx_y, sinr, snr
        obs_shape = (self.num_obs * num_txs,)
        return spaces.Box(low=-r, high=r, shape=obs_shape)

    def get_state(self, actions: Actions, state: dict, devices: Devices, device_map) -> Dict[str, np.array]:
        agent_obs = defaultdict(lambda: [0.0] * self.num_obs)
        for id_pair, action in actions.items():
            agent_obs[id_pair] = list(action.tx.position.as_tuple() + action.rx.position.as_tuple())
            agent_obs[id_pair].append(state['sinrs_db'][id_pair])
            agent_obs[id_pair].append(state['snrs_db'][id_pair])

        obses = {}
        for tx_id, rx_ids in device_map.items():
            for rx_id in rx_ids:
                id_pair = (tx_id, rx_id)
                tx_obs_copy = agent_obs[id_pair][:]
                for other_tx_id, other_rx_ids in device_map.items():
                    for other_rx_id in other_rx_ids:
                        other_id_pair = (other_tx_id, other_rx_id)
                        if other_id_pair != id_pair:
                            tx_obs_copy.extend(agent_obs[(other_tx_id, other_rx_id)])
                obses[':'.join((tx_id, rx_id))] = np.array(tx_obs_copy)

        return obses

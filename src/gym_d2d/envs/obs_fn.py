from abc import ABC, abstractmethod
from typing import Dict

from gym import Space, spaces
import numpy as np

from gym_d2d.channels import Channels
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
    def get_state(self, state: dict, channels: Channels, devices: Devices) -> Dict[str, np.array]:
        """Calculate the next observations for each agent.

        :param state: Dict of SINRs, etc. representing the simulation state after taking the actions.
        :param channels: Dict of channels generated by the actions.
        :param devices: Dict of devices in the simulation.
        :returns: A dict mapping tx-rx ID pairs to observations in numpy arrays.
        """
        pass


class LinearObsFunction(ObsFunction):
    def get_obs_space(self, env_config: EnvConfig) -> Space:
        r = env_config.cell_radius_m
        num_txs = env_config.num_cues + env_config.num_due_pairs
        num_obs = 6  # tx_x, tx_y, rx_x, rx_y, sinr, snr
        obs_shape = (num_obs * num_txs,)
        return spaces.Box(low=-r, high=r, shape=obs_shape)

    def get_state(self, state: dict, channels: Channels, devices: Devices) -> Dict[str, np.array]:
        agent_obs = {}
        for tx_rx_id, channel in channels.items():
            agent_obs[tx_rx_id] = list(channel.tx.position.as_tuple() + channel.rx.position.as_tuple())
            agent_obs[tx_rx_id].append(state['sinrs_db'][tx_rx_id])
            agent_obs[tx_rx_id].append(state['snrs_db'][tx_rx_id])

        obses = {}
        for tx_rx_id in agent_obs:
            tx_obs_copy = agent_obs[tx_rx_id][:]
            for other_tx_rx_id, other_obs in agent_obs.items():
                if other_tx_rx_id != tx_rx_id:
                    tx_obs_copy.extend(other_obs)
            obses[':'.join(tx_rx_id)] = np.array(tx_obs_copy)

        return obses

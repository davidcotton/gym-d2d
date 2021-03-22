from abc import ABC, abstractmethod
from typing import Dict

from gym import Space, spaces
import numpy as np

from gym_d2d.channels import Channels
from gym_d2d.devices import Devices
from gym_d2d.envs.env_config import EnvConfig
from gym_d2d.id import Id


class ObsFunction(ABC):
    @abstractmethod
    def get_obs_space(self, env_config: EnvConfig) -> Space:
        pass

    @abstractmethod
    def get_state(self, state: dict, channels: Channels, devices: Devices) -> Dict[Id, np.array]:
        pass


class LinearObsFunction(ObsFunction):
    def get_obs_space(self, env_config: EnvConfig) -> Space:
        r = env_config.cell_radius_m
        num_txs = env_config.num_cues + env_config.num_due_pairs
        num_obs = 6  # tx_x, tx_y, rx_x, rx_y, sinr, snr
        obs_shape = (num_obs * num_txs,)
        return spaces.Box(low=-r, high=r, shape=obs_shape)

    def get_state(self, state: dict, channels: Channels, devices: Devices) -> Dict[Id, np.array]:
        agent_obs = {}
        for (tx_id, rx_id), channel in channels.items():
            agent_obs[tx_id] = list(channel.tx.position.as_tuple() + channel.rx.position.as_tuple())
            agent_obs[tx_id].append(state['sinrs_db'][(tx_id, rx_id)])
            agent_obs[tx_id].append(state['snrs_db'][(tx_id, rx_id)])

        obses = {}
        for tx_id in agent_obs:
            tx_obs_copy = agent_obs[tx_id][:]
            for other_tx_id, other_obs in agent_obs.items():
                if other_tx_id != tx_id:
                    tx_obs_copy.extend(other_obs)
            obses[tx_id] = np.array(tx_obs_copy)

        return obses

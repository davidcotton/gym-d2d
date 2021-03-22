from abc import ABC, abstractmethod
from typing import Dict, Tuple

from gym import Space, spaces
import numpy as np

from gym_d2d.channel import Channel
from gym_d2d.envs.devices import Devices
from gym_d2d.envs.env_config import EnvConfig
from gym_d2d.id import Id


class ObsFunction(ABC):
    @abstractmethod
    def get_obs_space(self, env_config: EnvConfig) -> Space:
        pass

    @abstractmethod
    def get_state(self, state: dict, channels: Dict[Tuple[Id, Id], Channel], devices: Devices) -> Dict[Id, np.array]:
        pass


class LinearObsFunction(ObsFunction):
    def get_obs_space(self, env_config: EnvConfig) -> Space:
        r = env_config.cell_radius_m
        num_txs = env_config.num_cues + env_config.num_due_pairs
        num_obs = 6  # tx_x, tx_y, rx_x, rx_y, sinr, snr
        obs_shape = (num_obs * num_txs,)
        return spaces.Box(low=-r, high=r, shape=obs_shape)

    def get_state(self, state: dict, channels: Dict[Tuple[Id, Id], Channel], devices: Devices) -> Dict[Id, np.array]:
        obses = {}
        for (tx_id, rx_id), channel in channels.items():
            obses[tx_id] = list(channel.tx.position.as_tuple() + channel.rx.position.as_tuple())
            obses[tx_id].append(state['sinrs_db'][(tx_id, rx_id)])
            obses[tx_id].append(state['snrs_db'][(tx_id, rx_id)])

        obs_dict = {}
        for tx_id in obses:
            tx_obs_copy = obses[tx_id][:]
            for other_tx_id, other_obs in obses.items():
                if other_tx_id != tx_id:
                    tx_obs_copy.extend(other_obs)
            obs_dict[tx_id] = np.array(tx_obs_copy)

        return obs_dict

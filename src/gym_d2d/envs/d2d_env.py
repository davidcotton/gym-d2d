import json
from pathlib import Path
from typing import Dict, Tuple, Any

import gym
from gym import spaces
import numpy as np

from gym_d2d.action import Action
from gym_d2d.envs.obs_fn import LinearObsFunction
from gym_d2d.envs.reward_fn import SystemCapacityRewardFunction
from gym_d2d.id import Id
from gym_d2d.link_type import LinkType
from gym_d2d.simulator import Simulator


EPISODE_LENGTH = 10
DEFAULT_OBS_FN = LinearObsFunction
DEFAULT_REWARD_FN = SystemCapacityRewardFunction


class D2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_config=None) -> None:
        super().__init__()
        env_config = env_config or {}
        self.simulator = Simulator(env_config)
        self.obs_fn = env_config.get('obs_fn', DEFAULT_OBS_FN)()
        self.observation_space = self.obs_fn.get_obs_space(self.simulator.config)
        self.num_pwr_actions = {  # +1 because include max value, i.e. from [0, ..., max]
            'due': self.simulator.config.due_max_tx_power_dBm - self.simulator.config.due_min_tx_power_dBm + 1,
            'cue': self.simulator.config.cue_max_tx_power_dBm + 1,
            'mbs': self.simulator.config.mbs_max_tx_power_dBm + 1
        }
        self.action_space = spaces.Dict({
            'due': spaces.Discrete(self.simulator.config.num_rbs * self.num_pwr_actions['due']),
            'cue': spaces.Discrete(self.simulator.config.num_rbs * self.num_pwr_actions['cue']),
            'mbs': spaces.Discrete(self.simulator.config.num_rbs * self.num_pwr_actions['mbs']),
        })
        self.reward_fn = env_config.get('reward_fn', DEFAULT_REWARD_FN)()
        self.state = None
        self.num_steps = 0

    def reset(self):
        self.num_steps = 0
        self.simulator.reset()
        # take a step with random D2D actions to generate initial SINRs
        random_actions = self._reset_random_actions()
        self.state = self.simulator.step(random_actions)
        obs = self.obs_fn.get_state(self.state, self.simulator.channels, self.simulator.devices)
        return obs

    def _reset_random_actions(self) -> Dict[Id, Action]:
        cue_actions = {_id: self._extract_action(_id, self.action_space['cue'].sample())
                       for _id in self.simulator.devices.cues.keys()}
        due_actions = {_id: self._extract_action(_id, self.action_space['due'].sample())
                       for _id, _ in self.simulator.devices.dues.keys()}
        return {**cue_actions, **due_actions}

    def step(self, actions):
        actions = self._extract_actions(actions)
        self.state = self.simulator.step(actions)
        self.num_steps += 1
        obs = self.obs_fn.get_state(self.state, self.simulator.channels, self.simulator.devices)
        rewards = self.reward_fn(self.state, self.simulator.channels, self.simulator.devices)
        game_over = {'__all__': self.num_steps >= EPISODE_LENGTH}
        info = self._info(actions, self.state)

        return obs, rewards, game_over, info

    def _extract_actions(self, actions: Dict[Id, object]) -> Dict[Id, Action]:
        return {tx_id: self._extract_action(tx_id, action) for tx_id, action in actions.items()}

    def _extract_action(self, tx_id: Id, action) -> Action:
        if tx_id in self.simulator.devices.due_pairs:
            rx_id = self.simulator.devices.due_pairs[tx_id]
            link_type = LinkType.SIDELINK
            rb, tx_pwr_dBm = self._decode_action(action, 'due')
        elif tx_id in self.simulator.devices.cues:
            rx_id = self.simulator.devices.bs.id
            link_type = LinkType.UPLINK
            rb, tx_pwr_dBm = self._decode_action(action, 'cue')
        else:
            rx_id = self.simulator.devices.cues[tx_id]
            link_type = LinkType.DOWNLINK
            rb, tx_pwr_dBm = self._decode_action(action, 'mbs')
        return Action(tx_id, rx_id, link_type, rb, tx_pwr_dBm)

    def _decode_action(self, action: Any, tx_type: str) -> Tuple[int, int]:
        if isinstance(action, (int, np.integer)):
            rb = action // self.num_pwr_actions[tx_type]
            tx_pwr_dBm = action % self.num_pwr_actions[tx_type]
        elif isinstance(action, np.ndarray) and action.ndim == 2:
            rb, tx_pwr_dBm = action
        else:
            raise ValueError(f'Unable to decode action type "{type(action)}"')
        return int(rb), int(tx_pwr_dBm)

    def _info(self, actions: Dict[Id, Action], state: dict):
        return {
            tx_id: {
                'rb': actions[tx_id].rb,
                'tx_pwr_dbm': actions[tx_id].tx_pwr_dBm,
                # 'channel_gains_db': state['channel_gains_db'][(tx_id, rx_id)],
                'snr_db': state['snrs_db'][(tx_id, rx_id)],
                'sinr_db': sinr_db,
                'rate_bps': state['rate_bps'][(tx_id, rx_id)],
                'capacity_mbps': state['capacity_mbps'][(tx_id, rx_id)],
            } for (tx_id, rx_id), sinr_db in state['sinrs_db'].items()}

    def render(self, mode='human'):
        assert self.state is not None, 'Initialise environment with `reset()` before calling `render()`'
        obs = self.obs_fn.get_state(self.state, self.simulator.channels, self.simulator.devices)
        print(obs)

    def save_device_config(self, config_file: Path) -> None:
        """Save the environment's device configuration in a JSON file.

        :param config_file: The filepath to save to.
        """
        config = {device.id: {
                'position': device.position.as_tuple(),
                'config': device.config,
            } for device in self.simulator.devices.values()}
        with config_file.open(mode='w') as fid:
            json.dump(config, fid)

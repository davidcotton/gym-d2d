import json
from pathlib import Path
from typing import Dict, Tuple, Any

import gym
from gym import spaces
import numpy as np

from gym_d2d.actions import Action, Actions
from gym_d2d.envs.obs_fn import LinearObsFunction
from gym_d2d.envs.reward_fn import SystemCapacityRewardFunction
from gym_d2d.id import Id
from gym_d2d.link_type import LinkType
from gym_d2d.simulator import Simulator, BASE_STATION_ID

EPISODE_LENGTH = 10
DEFAULT_OBS_FN = LinearObsFunction
DEFAULT_REWARD_FN = SystemCapacityRewardFunction


class D2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_config=None) -> None:
        super().__init__()
        env_config = env_config or {}
        self.obs_fn = env_config.pop('obs_fn', DEFAULT_OBS_FN)()
        self.reward_fn = env_config.pop('reward_fn', DEFAULT_REWARD_FN)()
        self.simulator = Simulator(env_config)
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
        self.actions = None
        self.state = None
        self.num_steps = 0

    def reset(self):
        self.num_steps = 0
        self.simulator.reset()
        # take a step with random D2D actions to generate initial SINRs
        self.actions = self._reset_random_actions()
        self.state = self.simulator.step(self.actions)
        obs = self.obs_fn.get_state(self.actions, self.state, self.simulator.devices)
        return obs

    def _reset_random_actions(self) -> Actions:
        cue_actions = {
            (tx_id, BASE_STATION_ID): self._extract_action(tx_id, BASE_STATION_ID, self.action_space['cue'].sample())
            for tx_id in self.simulator.devices.cues.keys()}
        due_actions = {tx_rx_id: self._extract_action(*tx_rx_id, self.action_space['due'].sample())
                       for tx_rx_id in self.simulator.devices.dues.keys()}
        return Actions({**cue_actions, **due_actions})

    def step(self, raw_actions: Dict[str, Any]):
        self.actions = self._extract_actions(raw_actions)
        self.state = self.simulator.step(self.actions)
        self.num_steps += 1
        obs = self.obs_fn.get_state(self.actions, self.state, self.simulator.devices)
        rewards = self.reward_fn(self.actions, self.state)
        game_over = {'__all__': self.num_steps >= EPISODE_LENGTH}
        info = self._infos(self.actions, self.state)

        return obs, rewards, game_over, info

    def _extract_actions(self, raw_actions: Dict[str, Any]) -> Actions:
        actions = Actions()
        for id_pair_str, action in raw_actions.items():
            tx_rx_id = tuple([Id(_id) for _id in id_pair_str.split(':')])
            actions[tx_rx_id] = self._extract_action(*tx_rx_id, action)
        return actions

    def _extract_action(self, tx_id: Id, rx_id: Id, action: Any) -> Action:
        if tx_id in self.simulator.devices.due_pairs:
            link_type = LinkType.SIDELINK
            rb, tx_pwr_dBm = self._decode_action(action, 'due')
        elif tx_id in self.simulator.devices.cues:
            link_type = LinkType.UPLINK
            rb, tx_pwr_dBm = self._decode_action(action, 'cue')
        else:
            link_type = LinkType.DOWNLINK
            rb, tx_pwr_dBm = self._decode_action(action, 'mbs')
        tx, rx = self.simulator.devices[tx_id], self.simulator.devices[rx_id]
        return Action(tx, rx, link_type, rb, tx_pwr_dBm)

    def _decode_action(self, action: Any, tx_type: str) -> Tuple[int, int]:
        if isinstance(action, (int, np.integer)):
            rb = action // self.num_pwr_actions[tx_type]
            tx_pwr_dBm = action % self.num_pwr_actions[tx_type]
        elif isinstance(action, np.ndarray) and action.ndim == 2:
            rb, tx_pwr_dBm = action
        else:
            raise ValueError(f'Unable to decode action type "{type(action)}"')
        return int(rb), int(tx_pwr_dBm)

    def _infos(self, actions: Actions, state: dict) -> Dict[str, Any]:
        return {':'.join(id_pair): self._info(action, state) for id_pair, action in actions.items()}

    def _info(self, action: Action, state: dict) -> Dict[str, Any]:
        id_pair = (action.tx.id, action.rx.id)
        return {
            'rb': action.rb,
            'tx_pwr_dbm': action.tx_pwr_dBm,
            # 'channel_gains_db': state['channel_gains_db'][id_pair],
            'snr_db': state['snrs_db'][id_pair],
            'sinr_db': state['sinrs_db'][id_pair],
            'rate_bps': state['rate_bps'][id_pair],
            'capacity_mbps': state['capacity_mbps'][id_pair],
        }

    def render(self, mode='human'):
        assert self.state is not None and self.actions is not None, \
            'Initialise environment with `reset()` before calling `render()`'
        obs = self.obs_fn.get_state(self.actions, self.state, self.simulator.devices)
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

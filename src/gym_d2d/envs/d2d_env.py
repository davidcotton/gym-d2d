import json
from pathlib import Path
from typing import Dict, Tuple, Any

import gym
from gym import spaces
import numpy as np

from .devices import Devices
from .env_config import EnvConfig
from gym_d2d.action import Action
from gym_d2d.device import BaseStation, UserEquipment
from gym_d2d.id import Id
from gym_d2d.link_type import LinkType
from gym_d2d.position import Position, get_random_position, get_random_position_nearby
from gym_d2d.simulator import Simulator


BASE_STATION_ID = 'mbs'
EPISODE_LENGTH = 10


def create_devices(config: EnvConfig) -> Devices:
    """Initialise devices: BSs, CUEs & DUE pairs as per the env config.

    :returns: A dataclass containing BSs, CUEs, and DUE pairs.
    """

    base_cfg = {
        'num_subcarriers': config.num_subcarriers,
        'subcarrier_spacing_kHz': config.subcarrier_spacing_kHz,
    }

    # create macro base station
    cfg = config.devices[BASE_STATION_ID]['config'] if BASE_STATION_ID in config.devices else base_cfg
    bs = BaseStation(Id(BASE_STATION_ID), cfg)

    # create cellular UEs
    cues = {}
    default_cue_cfg = {**base_cfg, **{'max_tx_power_dBm': config.cue_max_tx_power_dBm}}
    for i in range(config.num_cues):
        cue_id = Id(f'cue{i:02d}')
        cfg = config.devices[cue_id]['config'] if cue_id in config.devices else default_cue_cfg
        cues[cue_id] = UserEquipment(cue_id, cfg)

    # create D2D UE pairs
    dues = {}
    due_cfg = {**base_cfg, **{'max_tx_power_dBm': config.due_max_tx_power_dBm}}
    for i in range(0, (config.num_due_pairs * 2), 2):
        due_tx_id, due_rx_id = Id(f'due{i:02d}'), Id(f'due{i + 1:02d}')

        due_tx_cfg = config.devices[due_tx_id]['config'] if due_tx_id in config.devices else due_cfg
        due_tx = UserEquipment(due_tx_id, due_tx_cfg)

        due_rx_cfg = config.devices[due_rx_id]['config'] if due_rx_id in config.devices else due_cfg
        due_rx = UserEquipment(due_rx_id, due_rx_cfg)

        dues[(due_tx.id, due_rx.id)] = due_tx, due_rx

    return Devices(bs, cues, dues)


class D2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_config=None) -> None:
        super().__init__()
        self.config = EnvConfig(**env_config or {})
        devices = create_devices(self.config)
        self.simulator = Simulator(
            devices,
            self.config.traffic_model(devices.bs, list(devices.cues.values()), self.config.num_rbs),
            self.config.path_loss_model(self.config.carrier_freq_GHz)
        )
        self.obs_fn = self.config.obs_fn(self.simulator)
        self.observation_space = self.obs_fn.get_obs_space(self.config.__dict__)
        self.num_pwr_actions = {  # +1 because include max value, i.e. from [0, ..., max]
            'due': self.config.due_max_tx_power_dBm - self.config.due_min_tx_power_dBm + 1,
            'cue': self.config.cue_max_tx_power_dBm + 1,
            'mbs': self.config.mbs_max_tx_power_dBm + 1
        }
        self.action_space = spaces.Dict({
            'due': spaces.Discrete(self.config.num_rbs * self.num_pwr_actions['due']),
            'cue': spaces.Discrete(self.config.num_rbs * self.num_pwr_actions['cue']),
            'mbs': spaces.Discrete(self.config.num_rbs * self.num_pwr_actions['mbs']),
        })
        self.reward_fn = self.config.reward_fn(self.simulator)
        self.num_steps = 0

    def reset(self):
        self.num_steps = 0
        for device in self.simulator.devices.values():
            if device.id == BASE_STATION_ID:
                pos = Position(0, 0)  # assume MBS fixed at (0,0) and everything else builds around it
            elif device.id in self.config.devices:
                pos = Position(*self.config.devices[device.id]['position'])
            elif any(device.id in d for d in [self.simulator.devices.cues, self.simulator.devices.due_pairs]):
                pos = get_random_position(self.config.cell_radius_m)
            elif device.id in self.simulator.devices.due_pairs_inv:
                due_tx_id = self.simulator.devices.due_pairs_inv[device.id]
                due_tx = self.simulator.devices[due_tx_id]
                pos = get_random_position_nearby(self.config.cell_radius_m, due_tx.position, self.config.d2d_radius_m)
            else:
                raise ValueError(f'Invalid configuration for device "{device.id}".')
            device.set_position(pos)

        self.simulator.reset()
        # take a step with random D2D actions to generate initial SINRs
        random_actions = self._reset_random_actions()
        results = self.simulator.step(random_actions)
        obs = self.obs_fn.get_state(results)
        return obs

    def _reset_random_actions(self) -> Dict[Id, Action]:
        cue_actions = {_id: self._extract_action(_id, self.action_space['cue'].sample())
                       for _id in self.simulator.devices.cues.keys()}
        due_actions = {_id: self._extract_action(_id, self.action_space['due'].sample())
                       for _id, _ in self.simulator.devices.dues.keys()}
        return {**cue_actions, **due_actions}

    def step(self, actions):
        actions = self._extract_actions(actions)
        results = self.simulator.step(actions)
        self.num_steps += 1
        obs = self.obs_fn.get_state(results)
        rewards = self.reward_fn(results)
        game_over = {'__all__': self.num_steps >= EPISODE_LENGTH}
        info = self._info(actions, results)

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

    def _info(self, actions: Dict[Id, Action], results: dict):
        return {
            tx_id: {
                'rb': actions[tx_id].rb,
                'tx_pwr_dbm': actions[tx_id].tx_pwr_dBm,
                # 'channel_gains_db': results['channel_gains_db'][(tx_id, rx_id)],
                'snr_db': results['snrs_db'][(tx_id, rx_id)],
                'sinr_db': sinr_db,
                'rate_bps': results['rate_bps'][(tx_id, rx_id)],
                'capacity_mbps': results['capacity_mbps'][(tx_id, rx_id)],
            } for (tx_id, rx_id), sinr_db in results['sinrs_db'].items()}

    def render(self, mode='human'):
        obs = self.obs_fn.get_state({})  # @todo need to find a way to handle SINRs here
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

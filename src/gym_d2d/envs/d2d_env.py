import json
from pathlib import Path
from typing import Dict, List, Tuple

import gym
from gym import spaces
import numpy as np

from .action import Action
from .id import Id
from .mode import Mode
from .path_loss import FreeSpacePathLoss
from .position import Position, get_random_position, get_random_position_nearby
from .simulator import D2DSimulator
from .traffic_model import SimplexTrafficModel


DEFAULT_CUE_MAX_TX_POWER_DBM = 23
DEFAULT_DUE_MAX_TX_POWER_DBM = 23
DEFAULT_CARRIER_FREQ_GHZ = 2.1
DEFAULT_NUM_RESOURCE_BLOCKS = 30
DEFAULT_NUM_SMALL_BASE_STATIONS = 0
DEFAULT_NUM_CELLULAR_USERS = 30
DEFAULT_NUM_D2D_PAIRS = 12
DEFAULT_CELL_RADIUS_M = 250.0
DEFAULT_D2D_RADIUS_M = 20.0
MACRO_BASE_STATION_ID = 'mbs'


class D2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_config=None) -> None:
        super().__init__()
        env_config = env_config or {}
        self.env_config = dict({
            'cue_max_tx_power_dBm': DEFAULT_CUE_MAX_TX_POWER_DBM,
            'due_max_tx_power_dBm': DEFAULT_DUE_MAX_TX_POWER_DBM,
            'carrier_freq_GHz': DEFAULT_CARRIER_FREQ_GHZ,
            'num_rbs': DEFAULT_NUM_RESOURCE_BLOCKS,
            'num_small_base_stations': DEFAULT_NUM_SMALL_BASE_STATIONS,
            'num_cellular_users': DEFAULT_NUM_CELLULAR_USERS,
            'num_d2d_pairs': DEFAULT_NUM_D2D_PAIRS,
            'cell_radius_m': DEFAULT_CELL_RADIUS_M,
            'd2d_radius_m': DEFAULT_D2D_RADIUS_M,
        }, **env_config)
        self.device_config = self._load_device_config(env_config)

        self.simulator = D2DSimulator(FreeSpacePathLoss(self.carrier_freq_GHz), self.num_rbs)
        self.bses, self.cues, self.due_pairs = self._create_devices()
        self.due_pairs_inv = {v: k for k, v in self.due_pairs.items()}

        num_txs = self.num_cellular_users + self.num_d2d_pairs
        num_rxs = 1 + self.num_d2d_pairs  # basestation + num D2D rxs
        num_tx_obs = 5  # sinrs, tx_pwrs, rbs, tx_pos_x, tx_pos_y
        num_rx_obs = 2  # rx_pos_x, rx_pos_y
        obs_shape = ((num_txs * num_tx_obs) + (num_rxs * num_rx_obs),)
        self.observation_space = spaces.Box(low=-self.cell_radius_m, high=self.cell_radius_m, shape=obs_shape)
        self.action_space = spaces.Discrete(self.num_rbs * self.due_max_tx_power_dBm)

    def _create_devices(self) -> Tuple[List[Id], List[Id], Dict[Id, Id]]:
        """Initialise small base stations, cellular UE & D2D UE pairs in the simulator as per the env config.

        :returns: A tuple containing a list of base station, CUE & a dict of DUE pair IDs created.
        """
        # create macro base station
        macro_bs = self.simulator.add_base_station(Id(MACRO_BASE_STATION_ID), {})
        bses = [macro_bs]
        # create small base stations
        for i in range(self.num_small_base_stations):
            sbs_id = Id(f'sbs{i:02d}')
            config = self.device_config[sbs_id]['config'] if sbs_id in self.device_config else {}
            self.simulator.add_base_station(sbs_id, config)
            bses.append(sbs_id)

        # create cellular UEs
        cues = []
        default_cue_cfg = {'max_tx_power_dBm': self.cue_max_tx_power_dBm}
        for i in range(self.num_cellular_users):
            cue_id = Id(f'cue{i:02d}')
            config = self.device_config[cue_id]['config'] if cue_id in self.device_config else default_cue_cfg
            cue = self.simulator.add_ue(cue_id, config)
            cues.append(cue_id)
            traffic_model = SimplexTrafficModel([cue, macro_bs])
            self.simulator.add_traffic_model(traffic_model)

        # create D2D UEs
        due_pairs = {}
        def_due_cfg = {'max_tx_power_dBm': self.due_max_tx_power_dBm}
        for i in range(0, (self.num_d2d_pairs * 2), 2):
            due_tx_id, due_rx_id = Id(f'due{i:02d}'), Id(f'due{i + 1:02d}')
            due_tx_config = self.device_config[due_tx_id]['config'] if due_tx_id in self.device_config else def_due_cfg
            self.simulator.add_ue(due_tx_id, due_tx_config)

            due_rx_config = self.device_config[due_rx_id]['config'] if due_rx_id in self.device_config else def_due_cfg
            self.simulator.add_ue(due_rx_id, due_rx_config)
            due_pairs[due_tx_id] = due_rx_id

        return bses, cues, due_pairs

    def reset(self):
        for device in self.simulator.devices.values():
            if device.id == MACRO_BASE_STATION_ID:
                pos = Position(0, 0)  # assume MBS fixed at (0,0) and everything else builds around it
            elif device.id in self.device_config:
                pos = Position(*self.device_config[device.id]['position'])
            elif any(device.id in d for d in [self.bses, self.cues, self.due_pairs]):
                pos = get_random_position(self.cell_radius_m)
            elif device.id in self.due_pairs_inv:
                due_tx_id = self.due_pairs_inv[device.id]
                due_tx = self.simulator.devices[due_tx_id]
                pos = get_random_position_nearby(self.cell_radius_m, due_tx.position, self.d2d_radius_m)
            else:
                raise ValueError(f'Invalid configuration for device "{device.id}".')
            device.set_position(pos)

        random_actions = {due_id: self._extract_action(due_id, self.action_space.sample())
                          for due_id in self.due_pairs.keys()}
        results = self.simulator.reset(random_actions)
        obs = self._get_state(results['sinrs'])
        return obs

    def step(self, actions):
        due_actions = {due_id: self._extract_action(due_id, action_idx) for due_id, action_idx in actions.items()}
        results = self.simulator.step(due_actions)
        obs = self._get_state(results['sinrs'])
        rewards = {}
        return obs, rewards, False, {}

    def _extract_action(self, due_tx_id: Id, action_idx: int) -> Action:
        rb = action_idx // self.due_max_tx_power_dBm
        tx_pwr = action_idx % self.due_max_tx_power_dBm
        tx = self.simulator.ues[due_tx_id]
        rx = self.simulator.ues[self.due_pairs[due_tx_id]]
        return Action(tx.id, rx.id, Mode.D2D_UNDERLAY, rb, tx_pwr)

    def render(self, mode='human'):
        obs = self._get_state()
        print(obs)

    def _get_state(self, sinrs: Dict[Id, float]):
        obs_dict = {}
        for due_id in self.due_pairs.keys():
            tx_pwrs_dBm = []
            rbs = []
            positions = []
            for channel in self.simulator.channels.values():
                positions.extend(list(channel.tx.position.as_tuple()))
                tx_pwrs_dBm.append(channel.tx_pwr_dBm)
                rbs.append(channel.rb)
            for channel in self.simulator.channels.values():
                positions.extend(list(channel.rx.position.as_tuple()))

            obs = list(sinrs.values())
            obs.extend(tx_pwrs_dBm)
            obs.extend(rbs)
            obs.extend(positions)
            obs_dict[due_id] = np.array(obs)

        return obs_dict

    def save_device_config(self, config_file: Path) -> None:
        """Save the environment's device configuration in a JSON file.

        :param config_file: The filepath to save to.
        """
        config = {}
        for device in self.simulator.devices.values():
            config[device.id] = {
                'position': device.position.as_tuple(),
                'config': device.config,
            }
        with config_file.open(mode='w') as fid:
            json.dump(config, fid)

    @staticmethod
    def _load_device_config(env_config: Dict) -> dict:
        if 'device_config_file' in env_config and isinstance(env_config['device_config_file'], Path):
            with env_config['device_config_file'].open(mode='r') as fid:
                return json.load(fid)
        else:
            return {}

    @property
    def cue_max_tx_power_dBm(self) -> int:
        return self.env_config['cue_max_tx_power_dBm']

    @property
    def due_max_tx_power_dBm(self) -> int:
        return self.env_config['due_max_tx_power_dBm']

    @property
    def carrier_freq_GHz(self) -> float:
        return self.env_config['carrier_freq_GHz']

    @property
    def num_rbs(self) -> int:
        return self.env_config['num_rbs']

    @property
    def num_small_base_stations(self) -> int:
        return self.env_config['num_small_base_stations']

    @property
    def num_cellular_users(self) -> int:
        return self.env_config['num_cellular_users']

    @property
    def num_d2d_pairs(self) -> int:
        return self.env_config['num_d2d_pairs']

    @property
    def cell_radius_m(self) -> float:
        return self.env_config['cell_radius_m']

    @property
    def d2d_radius_m(self) -> float:
        return self.env_config['d2d_radius_m']

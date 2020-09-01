from collections import defaultdict
import json
from math import log2
from pathlib import Path
from typing import Dict, List, Tuple, Type

import gym
from gym import spaces
import numpy as np

from gym_d2d.action import Action
from gym_d2d.conversion import dB_to_linear
from gym_d2d.id import Id
from gym_d2d.link_type import LinkType
from gym_d2d.path_loss import PathLoss, FreeSpacePathLoss
from gym_d2d.position import Position, get_random_position, get_random_position_nearby
from gym_d2d.simulator import D2DSimulator
from gym_d2d.traffic_model import SimplexTrafficModel


DEFAULT_CARRIER_FREQ_GHZ = 2.1
DEFAULT_NUM_SUBCARRIERS = 12
DEFAULT_SUBCARRIER_SPACING_KHZ = 15
DEFAULT_CHANNEL_BANDWIDTH_MHZ = 20.0
DEFAULT_NUM_RESOURCE_BLOCKS = 30
DEFAULT_PATH_LOSS_MODEL = FreeSpacePathLoss
DEFAULT_NUM_SMALL_BASE_STATIONS = 0
DEFAULT_NUM_CELLULAR_USERS = 30
DEFAULT_NUM_D2D_PAIRS = 12
DEFAULT_CELL_RADIUS_M = 250.0
DEFAULT_D2D_RADIUS_M = 20.0
DEFAULT_CUE_MAX_TX_POWER_DBM = 23
DEFAULT_DUE_MAX_TX_POWER_DBM = 23
MACRO_BASE_STATION_ID = 'mbs'


class D2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_config=None) -> None:
        super().__init__()
        env_config = env_config or {}
        self.env_config = dict({
            'carrier_freq_GHz': DEFAULT_CARRIER_FREQ_GHZ,
            'num_subcarriers': DEFAULT_NUM_SUBCARRIERS,
            'subcarrier_spacing_kHz': DEFAULT_SUBCARRIER_SPACING_KHZ,
            'channel_bandwidth_MHz': DEFAULT_CHANNEL_BANDWIDTH_MHZ,
            'num_rbs': DEFAULT_NUM_RESOURCE_BLOCKS,
            'path_loss_model': DEFAULT_PATH_LOSS_MODEL,
            'num_small_base_stations': DEFAULT_NUM_SMALL_BASE_STATIONS,
            'num_cellular_users': DEFAULT_NUM_CELLULAR_USERS,
            'num_d2d_pairs': DEFAULT_NUM_D2D_PAIRS,
            'cell_radius_m': DEFAULT_CELL_RADIUS_M,
            'd2d_radius_m': DEFAULT_D2D_RADIUS_M,
            'cue_max_tx_power_dBm': DEFAULT_CUE_MAX_TX_POWER_DBM,
            'due_max_tx_power_dBm': DEFAULT_DUE_MAX_TX_POWER_DBM,
        }, **env_config)
        self.device_config = self._load_device_config(env_config)

        self.simulator = D2DSimulator(self.path_loss_model(self.carrier_freq_GHz), self.channel_bandwidth_MHz, self.num_rbs)
        self.bses, self.cues, self.due_pairs = self._create_devices()
        self.due_pairs_inv = {v: k for k, v in self.due_pairs.items()}

        num_txs = self.num_cellular_users + self.num_d2d_pairs

        # get_state1
        # num_rxs = 1 + self.num_d2d_pairs  # basestation + num D2D rxs
        num_tx_obs = 5  # sinrs, tx_pwrs, rbs, xs, ys
        num_rx_obs = 2  # xs, ys
        # obs_shape = ((num_txs * num_tx_obs) + (num_rxs * num_rx_obs),)
        obs_shape = ((num_txs * num_tx_obs) + (num_txs * num_rx_obs),)

        # get_state2
        # num_due_obs = 2  # x, y
        # num_common_obs = 7  # tx_x, tx_y, rx_x, rx_y, tx_pwr, rb, sinr
        # obs_shape = (num_due_obs + (num_common_obs * num_txs),)

        self.observation_space = spaces.Box(low=-self.cell_radius_m, high=self.cell_radius_m, shape=obs_shape)
        num_tx_pwr_actions = self.due_max_tx_power_dBm + 1  # include max value, i.e. from [0, ..., max]
        self.action_space = spaces.Discrete(self.num_rbs * num_tx_pwr_actions)

    def _create_devices(self) -> Tuple[List[Id], List[Id], Dict[Id, Id]]:
        """Initialise small base stations, cellular UE & D2D UE pairs in the simulator as per the env config.

        :returns: A tuple containing a list of base station, CUE & a dict of DUE pair IDs created.
        """

        base_cfg = {
            'num_subcarriers': self.num_subcarriers,
            'subcarrier_spacing_kHz': self.subcarrier_spacing_kHz,
        }

        # create macro base station
        macro_bs = self.simulator.add_base_station(Id(MACRO_BASE_STATION_ID), {})
        bses = [macro_bs]
        # create small base stations
        for i in range(self.num_small_base_stations):
            sbs_id = Id(f'sbs{i:02d}')
            config = self.device_config[sbs_id]['config'] if sbs_id in self.device_config else base_cfg
            self.simulator.add_base_station(sbs_id, config)
            bses.append(sbs_id)

        # create cellular UEs
        cues = []
        default_cue_cfg = {**base_cfg, **{'max_tx_power_dBm': self.cue_max_tx_power_dBm}}
        for i in range(self.num_cellular_users):
            cue_id = Id(f'cue{i:02d}')
            config = self.device_config[cue_id]['config'] if cue_id in self.device_config else default_cue_cfg
            cue = self.simulator.add_ue(cue_id, config)
            cues.append(cue_id)
            traffic_model = SimplexTrafficModel([cue, macro_bs])
            self.simulator.add_traffic_model(traffic_model)

        # create D2D UEs
        due_pairs = {}
        def_due_cfg = {**base_cfg, **{'max_tx_power_dBm': self.due_max_tx_power_dBm}}
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

        self.simulator.reset()
        # take a step with random D2D actions to generate initial SINRs
        random_actions = {due_id: self._extract_action(due_id, self.action_space.sample())
                          for due_id in self.due_pairs.keys()}
        results = self.simulator.step(random_actions)
        obs = self._get_state(results['SINRs_dB'])
        return obs

    def step(self, actions):
        due_actions = {due_id: self._extract_action(due_id, action_idx) for due_id, action_idx in actions.items()}
        results = self.simulator.step(due_actions)
        obs = self._get_state(results['SINRs_dB'])
        rewards = self._calculate_rewards(results)

        info = {}
        num_cues = 0
        sum_cue_sinr, sum_cue_capacity, system_capacity = 0.0, 0.0, 0.0
        system_sum_rate_bps = 0.0
        for ((tx_id, rx_id), sinr_dB), capacity in zip(results['SINRs_dB'].items(), results['capacity_Mbps'].values()):
            system_capacity += capacity
            system_sum_rate_bps += results['sum_rate_bps'][(tx_id, rx_id)]
            if tx_id in self.due_pairs:
                info[tx_id] = {
                    'rb': due_actions[tx_id].rb,
                    'tx_pwr_dBm': due_actions[tx_id].tx_pwr_dBm,
                    'DUE_SINR_dB': sinr_dB,
                    'DUE_capacity_Mbps': capacity,
                    'total_DUE_sum_rate_bps': results['sum_rate_bps'][(tx_id, rx_id)]
                }
            else:
                num_cues += 1
                sum_cue_sinr += sinr_dB
                sum_cue_capacity += capacity
        info['__env__'] = {
            'mean_CUE_SINR_dB': sum_cue_sinr / num_cues,
            'CUE_capacity_Mbps': sum_cue_capacity,
            'system_capacity_Mbps': system_capacity,
            'system_sum_rate_bps': system_sum_rate_bps,
        }

        return obs, rewards, {'__all__': False}, info

    def _extract_action(self, due_tx_id: Id, action_idx: int) -> Action:
        rb = action_idx % self.num_rbs
        tx_pwr_dBm = action_idx // self.num_rbs
        return Action(due_tx_id, self.due_pairs[due_tx_id], LinkType.SIDELINK, rb, tx_pwr_dBm)

    def _calculate_rewards(self, results: dict) -> dict:
        # return self._rewards_capacity(results)
        return self._rewards_shannon(results)

    def _rewards_capacity(self, results: dict):
        # group by RB
        rbs = defaultdict(set)
        for ids, channel in self.simulator.channels.items():
            rbs[channel.rb].add(ids)

        reward = -1
        brake = False
        for tx_id, rx_id in self.due_pairs.items():
            if brake:
                break
            rb = self.simulator.channels[(tx_id, rx_id)].rb
            ix_channels = rbs[rb].difference({(tx_id, rx_id)})
            for ix_tx_id, ix_rx_id in ix_channels:
                if ix_tx_id in self.due_pairs:
                    continue
                if results['capacity_Mbps'][(ix_tx_id, ix_rx_id)] <= 0:
                    brake = True
                    break
        else:
            sum_capacity = sum(c for c in results['capacity_Mbps'].values())
            reward = sum_capacity / len(self.due_pairs)

        rewards = {}
        for tx_id, rx_id in self.due_pairs.items():
            rewards[tx_id] = reward
        return rewards

    def _rewards_shannon(self, results: dict):
        rewards = {}
        for ids in self.due_pairs.items():
            sinr = results['SINRs_dB'][ids]
            if sinr < 0:
                rewards[ids] = log2(1 + dB_to_linear(sinr))
            else:
                rewards[ids] = -1
        return rewards

    def render(self, mode='human'):
        obs = self._get_state({})  # @todo need to find a way to handle SINRs here
        print(obs)

    def _get_state(self, sinrs: Dict[Tuple[Id, Id], float]):
        return self._get_state1(sinrs)
        # return self._get_state2(sinrs)

    def _get_state1(self, sinrs: Dict[Tuple[Id, Id], float]):
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
        common_obs.extend(list(sinrs.values()))
        common_obs.extend(tx_pwrs_dBm)
        common_obs.extend(rbs)
        common_obs.extend(positions)

        return {due_id: np.array(common_obs) for due_id in self.due_pairs.keys()}

    def _get_state2(self, sinrs: Dict[Tuple[Id, Id], float]):
        common_obs = []
        for channel in self.simulator.channels.values():
            common_obs.extend(list(channel.tx.position.as_tuple()))
            common_obs.extend(list(channel.rx.position.as_tuple()))
            common_obs.append(channel.tx_pwr_dBm)
            common_obs.append(channel.rb)
            common_obs.append(sinrs[(channel.tx.id, channel.rx.id)])

        obs_dict = {}
        for due_id in self.due_pairs.keys():
            due_pos = list(self.simulator.devices[due_id].position.as_tuple())
            due_obs = []
            due_obs.extend(due_pos)
            due_obs.extend(common_obs)
            obs_dict[due_id] = np.array(due_obs)

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
    def carrier_freq_GHz(self) -> float:
        return float(self.env_config['carrier_freq_GHz'])

    @property
    def num_subcarriers(self) -> int:
        return int(self.env_config['num_subcarriers'])

    @property
    def subcarrier_spacing_kHz(self) -> int:
        return int(self.env_config['subcarrier_spacing_kHz'])

    @property
    def channel_bandwidth_MHz(self) -> float:
        return float(self.env_config['channel_bandwidth_MHz'])

    @property
    def num_rbs(self) -> int:
        return int(self.env_config['num_rbs'])

    @property
    def path_loss_model(self) -> Type[PathLoss]:
        return self.env_config['path_loss_model']

    @property
    def num_small_base_stations(self) -> int:
        return int(self.env_config['num_small_base_stations'])

    @property
    def num_cellular_users(self) -> int:
        return int(self.env_config['num_cellular_users'])

    @property
    def num_d2d_pairs(self) -> int:
        return int(self.env_config['num_d2d_pairs'])

    @property
    def cell_radius_m(self) -> float:
        return float(self.env_config['cell_radius_m'])

    @property
    def d2d_radius_m(self) -> float:
        return float(self.env_config['d2d_radius_m'])

    @property
    def cue_max_tx_power_dBm(self) -> int:
        return int(self.env_config['cue_max_tx_power_dBm'])

    @property
    def due_max_tx_power_dBm(self) -> int:
        return int(self.env_config['due_max_tx_power_dBm'])

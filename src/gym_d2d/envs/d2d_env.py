from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import gym
from gym import spaces
import numpy as np

from .devices import Devices
from .reward_functions import RewardFunction, SystemCapacityRewardFunction
from gym_d2d.action import Action
from gym_d2d.device import BaseStation, UserEquipment
from gym_d2d.id import Id
from gym_d2d.link_type import LinkType
from gym_d2d.path_loss import PathLoss, FreeSpacePathLoss
from gym_d2d.position import Position, get_random_position, get_random_position_nearby
from gym_d2d.simulator import D2DSimulator
from gym_d2d.traffic_model import TrafficModel, UplinkTrafficModel


BASE_STATION_ID = 'mbs'


@dataclass
class EnvConfig:
    num_rbs: int = 30
    num_cellular_users: int = 30
    num_d2d_pairs: int = 12
    num_small_base_stations: int = 0
    cell_radius_m: float = 500.0
    d2d_radius_m: float = 20.0
    cue_max_tx_power_dBm: int = 23
    due_min_tx_power_dBm: int = 0
    due_max_tx_power_dBm: int = 23
    path_loss_model: Type[PathLoss] = FreeSpacePathLoss
    traffic_model: Type[TrafficModel] = UplinkTrafficModel
    carrier_freq_GHz: float = 2.1
    num_subcarriers: int = 12
    subcarrier_spacing_kHz: int = 15
    channel_bandwidth_MHz: float = 20.0
    reward_function: RewardFunction = SystemCapacityRewardFunction()
    device_config_file: Optional[Path] = None

    def __post_init__(self):
        self.devices = self.load_device_config()

    def load_device_config(self) -> dict:
        if isinstance(self.device_config_file, Path):
            with self.device_config_file.open(mode='r') as fid:
                return json.load(fid)
        else:
            return {}


class D2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_config=None) -> None:
        super().__init__()
        self.config = EnvConfig(**env_config or {})
        self.devices = self._create_devices()
        traffic_model = self.config.traffic_model(self.devices.bs, list(self.devices.cues.values()), self.config.num_rbs)
        path_loss = self.config.path_loss_model(self.config.carrier_freq_GHz)
        self.simulator = D2DSimulator(self.devices.to_dict(), traffic_model, path_loss)

        num_txs = self.config.num_cellular_users + self.config.num_d2d_pairs

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

        self.observation_space = spaces.Box(low=-self.config.cell_radius_m, high=self.config.cell_radius_m, shape=obs_shape)
        num_tx_pwr_actions = self.config.due_max_tx_power_dBm - self.config.due_min_tx_power_dBm + 1  # include max value, i.e. from [0, ..., max]
        self.action_space = spaces.Discrete(self.config.num_rbs * num_tx_pwr_actions)

    def _create_devices(self) -> Devices:
        """Initialise small base stations, cellular UE & D2D UE pairs in the simulator as per the env config.

        :returns: A tuple containing a list of base station, CUE & a dict of DUE pair IDs created.
        """

        base_cfg = {
            'num_subcarriers': self.config.num_subcarriers,
            'subcarrier_spacing_kHz': self.config.subcarrier_spacing_kHz,
        }

        # create macro base station
        config = self.config.devices[BASE_STATION_ID]['config'] if BASE_STATION_ID in self.config.devices else base_cfg
        bs = BaseStation(Id(BASE_STATION_ID), config)

        # create cellular UEs
        cues = {}
        default_cue_cfg = {**base_cfg, **{'max_tx_power_dBm': self.config.cue_max_tx_power_dBm}}
        for i in range(self.config.num_cellular_users):
            cue_id = Id(f'cue{i:02d}')
            config = self.config.devices[cue_id]['config'] if cue_id in self.config.devices else default_cue_cfg
            cues[cue_id] = UserEquipment(cue_id, config)

        # create D2D UEs
        dues = {}
        due_cfg = {**base_cfg, **{'max_tx_power_dBm': self.config.due_max_tx_power_dBm}}
        for i in range(0, (self.config.num_d2d_pairs * 2), 2):
            due_tx_id, due_rx_id = Id(f'due{i:02d}'), Id(f'due{i + 1:02d}')

            due_tx_config = self.config.devices[due_tx_id]['config'] if due_tx_id in self.config.devices else due_cfg
            due_tx = UserEquipment(due_tx_id, due_tx_config)

            due_rx_config = self.config.devices[due_rx_id]['config'] if due_rx_id in self.config.devices else due_cfg
            due_rx = UserEquipment(due_rx_id, due_rx_config)

            dues[(due_tx.id, due_rx.id)] = due_tx, due_rx

        return Devices(bs, cues, dues)

    def reset(self):
        for device in self.simulator.devices.values():
            if device.id == BASE_STATION_ID:
                pos = Position(0, 0)  # assume MBS fixed at (0,0) and everything else builds around it
            elif device.id in self.config.devices:
                pos = Position(*self.config.devices[device.id]['position'])
            elif any(device.id in d for d in [self.devices.cues, self.devices.due_pairs]):
                pos = get_random_position(self.config.cell_radius_m)
            elif device.id in self.devices.due_pairs_inv:
                due_tx_id = self.devices.due_pairs_inv[device.id]
                due_tx = self.simulator.devices[due_tx_id]
                pos = get_random_position_nearby(self.config.cell_radius_m, due_tx.position, self.config.d2d_radius_m)
            else:
                raise ValueError(f'Invalid configuration for device "{device.id}".')
            device.set_position(pos)

        self.simulator.reset()
        # take a step with random D2D actions to generate initial SINRs
        random_actions = {due_id: self._extract_action(due_id, self.action_space.sample())
                          for due_id in self.devices.due_pairs.keys()}
        results = self.simulator.step(random_actions)
        obs = self._get_state(results['SINRs_dB'])
        return obs

    def step(self, actions):
        due_actions = {due_id: self._extract_action(due_id, action_idx) for due_id, action_idx in actions.items()}
        results = self.simulator.step(due_actions)
        obs = self._get_state(results['SINRs_dB'])
        rewards = self.config.reward_function(self.simulator, self.devices, results)

        info = {}
        num_cues = 0
        sum_cue_sinr, sum_cue_capacity, system_capacity = 0.0, 0.0, 0.0
        system_sum_rate_bps = 0.0
        for ((tx_id, rx_id), sinr_dB), capacity in zip(results['SINRs_dB'].items(), results['capacity_Mbps'].values()):
            system_capacity += capacity
            system_sum_rate_bps += results['sum_rate_bps'][(tx_id, rx_id)]
            if tx_id in self.devices.due_pairs:
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
        rb = action_idx % self.config.num_rbs
        tx_pwr_dBm = (action_idx // self.config.num_rbs) + self.config.due_min_tx_power_dBm
        return Action(due_tx_id, self.devices.due_pairs[due_tx_id], LinkType.SIDELINK, rb, tx_pwr_dBm)

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

        return {due_id: np.array(common_obs) for due_id in self.devices.due_pairs.keys()}

    def _get_state2(self, sinrs: Dict[Tuple[Id, Id], float]):
        common_obs = []
        for channel in self.simulator.channels.values():
            common_obs.extend(list(channel.tx.position.as_tuple()))
            common_obs.extend(list(channel.rx.position.as_tuple()))
            common_obs.append(channel.tx_pwr_dBm)
            common_obs.append(channel.rb)
            common_obs.append(sinrs[(channel.tx.id, channel.rx.id)])

        obs_dict = {}
        for due_id in self.devices.due_pairs.keys():
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

import json
from pathlib import Path
from typing import Dict

import gym
from gym import spaces

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
        self.devices = create_devices(self.config)
        self.simulator = Simulator(
            self.devices,
            self.config.traffic_model(self.devices.bs, list(self.devices.cues.values()), self.config.num_rbs),
            self.config.path_loss_model(self.config.carrier_freq_GHz)
        )

        self.obs_fn = self.config.obs_fn(self.simulator, self.devices)
        self.observation_space = self.obs_fn.get_obs_space(self.config.__dict__)
        # +1 because include max value, i.e. from [0, ..., max]
        num_tx_pwr_actions = self.config.due_max_tx_power_dBm - self.config.due_min_tx_power_dBm + 1
        self.action_space = spaces.Discrete(self.config.num_rbs * num_tx_pwr_actions)
        self.reward_fn = self.config.reward_fn(self.simulator, self.devices)
        self.num_steps = 0

    def reset(self):
        self.num_steps = 0
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
        obs = self.obs_fn.get_state(results)
        return obs

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
        return {due_id: self._extract_action(due_id, int(action_idx)) for due_id, action_idx in actions.items()}

    def _extract_action(self, due_tx_id: Id, action: int) -> Action:
        rb = action % self.config.num_rbs
        tx_pwr_dBm = (action // self.config.num_rbs) + self.config.due_min_tx_power_dBm
        return Action(due_tx_id, self.devices.due_pairs[due_tx_id], LinkType.SIDELINK, rb, tx_pwr_dBm)

    def _info(self, actions: Dict[Id, Action], results: dict):
        info = {}
        sum_cue_sinr, sum_system_sinr = 0.0, 0.0
        sum_cue_rate_bps, sum_due_rate_bps, sum_system_rate_bps = 0.0, 0.0, 0.0
        sum_cue_capacity, sum_due_capacity, sum_system_capacity = 0.0, 0.0, 0.0
        for ((tx_id, rx_id), sinr_db), capacity in zip(results['sinrs_db'].items(), results['capacity_mbps'].values()):
            sum_system_sinr += sinr_db
            sum_system_rate_bps += results['rate_bps'][(tx_id, rx_id)]
            sum_system_capacity += capacity
            if tx_id in self.devices.due_pairs:
                info[tx_id] = {
                    'rb': actions[tx_id].rb,
                    'tx_pwr_dbm': actions[tx_id].tx_pwr_dBm,
                    'due_sinr_db': sinr_db,
                    'due_rate_bps': results['rate_bps'][(tx_id, rx_id)],
                    'due_capacity_mbps': capacity,
                }
                sum_due_rate_bps += results['rate_bps'][(tx_id, rx_id)]
                sum_due_capacity += capacity
            else:
                sum_cue_sinr += sinr_db
                sum_cue_rate_bps += results['rate_bps'][(tx_id, rx_id)]
                sum_cue_capacity += capacity

        aggregate_info = {
            'env_mean_cue_sinr_db': sum_cue_sinr / len(self.devices.cues),
            'env_mean_system_sinr_db': sum_system_sinr / (len(self.devices.cues) + len(self.devices.due_pairs)),
            'env_sum_cue_rate_bps': sum_cue_rate_bps,
            'env_sum_due_rate_bps': sum_due_rate_bps,
            'env_sum_system_rate_bps': sum_system_rate_bps,
            'env_sum_cue_capacity_mbps': sum_cue_capacity,
            'env_sum_due_capacity_mbps': sum_due_capacity,
            'env_sum_system_capacity_mbps': sum_system_capacity,
        }
        if self.config.compressed_info:
            for tx_id, tx_info in info.items():
                tx_info.update(aggregate_info)
        else:
            info['__env__'] = aggregate_info

        return info

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

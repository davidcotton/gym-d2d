import gym
from gym import spaces
import numpy as np


DEFAULT_CUE_MAX_TX_POWER_DBM = 23
DEFAULT_DUE_MAX_TX_POWER_DBM = DEFAULT_CUE_MAX_TX_POWER_DBM
DEFAULT_NUM_RESOURCE_BLOCKS = 30
DEFAULT_NUM_BASE_STATIONS = 1
DEFAULT_NUM_CELLULAR_USERS = 30
DEFAULT_NUM_D2D_PAIRS = 12
DEFAULT_CELL_RADIUS_M = 250
DEFAULT_D2D_RADIUS_M = 20


class D2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_config=None) -> None:
        super().__init__()
        env_config = env_config or {}
        self.env_config = dict({
            'cue_max_tx_power_dBm': DEFAULT_CUE_MAX_TX_POWER_DBM,
            'due_max_tx_power_dBm': DEFAULT_DUE_MAX_TX_POWER_DBM,
            'num_rbs': DEFAULT_NUM_RESOURCE_BLOCKS,
            'num_base_stations': DEFAULT_NUM_BASE_STATIONS,
            'num_cellular_users': DEFAULT_NUM_CELLULAR_USERS,
            'num_d2d_pairs': DEFAULT_NUM_D2D_PAIRS,
            'cell_radius_m': DEFAULT_CELL_RADIUS_M,
            'd2d_radius_m': DEFAULT_D2D_RADIUS_M,
        }, **env_config)

        num_txs = self.num_cellular_users + self.num_d2d_pairs
        num_rxs = 1 + self.num_d2d_pairs  # basestation + num D2D rxs
        num_tx_obs = 5  # sinrs, tx_pwrs, rbs, tx_pos_x, tx_pos_y
        num_rx_obs = 2  # rx_pos_x, rx_pos_y
        self.obs_shape = ((num_txs * num_tx_obs) + (num_rxs * num_rx_obs),)
        self.obs_min = -self.cell_radius_m
        self.obs_max = self.cell_radius_m
        self.observation_space = spaces.Box(low=self.obs_min, high=self.obs_max, shape=self.obs_shape)
        self.action_space = spaces.Discrete(self.num_rbs * self.due_max_tx_power_dBm)

    @property
    def cue_max_tx_power_dBm(self) -> int:
        return self.env_config['cue_max_tx_power_dBm']

    @property
    def due_max_tx_power_dBm(self) -> int:
        return self.env_config['due_max_tx_power_dBm']

    @property
    def num_rbs(self) -> int:
        return self.env_config['num_rbs']

    @property
    def num_base_stations(self) -> int:
        return self.env_config['num_base_stations']

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

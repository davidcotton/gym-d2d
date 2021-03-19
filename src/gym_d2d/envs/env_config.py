from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Type

from .obs_fn import ObsFunction, LinearObsFunction
from .reward_fn import RewardFunction, SystemCapacityRewardFunction
from gym_d2d.path_loss import PathLoss, LogDistancePathLoss
from gym_d2d.traffic_model import TrafficModel, UplinkTrafficModel


@dataclass
class EnvConfig:
    num_rbs: int = 25
    num_cues: int = 25
    num_due_pairs: int = 25
    cell_radius_m: float = 500.0
    d2d_radius_m: float = 20.0
    due_min_tx_power_dBm: int = 0
    due_max_tx_power_dBm: int = 20
    cue_max_tx_power_dBm: int = 23
    mbs_max_tx_power_dBm: int = 46
    path_loss_model: Type[PathLoss] = LogDistancePathLoss
    traffic_model: Type[TrafficModel] = UplinkTrafficModel
    carrier_freq_GHz: float = 2.1
    num_subcarriers: int = 12
    subcarrier_spacing_kHz: int = 15
    channel_bandwidth_MHz: float = 20.0
    obs_fn: Type[ObsFunction] = LinearObsFunction
    reward_fn: Type[RewardFunction] = SystemCapacityRewardFunction
    compressed_info: bool = False
    device_config_file: Optional[Path] = None

    def __post_init__(self):
        self.devices = self.load_device_config()

    def load_device_config(self) -> dict:
        if isinstance(self.device_config_file, Path):
            with self.device_config_file.open(mode='r') as fid:
                return json.load(fid)
        else:
            return {}

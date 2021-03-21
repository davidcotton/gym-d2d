from gym_d2d.envs.env_config import EnvConfig
from gym_d2d.simulator import create_devices


def test_create_devices():
    env_config = EnvConfig(**{
        'num_cues': 2,
        'num_due_pairs': 2,
        'cue_max_tx_power_dBm': 9,
        'due_max_tx_power_dBm': 7,
    })
    devices = create_devices(env_config)
    assert len(devices.cues) == 2
    assert len(devices.dues) == 2
    assert devices.cues['cue00'].max_tx_power_dBm == 9.0
    due_pair = devices.dues[('due00', 'due01')]
    assert due_pair[0].max_tx_power_dBm == 7.0
    assert due_pair[1].max_tx_power_dBm == 7.0

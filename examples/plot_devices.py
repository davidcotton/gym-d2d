"""Example of how to visualise BS & UE locations.

Requires matplotlib to be installed separately (not included in package requirements).
"""

from pathlib import Path

import gym
import gym_d2d
from gym_d2d import plot_devices

# create a new environment, optionally load a device configuration from file,
# and use `reset()` to initialise the environment
env_config = {
    'device_config_file': Path.cwd() / 'device_config.json'
}
env = gym.make('D2DEnv-v0', env_config=env_config)
env.reset()

# use the plot helper to visualise device positions
plot_devices(env)

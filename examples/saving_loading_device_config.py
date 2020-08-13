"""Examples of how to save and load an environment's device configuration from file.

To make experiments repeatable and compare algorithms, you may wish to fix UE positions.
This can be achieved by saving device configurations to an editable JSON file and
reloading the same configuration each experiment.

GymD2D uses Python's inbuilt pathlib paths to refer to files as it simplifies file handling.
"""

from pathlib import Path

import gym
import gym_d2d

# SAVING DEVICE CONFIG
# --------------------
# create a new environment and use `reset()` to generate random UE positions
env = gym.make('D2DEnv-v0')
env.reset()
# save the existing device configuration to a JSON file
env.save_device_config(Path.cwd() / 'device_config.json')


# LOADING DEVICE CONFIG
# ---------------------
# initialise a new environment with a Pathlib path to a device config file
env_config = {'device_config_file': Path.cwd() / 'device_config.json'}
env = gym.make('D2DEnv-v0', env_config=env_config)
# calls to `reset()` will use the device configuration from file---if the device ID is in it.
env.reset()

# GymD2D: A Device-to-Device (D2D) Underlay Cellular Offload Evaluation Platform
__GymD2D is a toolkit for building, evaluating and comparing D2D cellular offload resource allocation algorithms.__ 
It uses [OpenAI Gym](https://gym.openai.com/) to make it easy to experiment with many popular reinforcement learning or AI algorithms. 
It is highly configurable, allowing users to experiment with UE configuration, path loss and traffic models.

GymD2D models a D2D cellular offload scenario containing a single macro base station surrounded with many cellular (CUE) and D2D (DUE) user equipment. 

This project is still under active development and we haven't finished the first stable release yet.
_Some functionality, such as `env.render()` are not currently working._

If you have found this project useful in your research, please consider citing our [white paper](https://arxiv.org/abs/2101.11188) (preprint, forthcoming at [IEEE WCNC 2021](https://wcnc2021.ieee-wcnc.org/)).
```bibtex
@article{cotton2021gymd2d,
  title={GymD2D: A Device-to-Device Underlay Cellular Offload Evaluation Platform},
  author={Cotton, David and Chaczko, Zenon},
  journal={arXiv preprint arXiv:2101.11188},
  year={2021}
}
```

## Contents
- [Requirements](#requirements)
- [Installation](#installation)
  - [Dev Installation](#dev-installation)
- [Usage](#usage)
- [Configuration](#configuration)
  - [Environment configuration](#environment-configuration)
  - [Device Configuration](#device-configuration)
  - [Observations and Rewards Configuration](#observations-and-rewards-configuration)
- [Examples](examples)


## Requirements
- Python 3.7 or greater (`dataclasses` & `__future__.annotations`)
- OpenAI Gym 0.9.6 or greater (env_config)
- NumPy

## Installation
Use pip to install

    pip install gym-d2d

### Dev Installation
Or, if you need to edit the source code, clone and install dev dependencies:

    git clone git@github.com:davidcotton/gym-d2d.git
    cd gym-d2d
    pip install -e ".[dev]"


## Usage
Import OpenAI Gym and GymD2D

    import gym
    import gym_d2d

Build a new D2D environment via the usual Gym factory method

    env = gym.make('D2DEnv-v0')

Then run in the standard Gym observation, action, reward loop.

    obs_dict = env.reset()
    game_over = False
    while not game_over:
        actions_dict = {}
        for agent_id, obs in obs_dict.items():
            action = env.action_space.sample()  # or: action = agent.act(obs)
            actions_dict[agent_id] = action
    
        obs_dict, rewards_dict, game_over, info = env.step(actions_dict)
        env.render()

There are two main differences with this environment to the usual classic control or ALE environments:
1. The environment is non-episodic, there is no terminal state, `game_over` will always be `False`.
1. Observations, actions and rewards are passed via Python dicts, see `gym.spaces.DictSpace`.

We have some common usage examples in the [examples directory](examples).


## Configuration
One of the design principles of this project is that environments should be easily configurable and customisable to meet the variety of research needs present in D2D cellular offload research.

### Environment configuration
Following the Gym API (gym>=0.9.6), you can configure the environment via an env_config dictionary.

    env = gym.make('D2DEnv-v0', env_config={'param': value})

The environment has the following configuration options:

| Parameter | Description | Datatype | Default Value |
|-----------|-------------|----------|---------------|
| num_rbs | The number of available resource blocks. | `int` | 30 |
| num_cellular_users | The number of cellular users. | `int`| 30 |
| num_d2d_pairs | The number of D2D pairs | `int` | 12 |
| cell_radius_m | The macro base station's cell radius in metres. This parameter controls the radius in which all other devices are contained. | `float` | 500.0 |
| d2d_radius_m  | The maximum distance between D2D pairs in metres. | `float` | 20.0 |
| cue_max_tx_power_dBm | The maximum CUE transmission power in dBm. This is a discrete value, from 0 to max. | `int` | 23 |
| due_min_tx_power_dBm | The minimum DUE transmission power in dBm. | `int` | 0 |
| due_max_tx_power_dBm | The maximum DUE transmission power in dBm. | `int` | 23 |
| path_loss_model | The type of path loss model to use. | `gym_d2d.` `PathLoss` | `gym_d2d.` `LogDistancePathLoss` |
| traffic_model | The model to generate automated traffic. | `gym_d2d.` `TrafficModel` | `gym_d2d.` `UplinkTrafficModel` |
| obs_fn | The function to calculate agent observations. | `gym_d2d.envs.` `ObsFunction` | `gym_d2d.envs.` `Linear1ObsFunction` |
| reward_fn | The function to calculate agent rewards. | `gym_d2d.envs.` `RewardFunction` | `gym_d2d.envs.` `SystemCapacityRewardFunction` |
| carrier_freq_GHz | The carrier frequency used, in GHz. | `float` | 2.1 |
| num_subcarriers | The number of subcarriers. | `int` | 12 |
| subcarrier_spacing_kHz | The spacing between subcarriers. | `int` | 15 |
| channel_bandwidth_MHz | The channel bandwidth in MHz. | `float` | 20.0 |
| device_config_file | A path to a device configuration JSON file. | `pathlib.Path` | None (random device positions) |

### Device Configuration
By default, each time the environment is `reset()`, each UE is randomly assigned a new position. 
To make experiments repeatable and compare algorithms, you may wish to fix UE positions.
This can be achieved by saving and loading device configurations.
Gym D2D uses Python's _Pathlib_ to make file handling easier.

    from pathlib import Path

To save an environment's device configuration to file:

    env = gym.make('D2DEnv-v0')
    env.reset()  # generate random device positions (if not supplied)
    env.save_device_config(Path.cwd() / 'device_config.json')

To load from an existing configuration:

    env_config = {'device_config_file': Path.cwd() / 'device_config.json'}
    env = gym.make('D2DEnv-v0', env_config=env_config)


### Observations and Rewards Configuration
More info coming soon on how to customise observations and rewards...

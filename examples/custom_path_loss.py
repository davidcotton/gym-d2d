from math import log10

import gym
from gym_d2d.device import Device
from gym_d2d.path_loss import PathLoss


class FooPathLoss(PathLoss):
    """Define your own custom path loss implementation.
    If you have a path loss model that isn't included, why not submit a pull request so others can use it too.
    """

    def __call__(self, tx: Device, rx: Device) -> float:
        d = tx.position.distance(rx.position)
        foo_path_loss = 20 * log10(d) - tx.tx_antenna_gain_dBi - rx.rx_antenna_gain_dBi
        return foo_path_loss


# pass your custom path loss model class (not instance) in the env_config
# alternatively, you can import and use one of the inbuilt path loss models
env_config = {'path_loss_model': FooPathLoss}
env = gym.make('D2DEnv-v0', env_config=env_config)

# run the standard agent loop, no change here
agent = 'D2DAgent()'
obs_dict = env.reset()
game_over = False
for _ in range(10):
    actions_dict = {}
    for agent_id, obs in obs_dict.items():
        action = env.action_space.sample()
        actions_dict[agent_id] = action

    obs_dict, rewards_dict, game_over, info = env.step(actions_dict)
    print(obs_dict)

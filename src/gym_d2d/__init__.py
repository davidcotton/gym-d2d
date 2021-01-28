from gym.envs.registration import register

from gym_d2d.utils import plot_devices


__all__ = ['plot_devices']

register(
    id='D2DEnv-v0',
    entry_point='gym_d2d.envs:D2DEnv',
)

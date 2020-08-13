from gym.envs.registration import register

register(
    id='D2DEnv-v0',
    entry_point='gym_d2d.envs:D2DEnv',
)

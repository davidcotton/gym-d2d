from gym.envs.registration import register

# import gym
#
#
# def register(id, entry_point):
#     env_specs = gym.envs.registry.env_specs
#     if id in env_specs.keys():
#         return
#     gym.register(id=id, entry_point=entry_point)


register(
    id='D2DEnv-v0',
    entry_point='gym_d2d.envs:D2DEnv',
)

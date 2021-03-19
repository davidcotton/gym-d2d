"""A simple example of how to use this environment.

Each step, GymD2D returns a dict of {agent_id: agent_observation}
    and requires a dict of {agent_id: actions}.
Agent rewards are also returned in a dict of {agent_id: agent_reward}.

The environment follows the Gym format of providing a game_over/terminal bool,
    but this will always be True as the environment is not episodic.
"""

import gym
import gym_d2d


env = gym.make('D2DEnv-v0')

due_agent = 'DUEAgent()'
obses = env.reset()
game_over = False
for _ in range(10):
    actions_dict = {}
    for agent_id, obs in obses.items():
        if agent_id.startswith('due'):
            action = env.action_space['due'].sample()
            # or action = due_agent.act(obs)
        elif agent_id.startswith('cue'):
            action = env.action_space['cue'].sample()
        else:
            action = env.action_space['mbs'].sample()
        actions_dict[agent_id] = action

    obses, rewards_dict, game_over, info = env.step(actions_dict)
    # env.render()
    print(obses)

from gym.envs.registration import register

register(id='logic-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='logic'))

register(id='contingency-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='contingency'))

register(id='changeAgent-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent'))

register(id='logic-shuffle-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='logic',
                     shuffle_keys=True))

register(id='contingency-shuffle-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='contingency',
                     shuffle_keys=True))

register(id='changeAgent-shuffle-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent',
                     shuffle_keys=True))

register(id='logic-noisy-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='logic',
                     noise=0.1))

register(id='contingency-noisy-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='contingency',
                     noise=0.1))

register(id='changeAgent-noisy-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent',
                     noise=0.1))

register(id='logic-shuffle-noisy-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='logic',
                     shuffle_keys=True,
                     noise=0.1))

register(id='contingency-shuffle-noisy-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='contingency',
                     shuffle_keys=True,
                     noise=0.1))

register(id='changeAgent-shuffle-noisy-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent',
                     shuffle_keys=True,
                     noise=0.1))

# register(id='logicExtended-v0',
#          entry_point='gym_gridworld.envs:GridworldEnv',
#          kwargs=dict(game_type='logic_extended'))
#
# register(id='contingencyExtended-v0',
#          entry_point='gym_gridworld.envs:GridworldEnv',
#          kwargs=dict(game_type='contingency_extended'))


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

# register(id='logicExtended-v0',
#          entry_point='gym_gridworld.envs:GridworldEnv',
#          kwargs=dict(game_type='logic_extended'))
#
# register(id='contingencyExtended-v0',
#          entry_point='gym_gridworld.envs:GridworldEnv',
#          kwargs=dict(game_type='contingency_extended'))


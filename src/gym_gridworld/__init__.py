from gym.envs.registration import register

register(id='logic-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='logic'))

register(id='logic-5-easy',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='logic-5-easy'))
register(id='logic-5-hard',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='logic-5-hard'))
register(id='logic-8-easy',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='logic-8-easy'))
register(id='logic-8-hard',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='logic-8-hard'))
register(id='logic-12-easy',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='logic-12-easy'))
register(id='logic-12-hard',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='logic-12-hard'))


register(id='contingency-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='contingency'))

register(id='contingency-5-easy',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='contingency-5-easy'))
register(id='contingency-5-hard',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='contingency-5-hard'))
register(id='contingency-8-easy',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='contingency-8-easy'))
register(id='contingency-8-hard',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='contingency-8-hard'))
register(id='contingency-12-easy',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='contingency-12-easy'))
register(id='contingency-12-hard',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='contingency-12-hard'))


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
register(id='contingency-shuffle-5-easy',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='contingency-5-easy',
                     shuffle_keys=True))
register(id='contingency-shuffle-5-hard',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='contingency-5-hard',
                     shuffle_keys=True))
register(id='contingency-shuffle-8-easy',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='contingency-8-easy',
                     shuffle_keys=True))
register(id='contingency-shuffle-8-hard',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='contingency-8-hard',
                     shuffle_keys=True))
register(id='contingency-shuffle-12-easy',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='contingency-12-easy',
                     shuffle_keys=True))
register(id='contingency-shuffle-12-hard',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='contingency-12-hard',
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

register(id='changeAgent-shuffle-noisy-5-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent',
                     shuffle_keys=True,
                     change_agent_every=5,
                     noise=0.1))

register(id='changeAgent-shuffle-noisy-7-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent',
                     shuffle_keys=True,
                     change_agent_every=7,
                     noise=0.1))

register(id='changeAgent-shuffle-noisy-10-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent',
                     shuffle_keys=True,
                     change_agent_every=10,
                     noise=0.1))

register(id='changeAgent-shuffle-noisy-15-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent',
                     shuffle_keys=True,
                     change_agent_every=15,
                     noise=0.1))

register(id='changeAgent-7-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent',
                     change_agent_every=7,
                     noise=0))



register(id='changeAgent-7-5-easy',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent-5-easy'))
register(id='changeAgent-7-5-hard',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent-5-hard'))
register(id='changeAgent-7-8-easy',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent-8-easy'))
register(id='changeAgent-7-8-hard',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent-8-hard'))
register(id='changeAgent-7-12-easy',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent-12-easy'))
register(id='changeAgent-7-12-hard',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent-12-hard'))

register(id='changeAgent-10-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent',
                     change_agent_every=10,
                     noise=0))

register(id='changeAgent-15-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent',
                     change_agent_every=15,
                     noise=0))

register(id='changeAgent-markovian-7-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent',
                     p_switch=1/7,
                     markovian=True,
                     noise=0))

register(id='changeAgent-markovian-10-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent',
                     p_switch=1/10,
                     markovian=True,
                     noise=0))

register(id='changeAgent-markovian-15-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent',
                     p_switch=1/15,
                     markovian=True,
                     noise=0))

register(id='changeAgent-noisy-7-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent',
                     change_agent_every=7,
                     noise=0.1))

register(id='changeAgent-noisy-10-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent',
                     change_agent_every=10,
                     noise=0.1))

register(id='changeAgent-noisy-15-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent',
                     change_agent_every=15,
                     noise=0.1))

register(id='changeAgent-shuffle-noisy-20-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent',
                     shuffle_keys=True,
                     change_agent_every=20,
                     noise=0.1))

register(id='changeAgent-shuffle-noisy-30-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent',
                     shuffle_keys=True,
                     change_agent_every=30,
                     noise=0.1))

register(id='changeAgent-shuffle-noisy-oneswitch-v0',
         entry_point='gym_gridworld.envs:GridworldEnv',
         kwargs=dict(game_type='change_agent',
                     shuffle_keys=True,
                     oneswitch=True,
                     noise=0.1))

# register(id='logicExtended-v0',
#          entry_point='gym_gridworld.envs:GridworldEnv',
#          kwargs=dict(game_type='logic_extended'))
#
# register(id='contingencyExtended-v0',
#          entry_point='gym_gridworld.envs:GridworldEnv',
#          kwargs=dict(game_type='contingency_extended'))


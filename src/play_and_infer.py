from copy import deepcopy
import pygame
import gym
from inferself import InferSelf
from foil_inferself import InferSelfFoil
from gym_gridworld import __init__
import warnings
warnings.filterwarnings("ignore") 

#ENV = 'logic_u-v0'
ENV = 'contingency_u-v0'
#ENV = 'changeAgent_u-7-v0'
#ENV = 'logicExtended-v0'
#ENV = 'contingency_u-shuffle-v0'

#run this screenshot plots and send to josh


#ENV = 'contingency-v0'
#ENV = 'contingency-12-hard'
#ENV = 'contingency-shuffle-v0' #infer mapping true
#ENV = 'changeAgent-7-v0'

ARGS = dict(max_steps=2,
            is_foil=False,
            # what to infer
            infer_mapping=True,
            infer_switch=True,
            # priors
            biased_action_mapping=False,
            biased_action_mapping_factor=100,
            bias_bot_mvt='uniform', # static or uniform
            p_switch=0.01,
            # learning strategies and biases
            likelihood_weight=1,
            explicit_resetting=False,
            noise_prior = 0.01,
            # exploration
            uniform_attention_bias=True,
            explore_only=False,  # if true, the agent only explores and the goal is removed from the env
            explore_randomly=False,
            simulation='sampling',  # exhaustive or sampling
            n_simulations=10,  # number of simulations if sampling
            attention_bias=False,
            mapping_forgetting_factor=0.25,
            forget_action_mapping=True,
            n_objs_attended_to=1,
            # explore-exploit
            verbose=True,
            )


def play_and_infer(env=ENV):
    screen = pygame.display.set_mode((100, 100))
    print(env)
    env = gym.make(env)
    prev_obs, prev_info, prev_obs_state = env.reset()
    env.render(None)
    args = ARGS.copy()
    if args['explore_only']:
        env.no_goal = True
    args.update(n_objs=env.n_candidates)
    obs = env.get_obs_state()
    if args['is_foil']:
        inferself = InferSelfFoil(obs=obs,
                            args=args)
    else:
        inferself = InferSelf(obs=obs,
                            args=args)
    inferself.render(true_agent=env.unwrapped.agent_id)

    running = True
    step = 0
    while running:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            # checking if keydown event happened or not
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_e, pygame.K_UP, pygame.K_DOWN, pygame.K_RIGHT]:
                    print(f'Step {step}')
                if event.key == pygame.K_e:
                    action, _ = inferself.get_action(get_observable(env.semantic_state))
                # checking if key "A" was pressed
                elif event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_RIGHT:
                    action = 3
                else:
                    pass
                if action is not None:
                    break
        if action is not None:
            print('  Action:', env.unwrapped.get_action_name(action))
            #print('  Action 2:', env.unwrapped.get_action_name(action2))
            step += 1
            obs, rew, done, info, obs_state = env.step(action)
            env.render(None)
            inferself.update_theory(prev_obs_state, obs_state, action)
            #inferself2.update_theory(prev_info['semantic_state'], info['semantic_state'], action)
            inferself.render(true_agent=env.unwrapped.agent_id)
            prev_obs_state = obs_state.copy()
            #prev_info = deepcopy(info)
            if done:
                running = False

def get_observable(state):
    map = state['map'].copy()
    map[map == 4] = 8 #true self to poss self
    map[map == 3] = 2 #true goal to poss goal
    return {'map': map, 'objects': state['objects'].copy(), 'success': state['success'], 'goal': state['goal'].copy()}

if __name__ == '__main__':
    play_and_infer()

from copy import deepcopy
import pygame
import gym
from inferself import InferSelf
from foil_inferself import InferSelfFoil
from gym_gridworld import __init__
import warnings
warnings.filterwarnings("ignore") 


#ENV = 'logic-v0'
#ENV = 'contingency-v0'
ENV = 'contingency-shuffle-v0'
#ENV = 'contingency-shuffle-v0' #infer mapping true
#ENV = 'changeAgent-7-v0'

ARGS = dict(max_steps=2,
            is_foil=True,
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
            #noise_prior_beta=[1, 15],
            noise_prior = 0.1,
            # exploration
            explore_only=False,  # if true, the agent only explores and the goal is removed from the env
            explore_randomly=False,
            simulation='sampling',  # exhaustive or sampling
            n_simulations=10,  # number of simulations if sampling
            attention_bias=False,
            mapping_forgetting_factor=0.5,
            forget_action_mapping=False,
            n_objs_attended_to=4,
            # explore-exploit
            explore_exploit_threshold=0.5, # confidence threshold for agent id
            verbose=True,
            )


def play_and_infer(env=ENV):
    screen = pygame.display.set_mode((300, 300))
    print(env)
    env = gym.make(env)
    prev_obs, prev_info = env.reset()
    env.render(None)
    args = ARGS.copy()
    if args['explore_only']:
        env.no_goal = True
    args.update(n_objs=env.n_candidates)
    if args['is_foil']:
        inferself = InferSelfFoil(env=env,
                            args=args)
    else:
        inferself = InferSelf(env=env,
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
                    action, _ = inferself.get_action(env.semantic_state)
                    #action2 = inferself2.get_action(env.semantic_state)
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
            obs, rew, done, info = env.step(action)
            env.render(None)
            inferself.update_theory(prev_info['semantic_state'], info['semantic_state'], action)
            #inferself2.update_theory(prev_info['semantic_state'], info['semantic_state'], action)
            inferself.render(true_agent=env.unwrapped.agent_id)
            prev_obs = obs.copy()
            prev_info = deepcopy(info)
            if done:
                running = False


if __name__ == '__main__':
    play_and_infer()

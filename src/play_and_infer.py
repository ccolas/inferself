from copy import deepcopy

import pygame
import gym
import gym_gridworld
from inferself import InferSelf
import numpy as np

#TODO:
#infer distrib over p_change?

ENV = 'changeAgent-shuffle-noisy-v0'
temp_noise = np.array([10, 10, 10, 5, 3, 1, 0.5, 0.1, 0.05, 0.01, 0.01])
noise_values = np.array([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])  #11
temp_noise = np.array([1, 1, 1., 1, 1])
noise_values = np.array([0, 0.05, 0.1, 0.15, 0.2])
temp_noise /= sum(temp_noise)
ARGS = dict(n_objs=4,
            biased_input_mapping=False,
            bias_bot_mvt='uniform', # static or uniform
            simulation='sampling',  # exhaustive or sampling
            n_simulations=1,  # number of simulations if sampling
            infer_mapping=True,
            threshold=0.9, # confidence threshold for agent id
            noise_prior_beta=[1, 15],
            noise_prior_discrete=temp_noise, #np.full(21, 1/21),
            noise_values_discrete= noise_values,
            forget_param=None, #the smaller this is, the more forgetful we are when computing noise
            likelihood_weight=1,
            explicit_resetting=False,
            print_status=True,
            hierarchical=True,
            p_change=0.1,
            explore_only=True  # if true, the agent only explores and the goal is removed from the env
            )

def play_and_infer(env=ENV):
    screen = pygame.display.set_mode((300, 300))

    env = gym.make(env)
    prev_obs, prev_info = env.reset()
    env.render(None)
    args = ARGS.copy()
    if args['explore_only']:
        env.no_goal = True
    args.update(n_objs=env.n_candidates)
    inferself = InferSelf(env=env,
                          args=args)

    running = True
    while running:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            # checking if keydown event happened or not
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_e:
                    action = inferself.get_action(env.semantic_state)
                # checking if key "A" was pressed
                elif event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_RIGHT:
                    action = 3
                if action is not None:
                    break
        if action is not None:
            print('Action:', env.unwrapped.get_action_name(action))
            obs, rew, done, info = env.step(action)
            env.render(None)
            theory, proba = inferself.update_theory(prev_info['semantic_state'], info['semantic_state'], action)
            inferself.render(true_agent=env.unwrapped.agent_id)
            print(f' guess: agent id={theory["agent_id"]}, proba={proba}, estimated noise: {inferself.get_noise_mean(theory, std=True)}'), #beta = {theory["beta_params"]}')
            prev_obs = obs.copy()
            prev_info = deepcopy(info)
            if done:
                running = False


if __name__ == '__main__':
    play_and_infer()

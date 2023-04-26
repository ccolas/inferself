from copy import deepcopy

import pygame
import gym
import gym_gridworld
from inferself import InferSelf

ENV = 'changeAgent-shuffle-noisy-v0'
ARGS = dict(n_objs=4,
            biased_input_mapping=False,
            bias_bot_mvt='uniform', # static or uniform
            simulation='sampling',  # exhaustive or sampling
            n_simulations=50,  # number of simulations if sampling
            infer_mapping=True,
            threshold=0.9, # confidence threshold for agent id
            beta_prior=[1, 15]
            )

def play_and_infer(env=ENV):
    screen = pygame.display.set_mode((300, 300))

    env = gym.make(env)
    prev_obs, prev_info = env.reset()
    env.render(None)
    args = ARGS
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
            print(f' guess: agent id={theory["agent_id"]}, proba={proba}, estimated noise: {inferself.get_beta_mean(theory, std=True)}, beta = {theory["beta_params"]}')
            prev_obs = obs.copy()
            prev_info = deepcopy(info)
            if done:
                running = False


if __name__ == '__main__':
    play_and_infer()

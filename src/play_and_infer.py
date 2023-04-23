from copy import deepcopy

import pygame
import gym
import gym_gridworld
from inferself import InferSelf

ENV = 'logic-v0'
ARGS = dict(n_objs=4,
            biased_input_mapping=True,
            bias_bot_mvt='uniform', # static or uniform
            simulation='exhaustive',  # exhaustive or sampling
            n_simulations=50,  # number of simulations if sampling
            infer_mapping=True,
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
            print('Action:', ['up', 'down', 'left', 'right'][action])
            obs, rew, done, info = env.step(action)
            env.render(None)
            theory, proba = inferself.update_theory(prev_info['semantic_state'], info['semantic_state'], action)
            print(f'best guess: agent id={theory["agent_id"]}, proba={proba}')
            prev_obs = obs.copy()
            prev_info = deepcopy(info)
            if done:
                running = False


if __name__ == '__main__':
    play_and_infer()

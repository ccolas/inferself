from copy import deepcopy

import pygame
import gym
import gym_gridworld
from inferself import InferSelf

ENV = 'logic'

def play_and_infer(env=ENV):
    screen = pygame.display.set_mode((300, 300))

    env = gym.make(env)
    prev_obs, prev_info = env.reset()
    env.render(None)

    inferself = InferSelf(args=dict(n_objs=env.n_candidates))

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
            obs, rew, done, info = env.step(action)
            env.render(None)
            theory, proba = inferself.update(action, prev_info['semantic_state'], info['semantic_state'])
            if proba == 1:
                assert theory['agent_id'] == env.unwrapped.agent_id
            prev_obs = obs.copy()
            prev_info = deepcopy(info)
            if done:
                running = False


if __name__ == '__main__':
    play_and_infer()

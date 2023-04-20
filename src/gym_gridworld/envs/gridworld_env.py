import json
import os
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces

# define colors
# 0: black; 1 : gray; 2 : blue; 3 : green; 4 : red

COLORS = {0: [0.0, 0.0, 0.0], 1: [0.5, 0.5, 0.5],
          2: [0.0, 0.0, 1.0], 3: [0.0, 1.0, 0.0],
          4: [1.0, 0.0, 0.0], 6: [1.0, 0.0, 1.0],
          7: [1.0, 1.0, 0.0], 8: [1.0, 0.0, 0.0],
          9: [1.0, 0.0, 0.0], 10: [1.0, 0.0, 0.0]}


DEFAULT_ARGS = dict(shuffle_keys=True,
                    change_agent_every=7)


class GridworldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, game_type, args=DEFAULT_ARGS):
        assert game_type in ['logic', 'logic_extended', 'logic_extended_h',
                             'contingency', 'contingency_extended',
                             'change_agent', 'change_agent_extended', 'change_agent_extended_1', 'change_agent_extended_2']

        self.game_type = game_type
        self._seed = None

        self.actions = [0, 1, 2, 3]
        self.action_space = spaces.Discrete(4)
        self.action_pos_dict = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

        self.shuffle_keys = args['shuffle_keys']  # whether to shuffle the action mapping between episode
        self.change_agent_every = args['change_agent_every']

        layout_path = os.path.dirname(os.path.realpath(__file__)) + '/' + self.game_type + '/'
        self.possible_layouts_paths = [layout_path + f for f in os.listdir(layout_path)]

        self.obs_shape = [128, 128, 3]  # observation space shape
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float32)

        if 'logic' in self.game_type:
            self.agent_start_locs = [[1, 1], [1, 7], [7, 1], [7, 7]]
        elif 'contingency' in self.game_type or 'change_agent' in self.game_type:
            self.agent_start_locs = [[6, 6], [6, 14], [14, 6], [14, 14]]


            #
            # self.perim = 3
            # self.oscil_dirs = [np.random.randint(1), np.random.randint(1), np.random.randint(1)]  # whether to oscil ud (0) or lr (1)
            #
            # if 'extended' in self.game_type:
            #     self.oscil_dirs.append(np.random.randint(1))
        self.fig = None
        self.reset()
        stop = 1

    def reset(self):

        self.step_counter = 0
        # sample new grid map, read it and convert it to numpy array
        grid_map_path = np.random.choice(self.possible_layouts_paths)
        with open(grid_map_path, 'r') as f:
            grid_map = f.readlines()
        self.start_grid_map = np.array([[int(el) for el in line.replace('\n', '').split(' ') ] for line in grid_map])

        # sample agent location
        agent_pos = self.agent_start_locs[np.random.randint(4)]
        self.start_grid_map[agent_pos[0], agent_pos[1]] = 4  # add it to the map

        # get object positions
        self.goal_pos = self.get_pos(of_what='goal', map=self.start_grid_map)
        self_pos = self.get_pos(of_what='self', map=self.start_grid_map)
        self.candidates_pos = self.get_pos(of_what='candidates', map=self.start_grid_map)
        self.agent_id = np.argwhere([np.all(cpos == self_pos) for cpos in self.candidates_pos]).flatten()[0]
        self.start_grid_map[np.where(np.logical_and(self.start_grid_map != 1, self.start_grid_map != 0))] = 0  # only keep walls and empty cells

        # get current map and observation
        self.build_current_map()
        self.observation = self._gridmap_to_observation(self.current_grid_map)
        self.grid_map_shape = self.start_grid_map.shape

        # shuffle action mapping
        if self.shuffle_keys:
            np.random.shuffle(self.action_pos_dict)

        if ('contingency' in self.game_type or 'change' in self.game_type) and ('extended_1' in self.game_type or 'extended_2' in self.game_type):
            self.mock_location = np.random.choice([[10, 6], [6, 10], [10, 14], [14, 10]])

        return self.observation

    @property
    def self_pos(self):
        return self.candidates_pos[self.agent_id]

    @property
    def non_self_candidates_pos(self):
        return [self.candidates_pos[i] for i in range(len(self.candidates_pos)) if i != self.agent_id]

    def build_current_map(self):
        self.current_grid_map = self.start_grid_map.copy()
        self.current_grid_map[self.goal_pos[0], self.goal_pos[1]] = 3
        self.current_grid_map[self.self_pos[0], self.self_pos[1]] = 4
        for candidate_pos in self.non_self_candidates_pos:
            self.current_grid_map[candidate_pos[0], candidate_pos[1]] = 8

    def render(self, mode):

        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            self.plot = self.ax.imshow(self.observation)
            plt.axis('off')
            plt.show(block=False)
            plt.pause(0.01)
        else:
            self.plot.set_data(self.observation)
            self.fig.canvas.draw()
            plt.pause(0.01)

    def seed(self, seed):
        self._seed = seed
        np.random.seed(self._seed)

    def step(self, action):
        self.step_counter += 1
        if 'logic' in self.game_type:
            new_obs, rew, done, info = self.step_logic(action)
        elif 'contingency' in self.game_type:
            new_obs, rew, done, info = self.step_contingency(action)
        elif 'change_agent' in self.game_type:
            new_obs, rew, done, info = self.step_change_agent(action)
        else:
            raise NotImplementedError

        return new_obs, rew, done, info

    def is_empty(self, pos):

        return self.current_grid_map[pos[0], pos[1]] in [0, 3]

    def step_logic(self, action):
        action = int(action)
        info = dict(success=False)

        new_candidates_pos = [None for _ in range(len(self.candidates_pos))]

        # update agent pos first
        next_agent_pos = self.candidates_pos[self.agent_id] + self.action_pos_dict[action]
        if self.is_empty(next_agent_pos):
            new_candidates_pos[self.agent_id] = next_agent_pos
            if np.all(next_agent_pos == self.goal_pos):
                info['success'] = True
        else:
            new_candidates_pos[self.agent_id] = self.candidates_pos[self.agent_id]

        # update other  pos
        for i_candidate, candidate_pos in enumerate(self.candidates_pos):
            if i_candidate != self.agent_id:
                new_candidates_pos[i_candidate] = self.candidates_pos[i_candidate]
        self.candidates_pos = new_candidates_pos.copy()

        self.build_current_map()
        self.observation = self._gridmap_to_observation(self.current_grid_map)
        return self.observation, int(info['success']), info['success'], info

    def pos_in_list(self, pos, list_pos):
        for p in list_pos:
            if p is not None:
                if np.all(pos == p):
                    return True
        return False

    def step_contingency(self, action):
        action = int(action)
        info = dict(success=False)

        new_candidates_pos = [None for _ in range(len(self.candidates_pos))]

        # update agent pos first
        next_agent_pos = self.candidates_pos[self.agent_id] + self.action_pos_dict[action]
        if self.is_empty(next_agent_pos):
            new_candidates_pos[self.agent_id] = next_agent_pos
            if np.all(next_agent_pos == self.goal_pos):
                info['success'] = True
        else:
            new_candidates_pos[self.agent_id] = self.candidates_pos[self.agent_id]

        # update other  pos
        possible_directions = [[0, 1], [1, 0], [-1, 0], [0, -1]]
        for i_candidate, candidate_pos in enumerate(self.candidates_pos):
            if i_candidate != self.agent_id:
                next_pos = candidate_pos + possible_directions[np.random.choice(np.arange(len(possible_directions)))]
                if self.is_empty(next_pos) and not self.pos_in_list(next_pos, new_candidates_pos):
                    new_candidates_pos[i_candidate] = next_pos
                else:
                    new_candidates_pos[i_candidate] = self.candidates_pos[i_candidate]
        self.candidates_pos = new_candidates_pos.copy()

        self.build_current_map()
        self.observation = self._gridmap_to_observation(self.current_grid_map)
        return self.observation, int(info['success']), info['success'], info

    def step_change_agent(self, action):

        if self.step_counter % self.change_agent_every == 0:
            self.agent_id = np.random.choice([i for i in range(len(self.candidates_pos)) if i != self.agent_id])

        action = int(action)
        info = dict(success=False)

        new_candidates_pos = [None for _ in range(len(self.candidates_pos))]

        # update agent pos first
        next_agent_pos = self.candidates_pos[self.agent_id] + self.action_pos_dict[action]
        if self.is_empty(next_agent_pos):
            new_candidates_pos[self.agent_id] = next_agent_pos
            if np.all(next_agent_pos == self.goal_pos):
                info['success'] = True
        else:
            new_candidates_pos[self.agent_id] = self.candidates_pos[self.agent_id]

        # update other  pos
        possible_directions = [[0, 1], [1, 0], [-1, 0], [0, -1]]
        for i_candidate, candidate_pos in enumerate(self.candidates_pos):
            if i_candidate != self.agent_id:
                next_pos = candidate_pos + possible_directions[np.random.choice(np.arange(len(possible_directions)))]
                if self.is_empty(next_pos) and not self.pos_in_list(next_pos, new_candidates_pos):
                    new_candidates_pos[i_candidate] = next_pos
                else:
                    new_candidates_pos[i_candidate] = self.candidates_pos[i_candidate]
        self.candidates_pos = new_candidates_pos.copy()

        self.build_current_map()
        self.observation = self._gridmap_to_observation(self.current_grid_map)
        return self.observation, int(info['success']), info['success'], info


    def get_pos(self, of_what, map=None):
        if map is None:
            map = self.current_grid_map
        if of_what == 'goal':
            return np.argwhere(map == 3).flatten()
        elif of_what == 'self':
            return np.argwhere(map == 4).flatten()
        elif of_what == 'candidates':
            return np.argwhere(np.logical_or(map == 4, map==8))
        elif of_what == 'non_self_candidates':
            return np.argwhere(map==8)

    def _gridmap_to_observation(self, grid_map, obs_shape=None):
        if obs_shape is None:
            obs_shape = self.obs_shape
        observation = np.zeros(obs_shape, dtype=np.float32)
        gs0 = int(observation.shape[0] / grid_map.shape[0])
        gs1 = int(observation.shape[1] / grid_map.shape[1])
        for i in range(grid_map.shape[0]):
            for k in range(grid_map.shape[1]):
                observation[i * gs0:(i + 1) * gs0, k * gs1:(k + 1) * gs1] = np.array(COLORS[grid_map[i, k]])

        return observation

    def _close_env(self):
        plt.close('all')

def play(env):
    import pygame
    screen = pygame.display.set_mode((300, 300))
    env.reset()
    env.render()

    running = True

    while running:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            # checking if keydown event happened or not
            if event.type == pygame.KEYDOWN:
                # checking if key "A" was pressed
                if event.key == pygame.K_UP:
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
            env.render()
            if done:
                running = False



if __name__ == '__main__':
    from gym.envs.registration import register

    register(id='logic-v0',
             entry_point='src.gym_gridworld.envs:GridworldEnv',
             kwargs=dict(game_type='logic'))

    env = gym.make('logic-v0')

    play(env)
    stop = 1
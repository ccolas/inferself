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


class GridworldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, game_type, noise=0, no_goal=False, shuffle_keys=False, change_agent_every=15, oneswitch=False):
        assert game_type in ['logic', 'logic_extended', 'logic_extended_h',
                             'contingency', 'contingency_extended',
                             'change_agent', 'change_agent_extended', 'change_agent_extended_1', 'change_agent_extended_2']

        self.game_type = game_type
        self._seed = None

        self.actions = [0, 1, 2, 3]
        self.action_space = spaces.Discrete(4)
        self.action_pos_dict = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        self.action_names =  ['up', 'down', 'left', 'right']

        self.shuffle_keys = shuffle_keys  # whether to shuffle the action mapping between episode
        self.noise = noise
        self.change_agent_every = change_agent_every
        self.no_goal = no_goal
        self.oneswitch = oneswitch

        layout_path = os.path.dirname(os.path.realpath(__file__)) + '/' + self.game_type + '/'
        self.possible_layouts_paths = [layout_path + f for f in os.listdir(layout_path)]

        self.obs_shape = [128, 128, 3]  # observation space shape
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float32)

        if 'logic' in self.game_type:
            self.agent_start_locs = [[1, 1], [1, 7], [7, 1], [7, 7]]
        elif 'contingency' in self.game_type or 'change_agent' in self.game_type:
            self.agent_start_locs = [[6, 6], [6, 14], [14, 6], [14, 14]]
        self.fig = None
        self.reset()

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
        #got idx out of bounds error
        #should self pos only have 1 option?
        self.agent_id = np.argwhere([np.all(cpos == self_pos) for cpos in self.candidates_pos]).flatten()[0]
        self.start_grid_map[np.where(np.logical_and(self.start_grid_map != 1, self.start_grid_map != 0))] = 0  # only keep walls and empty cells

        # get current map and observation
        self.build_current_map()
        self.observation = self._gridmap_to_observation(self.current_grid_map)
        self.grid_map_shape = self.start_grid_map.shape

        # shuffle action mapping
        if self.shuffle_keys:
            action_indices = np.arange(4)
            np.random.shuffle(action_indices)
            self.action_pos_dict = [self.action_pos_dict[i] for i in action_indices]
            self.action_names = [self.action_names[i] for i in action_indices]

        if ('contingency' in self.game_type or 'change' in self.game_type) and ('extended_1' in self.game_type or 'extended_2' in self.game_type):
            self.mock_location = np.random.choice([[10, 6], [6, 10], [10, 14], [14, 10]])

        self.contingency_directions = np.random.randint(1, size=len(self.candidates_pos))
        self.semantic_state = self.get_semantic_state()
        return self.observation, dict(semantic_state=deepcopy(self.semantic_state))

    def get_action_name(self, action_id):
        return self.action_names[action_id]

    def get_semantic_state(self):
        semantic_state = dict(objects=self.candidates_pos,
                              map=self.current_grid_map,
                              goal=self.goal_pos)
        return semantic_state
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
        self.plot.set_data(self.observation)
        self.fig.canvas.draw()
        plt.pause(0.1)

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
        self.semantic_state = self.get_semantic_state()
        info['semantic_state'] = deepcopy(self.semantic_state)
        return new_obs, rew, done, info

    def is_empty(self, pos, agent=False, map=None):
        # checks whether the position is empty in the current map. If the agent, the goal is considered as an empty location
        if agent: empty = [0, 3]
        else: empty = [0]
        if map is None: map = self.current_grid_map
        return map[int(pos[0]), int(pos[1])] in empty

    def step_logic(self, action):
        action = int(action)
        info = dict(success=False)

        new_candidates_pos = [None for _ in range(len(self.candidates_pos))]

        # update agent pos first
        if np.random.rand() < self.noise:
            candidate_actions = sorted(set(range(5)) - set([action]))
            action = np.random.choice(candidate_actions)
            if action < 4:
                action_dir = self.action_pos_dict[action]
            else:
                action_dir = np.zeros(2)
        else:
            action_dir = self.action_pos_dict[action]
        next_agent_pos = self.candidates_pos[self.agent_id] + action_dir
        if not self.no_goal:
            stop = 1
        if self.is_empty(next_agent_pos, agent=True):
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
        current_agent_pos = self.candidates_pos[self.agent_id]
        if np.random.rand() < self.noise:
            candidate_actions = sorted(set(range(5)) - set([action]))
            action = np.random.choice(candidate_actions)
            if action < 4:
                action_dir = self.action_pos_dict[action]
            else:
                action_dir = np.zeros(2)
        else:
            action_dir = self.action_pos_dict[action]
        next_agent_pos = current_agent_pos + action_dir
        if self.is_empty(next_agent_pos, agent=True):
            new_candidates_pos[self.agent_id] = next_agent_pos
            if np.all(next_agent_pos == self.goal_pos):
                info['success'] = True
            # update position of the agent in the current map
            self.current_grid_map[current_agent_pos[0], current_agent_pos[1]] = 0
            self.current_grid_map[next_agent_pos[0], next_agent_pos[1]] = 4
        else:
            new_candidates_pos[self.agent_id] = self.candidates_pos[self.agent_id]

        # update other  pos
        for i_candidate, candidate_pos in enumerate(self.candidates_pos):
            if i_candidate != self.agent_id:
                mvt = np.zeros(2)
                mvt[self.contingency_directions[i_candidate]] = np.random.choice([-1, 1])
                next_pos = (candidate_pos + mvt).astype(int)
                if self.is_empty(next_pos):
                    new_candidates_pos[i_candidate] = next_pos
                    # update position of the agent in the current map
                    self.current_grid_map[candidate_pos[0], candidate_pos[1]] = 0
                    self.current_grid_map[next_pos[0], next_pos[1]] = 8
                else:
                    new_candidates_pos[i_candidate] = self.candidates_pos[i_candidate]
        self.candidates_pos = new_candidates_pos.copy()

        self.build_current_map()
        self.observation = self._gridmap_to_observation(self.current_grid_map)
        return self.observation, int(info['success']), info['success'], info

    def step_change_agent(self, action):

        if self.oneswitch:
            if self.step_counter == 30:
                self.agent_id = np.random.choice([i for i in range(len(self.candidates_pos)) if i != self.agent_id])
                print(f'AGENT CHANGES OMGGG: {self.agent_id}')
        else:
            if self.step_counter % self.change_agent_every == 0:
                self.agent_id = np.random.choice([i for i in range(len(self.candidates_pos)) if i != self.agent_id])
                print(f'AGENT CHANGES OMGGG: {self.agent_id}')

        action = int(action)
        info = dict(success=False)

        new_candidates_pos = [None for _ in range(len(self.candidates_pos))]

        # update agent pos first
        current_agent_pos = self.candidates_pos[self.agent_id]
        if np.random.rand() < self.noise:
            candidate_actions = sorted(set(range(5)) - set([action]))
            action = np.random.choice(candidate_actions)
            if action < 4:
                action_dir = self.action_pos_dict[action]
            else:
                action_dir = np.zeros(2)
        else:
            action_dir = self.action_pos_dict[action]
        next_agent_pos = current_agent_pos +  action_dir
        if self.is_empty(next_agent_pos, agent=True):
            new_candidates_pos[self.agent_id] = next_agent_pos
            if np.all(next_agent_pos == self.goal_pos) and not self.no_goal and (not self.oneswitch or self.step_counter > 30):
                # there is a goal if not no_goal and in the case of oneswitch, if step_counter is after the switch
                info['success'] = True
            # update position of the agent in the current map
            self.current_grid_map[current_agent_pos[0], current_agent_pos[1]] = 0
            self.current_grid_map[next_agent_pos[0], next_agent_pos[1]] = 4
        else:
            new_candidates_pos[self.agent_id] = self.candidates_pos[self.agent_id]

        # update other  pos
        possible_directions = [[0, 1], [1, 0], [-1, 0], [0, -1]]
        for i_candidate, candidate_pos in enumerate(self.candidates_pos):
            if i_candidate != self.agent_id:
                mvt = np.zeros(2)
                mvt[self.contingency_directions[i_candidate]] = np.random.choice([-1, 1])
                next_pos = (candidate_pos + mvt).astype(int)
                if self.is_empty(next_pos) and not self.pos_in_list(next_pos, new_candidates_pos):
                    new_candidates_pos[i_candidate] = next_pos
                    # update position of the agent in the current map
                    self.current_grid_map[candidate_pos[0], candidate_pos[1]] = 0
                    self.current_grid_map[next_pos[0], next_pos[1]] = 8
                else:
                    new_candidates_pos[i_candidate] = candidate_pos
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

    @property
    def n_candidates(self):
        return len(self.candidates_pos)

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
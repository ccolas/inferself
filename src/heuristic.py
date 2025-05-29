import numpy as np
import scipy
from copy import deepcopy
import matplotlib.pyplot as plt
import heapq
import copy
import random

"""
Proximity heuristic agent for the Avatar Games
Attempts to move closest charactar to goal towards the goal
When the action mapping is unknown, selects action mapping based on n. observations consistent with that mapping
When the true goal is unknown, moves character to goal for closest character/goal pair, rules out previously visited goals
"""

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
          '#7f7f7f', '#bcbd22', '#17becf']
poss_goal_num = 2 # number indicating a possible goal in the grid

class Heuristic:
    def __init__(self, obs, args):
        self.args = args
        self.check_oob = args['check_oob']
        self.prior_p_switch = args['p_switch']
        self.n_objs = np.count_nonzero(obs['map'] == 8)
        self.prev_agent_list = []
        self.no_action_effect = False
        self.actions = range(4)
        self.directions = [[-1, 0], [1, 0], [0, -1], [0, 1]] # up down left right
        self.directions_str = [l2s(l) for l in self.directions] # convert directions to string form
        self.mode = 'exploit'
        if self.args['bias_bot_mvt'] == 'static':
            self.prior_npc_mvt = np.tile(np.append(np.zeros(len(self.directions)), 1), (self.n_objs, 1))
        elif self.args['bias_bot_mvt'] == 'uniform':
            self.prior_npc_mvt = np.full((self.n_objs, len(self.directions) + 1), 1 / (len(self.directions) + 1))
        else: raise NotImplementedError
        self.last_action = None
        self.history_posteriors_over_agents = [[1 / self.n_objs for _ in range(self.n_objs)]]  # fill history with prior on agent identity
        self.history_posteriors_p_switch = [-1]
        self.key_success_counts = None
        self.setup_theories()
        self.fig = None
        self.poss_goal_locs = np.transpose(np.nonzero(obs['map']==poss_goal_num))
        self.prev_selected_char = None
        self.noise = args['heuristic_noise']
        self.noise_mean_prior = self.args['noise_prior']


    # # # # # # # # # # # # # # # #
    # Setting up inference
    # # # # # # # # # # # # # # # #
    def setup_theories(self):
        self.theories = {}
        self.initial_prior_over_theories = {}
        dirs = self.directions_str.copy()
        if self.args['infer_mapping']:
            self.key_success_counts = []
            for _ in range(len(self.actions)):
                self.key_success_counts.append(np.zeros(len(self.directions)+1))
            
        # list all theories
        # for each direction, how many times did each key make it move in that direction?
        self.mappings = []
        self.mapping_consistency_counts = []
        for dir0 in dirs:
            remaining_dirs0 = dirs.copy()
            remaining_dirs0.remove(dir0)
            for dir1 in remaining_dirs0:
                remaining_dirs1 = remaining_dirs0.copy()
                remaining_dirs1.remove(dir1)
                for dir2 in remaining_dirs1:
                    remaining_dir2 = remaining_dirs1.copy()
                    remaining_dir2.remove(dir2)
                    for dir3 in remaining_dir2:
                        dir0l, dir1l, dir2l, dir3l = [s2l(dir) for dir in [dir0, dir1, dir2, dir3]]  # convert string forms into list forms
                        mapping = dict(input_mapping={0: dir0l, 1: dir1l, 2: dir2l, 3: dir3l},
                                            input_reverse_mapping={dir0: 0, dir1: 1, dir2: 2, dir3: 3})
                        self.mappings.append(mapping)
                        self.mapping_consistency_counts.append(0)
            self.reset_prior_over_theories()

    def reset_prior_over_theories(self):
        self.current_posterior_over_theories = self.initial_prior_over_theories.copy()

    def update_theory(self, prev_obs, new_obs, action):
        # if prev selected char at prev selected goal and no success, remove from poss goal list
        if self.prev_selected_char != None:
            if np.all(new_obs['objects'][self.prev_selected_char] == self.prev_selected_goal):
                self.poss_goal_locs = [l for l in self.poss_goal_locs if ~(np.all(l==self.prev_selected_goal))]
        if self.args['infer_mapping']:
            # update key_success here:
            # what was the previously moved agent and which dir did it move?
            if self.prev_selected_char != None:
                dir = new_obs['objects'][self.prev_selected_char] - prev_obs['objects'][self.prev_selected_char]
            # was this the expected dir?
            for (i, mapping) in enumerate(self.mappings):
                intended_dir = mapping['input_mapping'][action]
                if self.is_agent_mvt_consistent(self.prev_selected_char, intended_dir, prev_obs, new_obs):
                    self.mapping_consistency_counts[i] +=1


    # # # # # # # # # # # #
    # exploitation
    # # # # # # # # # # # #
    def exploit(self, obs, agent_id, goal_loc):
        # Define the heuristic function (Manhattan distance)
        def heuristic_dist(loc, goal_loc):
            return abs(loc[0] - goal_loc[0]) + abs(loc[1] - goal_loc[1])
        
        # Move whichever is closest to the goal
        agent_pos = list(obs['objects'][agent_id])

        parents = {}
        parents[tuple(agent_pos)] = None
        parent_actions = {}
        parent_actions[tuple(agent_pos)] = self.last_action
        g_costs = {}
        g_costs[tuple(agent_pos)] = 0
        
        # initialize the open and closed lists
        frontier = [[heuristic_dist(agent_pos, goal_loc), 0, agent_pos]]
        visited = set()

        iter = 0
        while frontier:
            iter=iter+1
            # get lowest cost loc in open list
            (_, a, current_loc) = heapq.heappop(frontier)
            # add to closed set
            visited.add(tuple(current_loc))
            # if at goal, return path
            if np.all(current_loc == goal_loc):
                path = []
                while current_loc:
                    path.append(current_loc)
                    current_loc = parents[tuple(current_loc)]
                path.reverse() 
                return path
            # get possible neighbors
            poss_neighbors = [([current_loc[0]-1, current_loc[1]], 0), ([current_loc[0]+1, current_loc[1]], 1),
                                ([current_loc[0], current_loc[1]-1], 2), ([current_loc[0], current_loc[1]+1], 3)]
            random.shuffle(poss_neighbors)
            for (i, (neighbor, action)) in enumerate(poss_neighbors):
                # check if we can move to this neighbor
                if not(self.is_empty(obs['map'], neighbor, agent=True)) or tuple(neighbor) in visited:
                    continue
                new_g_cost = g_costs[tuple(current_loc)] + 1  # assuming each step has a cost of 1
                # update 
                if new_g_cost < g_costs.get(tuple(neighbor), float('inf')):
                    g_costs[tuple(neighbor)] = new_g_cost
                    f_cost = g_costs[tuple(neighbor)] + heuristic_dist(neighbor, goal_loc)
                    parents[tuple(neighbor)] = current_loc 
                    parent_actions[tuple(neighbor)] = action
                    if parent_actions[tuple(current_loc)] == action:
                        tie_breaker=0
                    else:
                        tie_breaker = i+1
                        f_cost = f_cost + 1/100
                        g_costs[tuple(neighbor)] = g_costs[tuple(neighbor)] + 1/100
                    heapq.heappush(frontier, [f_cost, tie_breaker, neighbor])
        # no path found
        return None

    def get_action(self, obs, enforce_mode=None):
        # get length of path to goal for each character
        path_lengths = []
        paths = []
        agents = []
        goal_locs = []
        
        # with high prob, choose agent/goal_loc pair with min path length
        if np.random.rand() >= self.noise:
            for id in range(self.n_objs):
                for goal_loc in self.poss_goal_locs:
                    path = self.exploit(obs, id, goal_loc)
                    # move to dif goal of already at one
                    if path != None and len(path)<2:
                        path = None
                    paths.append(path)
                    if path == None:
                        path_lengths.append(float('inf'))
                    else:
                        path_lengths.append(len(path))
                    agents.append(id)
                    goal_locs.append(goal_loc)
            path_idx = np.random.choice(np.flatnonzero(path_lengths == np.min(path_lengths)), 1)[0]
            path = paths[path_idx]
            agent_id = agents[path_idx]
            goal_loc = goal_locs[path_idx]
        else: # otherwise choose random agent and move it to the closest goal 
            agent_id = np.random.choice(range(self.n_objs), p=np.ones(self.n_objs) / self.n_objs, size=1)[0]
            for goal_loc in self.poss_goal_locs:
                path = self.exploit(obs, agent_id, goal_loc)
                # move to dif goal of already at one
                if path != None and len(path)<2:
                    path = None
                paths.append(path)
                if path == None:
                    path_lengths.append(float('inf'))
                else:
                    path_lengths.append(len(path))
                agents.append(agent_id)
                goal_locs.append(goal_loc)
            path_idx = np.random.choice(np.flatnonzero(path_lengths == np.min(path_lengths)), 1)[0]
            path = paths[path_idx]
            goal_loc = self.poss_goal_locs[path_idx]
        if path == None:
            dir_to_go_str = np.random.choice(self.directions_str, 1)[0]
        else:
            dir_to_go_str = l2s([path[1][0] - path[0][0], path[1][1] - path[0][1]])
        # based on dir, get action
        if self.args['infer_mapping']:
            dir_idx = self.directions_str.index(dir_to_go_str)
            # choose best action mapping
            mapping_idx = np.random.choice(np.flatnonzero(np.array(self.mapping_consistency_counts) == max(self.mapping_consistency_counts)))
            action = self.mappings[mapping_idx]['input_reverse_mapping'][dir_to_go_str]
        else:
            action = self.directions_str.index(dir_to_go_str)
        self.last_action = action
        self.prev_selected_char = agent_id
        self.prev_selected_goal = goal_loc
        return action, 'exploit'

    # # # # # # # # # # # #
    # utilities 
    # # # # # # # # # # # #
    def out_of_bounds(self, obs, id):
        pos = list(obs['objects'][id])
        if pos[0] < 2 or pos[0] > 18 or pos[1] < 2 or pos[1] > 18:
            return True
        else:
            return False
        
    def get_noise_mean(self, theory):
        return self.noise_mean_prior

    def next_obj_pos(self, prev_pos, action_dir, current_map, agent):
        predicted_pos = prev_pos + action_dir

        if agent and self.agent_at_goal(current_map):
            return prev_pos
        elif self.is_empty(current_map, predicted_pos, agent=agent):
            return predicted_pos
        else:
            return prev_pos

    def next_to_wall(self, map, loc):
        if np.shape(map)[0]<10:
            return False
        adj = [map[int(loc[0])-1, int(loc[1])], map[int(loc[0])+1, int(loc[1])], map[int(loc[0]), int(loc[1])-1], map[int(loc[0]), int(loc[1])+1]]
        if 1 in adj:
            return True
        else:
            return False

    # based off current beliefs about what the goal is
    def agent_at_goal(self, current_map):
        if self.mode == 'explore':
            return False
        else:
            return False

    def is_agent_mvt_consistent(self, id, action_dir, prev_obs, new_obs):
        current_map = prev_obs['map'].copy()
        prev_obj_pos = prev_obs['objects']
        new_obj_pos = new_obs['objects']
        # compute likelihood of agent movement
        prev_pos = prev_obj_pos[id]
        new_pos = new_obj_pos[id]
        predicted_pos = self.next_obj_pos(prev_pos, action_dir, current_map, True)
        return np.all(predicted_pos == new_pos)

    def is_empty(self, map, loc, agent=True):
        # can only move to goal if agent
        if agent: empty = [0, 2]
        else: empty = [0]
        # check for out of bounds
        if (int(loc[0]) < 0) or (int(loc[0]) >= np.shape(map)[0]) or (int(loc[1]) < 0) or (int(loc[1]) >= np.shape(map)[1]):
            return False
        return map[int(loc[0]), int(loc[1])] in empty

    @property
    def n_theories(self):
        return len(self.theories)

    def get_smooth_posterior_over_theories(self, smooth=5):
        posterior_over_agents = np.atleast_2d(np.array(self.history_posteriors_over_agents.copy()))
        smooth_posterior_over_theories = np.zeros(posterior_over_agents.shape)
        for i in range(posterior_over_agents.shape[0]):
            smooth_posterior_over_theories[i, :] = np.mean(posterior_over_agents[max(0, i - smooth): i+1, :], axis=0)
        return smooth_posterior_over_theories

    def get_posterior_over_agents(self, theories, probs):
        agent_probs = np.zeros(self.n_objs)
        for i, t in enumerate(theories):
            id = t['agent_id']
            agent_probs[id] += probs[i]
        return agent_probs

    def print_top(self, theories, probs):
        probs = copy.deepcopy(probs)
        print("  Top mapping theory:")
        print(self.key_success_counts)

    def render(self, true_agent=None, smooth=5):
        data = np.atleast_2d(np.array(self.history_posteriors_over_agents.copy()))
        smooth_data = self.get_smooth_posterior_over_theories(smooth=smooth)

        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            for i, d in zip(range(data.shape[1]), data.T):
                self.ax.plot(d, c=COLORS[i],  label=f'{i}')
            #for i, d in zip(range(data.shape[1]), smooth_data.T):
            #    self.ax.plot(d, linestyle='--', c=COLORS[i], label=f'{i} smoothed')
            if true_agent is not None:
                self.ax.scatter(data.shape[0] - 1, 1, c=COLORS[true_agent])
            plt.legend()
            plt.ylim([0, 1.05])
            plt.show(block=False)
        if true_agent is not None:
            self.ax.scatter(data.shape[0] - 1, 1, c=COLORS[true_agent])
        for i, d in zip(range(data.shape[1]), data.T):
            self.ax.plot(d, c=COLORS[i])

        self.fig.canvas.draw()
        stop = 1

def dict2s(d):
    d2 = {}
    d2['goal'] = list(d['goal'])
    d2['map'] = [list(row) for row in d['map']]
    d2['objects'] = [list(o) for o in d['objects']]
    return str(list(sorted(d2.items(), key=lambda x: x[0])))

def s2dict(s):
    l = eval(s)
    d = {}
    for (k,v) in l:
        d[k] = v
    d2 = {}
    d2['goal'] = np.array(d['goal'])
    d2['map'] = np.array([np.array(row) for row in d['map']])
    d2['objects'] = [np.array(o) for o in d['objects']]
    return d2

def l2s(l):
    return f'{l[0]}_{l[1]}'

def s2l(s):
    return [int(ss) for ss in s.split('_')]

def information_gain(p0, p1):
    p0 = [round(x,10) for x in p0]
    p1 = [round(x,10) for x in p1]
    return scipy.spatial.distance.jensenshannon(p0, p1)

def fix_p( p):
    if p.sum() != 1.0:
        p = p*(1./p.sum())
    return p

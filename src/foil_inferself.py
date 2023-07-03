import numpy as np
import scipy
from copy import deepcopy
import matplotlib.pyplot as plt
import heapq
import copy

#here, we have no representation of the "self"
#in our simulation of the world, each object has some probability of acting as we expect (0.25 + noise)

#try to move whichever agent is closest to the goal
#if the world does not change, move a different agent until you've tried to move each agent
#how to do mapping inference?
#believe that at any time, agent you move has a 0.25 prob of moving the way you expect, .75/4 of moving all ways
#compute mapping for each agent
#so still computing, if this were me, what's the prob of this mapping?
#but don't compute prob of being me

#maybe a dif version is, think that any point in time you have .25 prob of moving any agent?

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
          '#7f7f7f', '#bcbd22', '#17becf']

class InferSelfFoil:
    def __init__(self, env, args):
        self.args = args
        self.prior_p_switch = args['p_switch']
        self.env = env
        self.prev_agent_list = []
        self.no_action_effect = False
        self.n_objs = self.env.n_objs
        assert 'noise' not in self.env.__str__()
        self.actions = range(4)
        self.directions = [[-1, 0], [1, 0], [0, -1], [0, 1]] # up down left right
        self.directions_str = [l2s(l) for l in self.directions] # convert directions to string form

        if self.args['bias_bot_mvt'] == 'static':
            self.prior_npc_mvt = np.tile(np.append(np.zeros(len(self.directions)), 1), (self.n_objs, 1))
        elif self.args['bias_bot_mvt'] == 'uniform':
            self.prior_npc_mvt = np.full((self.n_objs, len(self.directions) + 1), 1 / (len(self.directions) + 1))
        else: raise NotImplementedError

        self.history_posteriors_over_agents = [[1 /  self.n_objs for _ in range(self.n_objs)]]  # fill history with prior on agent identity
        self.history_posteriors_p_switch = [-1]
        self.setup_theories()
        self.fig = None
        self.noise_mean_prior = self.args['noise_prior']
        #self.forget_action_mappings


    # # # # # # # # # # # # # # # #
    # Setting up inference
    # # # # # # # # # # # # # # # #
    def setup_theories(self):
        self.theories = {}
        self.initial_prior_over_theories = {}
        dirs = self.directions_str.copy()
        # list all theories
        for agent_id in range(self.n_objs):
            if self.args['infer_mapping']:
                self.theories[agent_id] = []
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
                                new_theory = dict(input_mapping={0: dir0l, 1: dir1l, 2: dir2l, 3: dir3l},
                                                  input_reverse_mapping={dir0: 0, dir1: 1, dir2: 2, dir3: 3})
                                self.theories[agent_id].append(new_theory)
            else:
                self.theories[agent_id] = [dict(input_mapping={0: s2l(dirs[0]), 1: s2l(dirs[1]), 2: s2l(dirs[2]), 3: s2l(dirs[3])},
                                                  input_reverse_mapping={dirs[0]: 0, dirs[1]: 1, dirs[2]: 2, dirs[3]: 3})]
            self.initial_prior_over_theories[agent_id] = np.ones(len(self.theories[agent_id])) / len(self.theories[agent_id])  # uniform probability distribution over theories
            if self.args['biased_action_mapping']:
                self.initial_prior_over_theories[agent_id][0] *= self.args['biased_action_mapping_factor']
                self.initial_prior_over_theories[agent_id] /= self.initial_prior_over_theories[agent_id].sum()
            self.reset_prior_over_theories()
                #self.update_objs_attended_to()


    def reset_prior_over_theories(self):
        self.current_posterior_over_theories = self.initial_prior_over_theories.copy()

    # # # # # # # # # # # # # # # #
    # Running inference
    # # # # # # # # # # # # # # # #
    def update_theory(self, prev_obs, new_obs, action):
        changes = [i for i in range(len(prev_obs['objects'])) if list(prev_obs['objects'][i]) != list(new_obs['objects'][i])]
        if len(changes) == 0:
            self.no_action_effect = True
        else:
            self.no_action_effect = False
            self.prev_agent_list = []
        self.current_posterior_over_theories = self.compute_posteriors(prev_obs, new_obs, self.current_posterior_over_theories, action)
        #after updating, forget!
        if self.args['infer_mapping'] and self.args['forget_action_mapping']:
            self.forget_action_mapping()

        if self.args['verbose']: self.print_top(self.theories, self.current_posterior_over_theories)

    def get_agent_theories(self, id):
        return self.theories[id]

    def get_agent_mapping_probs(self, id):
        agent_ids = np.array([i_t for i_t in range(self.n_theories) if self.theories[i_t]['agent_id']==id])
        return self.current_posterior_over_theories[agent_ids], agent_ids

    def get_agent_mapping_init_probs(self, id):
        return np.array([p for p, t in zip(self.initial_prior_over_theories,
                                  self.theories) if t['agent_id']==id])

    def forget_action_mapping(self):
        #print(self.current_posterior_over_theories.sum())
        #assert self.current_posterior_over_theories.sum() == 1
        for id in range(self.n_objs):
            theories = self.theories[id]
            probs = self.current_posterior_over_theories[id]
            init_probs = self.initial_prior_over_theories[id]
            final_probs = self.args['mapping_forgetting_factor'] * init_probs + \
            (1 - self.args['mapping_forgetting_factor']) * probs
            self.current_posterior_over_theories[id] = final_probs / final_probs.sum()

    def compute_posteriors(self, prev_obs, new_obs, prior_over_theories, action):
        #belief update for each agent indivually
        #represent the fact that any agent could take your action
        posterior_over_theories = {}
        for id in range(self.n_objs):
            likelihoods = np.array([self.compute_likelihood(theory, id, prev_obs, new_obs, action) for theory in self.theories[id]])
            posterior_over_theories[id] = np.asarray(prior_over_theories[id] * likelihoods).astype('float64')
            if np.max(posterior_over_theories[id]) > 0.99:
                posterior_over_theories[id][np.argmax(posterior_over_theories[id])] = 0.99
            posterior_over_theories[id] = posterior_over_theories[id] / posterior_over_theories[id].sum()
        return posterior_over_theories

    def compute_likelihood(self, theory, agent_id, prev_obs, new_obs, action):
        # probability of data (object positions) given theory and action
        # product of the probabilities of each of the observed movements
        action_dir = theory['input_mapping'][action]
        current_map = prev_obs['map'].copy()
        prev_obj_pos = prev_obs['objects']
        new_obj_pos = new_obs['objects']
        proba_movements = []  # list of probs for each movement

        # compute likelihood of agent movement
        prev_pos = prev_obj_pos[agent_id]
        new_pos = new_obj_pos[agent_id]
        predicted_pos = self.next_obj_pos(prev_pos, action_dir, current_map, True)
        if np.all(predicted_pos == new_pos):
            proba_movements.append(1 - self.get_noise_mean(theory))
        else:
            proba_movements.append(self.get_noise_mean(theory))
        # move the agent in the map
        current_map[prev_pos[0], prev_pos[1]] = 0
        current_map[new_pos[0], new_pos[1]] = 4

        for obj_id, prev_pos, new_pos in zip(range(self.n_objs), prev_obj_pos, new_obj_pos):
            if obj_id != agent_id:
                mvt = new_pos - prev_pos  # observed movement
                if np.sum(np.abs(mvt)) > 0:  # actual movement
                    # the likelihood is the prior for that object and that movement
                    proba_movements.append(self.prior_npc_mvt[obj_id][self.directions_str.index(l2s(mvt))])
                else: # if no actual movement, then it might be because the bot didn't move or because it was blocked
                    # let's sum the prior probabilities of each movement when these movements are blocked by collisions
                    probas_to_sum = [self.prior_npc_mvt[obj_id][-1]]  # start with the proba of no movement
                    for dir in self.directions:
                        dir_prior = self.prior_npc_mvt[obj_id][self.directions_str.index(l2s(dir))]
                        if dir_prior > 0:
                            if np.all(self.next_obj_pos(prev_pos, dir, current_map, False) == new_pos):  # if the expected movement results in new pos
                                probas_to_sum.append(dir_prior)
                    proba_movements.append(np.sum(probas_to_sum))
                # add movement of that object in the map
                current_map[prev_pos[0], prev_pos[1]] = 0
                current_map[new_pos[0], new_pos[1]] = 8
        return np.prod(proba_movements)


    # # # # # # # # # # # #
    # exploitation
    # # # # # # # # # # # #

    def exploit(self, obs, agent_id):
        # Define the heuristic function (Manhattan distance)
        def heuristic_dist(loc, goal_loc):
            return abs(loc[0] - goal_loc[0]) + abs(loc[1] - goal_loc[1])
        
        #move whichever is closest to the goal?
        goal_loc = list(obs['goal'])
        agent_pos = list(obs['objects'][agent_id])

        parents = {}
        parents[tuple(agent_pos)] = None
        g_costs = {}
        g_costs[tuple(agent_pos)] = 0
        # Initialize the open and closed lists
        frontier = [(heuristic_dist(agent_pos, goal_loc), agent_pos)]
        visited = set()

        while frontier:
            #get lowest cost loc in open list
            current_loc = heapq.heappop(frontier)[1]
            #add to closed set
            visited.add(tuple(current_loc))
            #if at goal, return path
            if current_loc == goal_loc:
                path = []
                while current_loc:
                    path.append(current_loc)
                    current_loc = parents[tuple(current_loc)]
                path.reverse()
                return path
            #get possible neighbors
            poss_neighbors = [[current_loc[0]-1, current_loc[1]], [current_loc[0]+1, current_loc[1]],
                                [current_loc[0], current_loc[1]-1], [current_loc[0], current_loc[1]+1]]
            for neighbor in poss_neighbors:
                #check if we can move to this neighbor
                if not(self.env.unwrapped.is_empty(neighbor, agent=True, map=obs['map'])) or tuple(neighbor) in visited:
                    continue
                new_g_cost = g_costs[tuple(current_loc)] + 1  # assuming each step has a cost of 1
                #update 
                if new_g_cost < g_costs.get(tuple(neighbor), float('inf')):
                    g_costs[tuple(neighbor)] = new_g_cost
                    f_cost = g_costs[tuple(neighbor)] + heuristic_dist(neighbor, goal_loc)
                    parents[tuple(neighbor)] = current_loc
                    heapq.heappush(frontier, (f_cost, neighbor))
        # No path found
        print("no path found...")
        print(obs['map'])
        print(agent_pos)
        return None

    def get_action(self, obs, enforce_mode=None):
        #choose an agent to move towards the goal
        #choose the closest one 
        path_lengths = []
        paths = []
        for id in range(self.n_objs):
            path = self.exploit(obs, id)
            paths.append(path)
            if path == None:
                path_lengths.append(float('inf'))
            else:
                path_lengths.append(len(path))
        if self.no_action_effect:
            for a in self.prev_agent_list:
                path_lengths[a] = float('inf')
        agent_id = np.argmin(path_lengths)
        self.prev_agent_list.append(agent_id)
        path = paths[agent_id]
        dir_to_go_str = l2s([path[1][0] - path[0][0], path[1][1] - path[0][1]])
        reverse_mapping = self.get_best_theory(agent_id)['input_reverse_mapping']
        return reverse_mapping[dir_to_go_str], 'exploit'

    # # # # # # # # # # # #
    # utilities
    # # # # # # # # # # # #
    def get_noise_mean(self, theory):
        return self.noise_mean_prior

    def next_obj_pos(self, prev_pos, action_dir, current_map, agent):
        predicted_pos = prev_pos + action_dir

        if agent and self.env.unwrapped.agent_at_goal(map=current_map):
            return prev_pos
        elif self.env.unwrapped.is_empty(predicted_pos, agent=agent, map=current_map):
            return predicted_pos
        else:
            return prev_pos

    def is_agent_mvt_consistent(self, theory, prev_obs, new_obs, action):
        agent_id = theory['agent_id']
        action_dir = theory['input_mapping'][action]
        current_map = prev_obs['map'].copy()
        prev_obj_pos = prev_obs['objects']
        new_obj_pos = new_obs['objects']

        # compute likelihood of agent movement
        prev_pos = prev_obj_pos[agent_id]
        new_pos = new_obj_pos[agent_id]
        predicted_pos = self.next_obj_pos(prev_pos, action_dir, current_map, True)
        return np.all(predicted_pos == new_pos)

    def get_best_theory(self, agent_id):
        theory_id = np.argmax(self.current_posterior_over_theories[agent_id])
        return self.theories[agent_id][theory_id]

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
        print("  Top theories:")
        for agent_id in range(self.n_objs):
            print("agent:", agent_id)
            for _ in range(2):
                id = np.argmax(probs[agent_id])
                print("   action mapping: ", theories[agent_id][id]['input_mapping'], ", prob: ", probs[agent_id][id])
                probs[agent_id][id] = 0

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
        #for i, d in zip(range(data.shape[1]), smooth_data.T):
        #    self.ax.plot(d, linestyle='--', c=COLORS[i])
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
    # errors in js distance with very small numbers
    p0 = [round(x,10) for x in p0]
    p1 = [round(x,10) for x in p1]
    return scipy.spatial.distance.jensenshannon(p0, p1)

def fix_p( p):
    if p.sum() != 1.0:
        p = p*(1./p.sum())
    return p

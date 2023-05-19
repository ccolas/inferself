import numpy as np
import itertools
import math
import scipy
from copy import deepcopy
import gym
import gym_gridworld
from scipy.stats import beta
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
          '#7f7f7f', '#bcbd22', '#17becf']
class InferSelfNoiseless:
    def __init__(self, env, args):
        print("NOOO NOIIIISSSEEEE!!!!!!!!!!")
        self.args = args
        self.n_objs = args['n_objs']
        self.env = env

        self.directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        self.directions_str = [l2s(l) for l in self.directions] # convert directions to string form

        if self.args['bias_bot_mvt'] == 'static':
            self.obj_direction_prior = np.tile(np.append(np.zeros(len(self.directions)), 1),(self.n_objs,1))
        elif self.args['bias_bot_mvt'] == 'uniform':
            self.obj_direction_prior = np.full((self.n_objs, len(self.directions)+1), 1/(len(self.directions)+1))
        else:
            raise NotImplementedError
        self.history_agent_probas = [[1/self.n_objs for _ in range(self.n_objs)]]
        self.reset_memory()
        self.reset_theories()
        self.fig = None
        self.p_change = args['p_change']
        self.pool = Pool(10)

    def reset_theories(self):
        self.time_since_last_reset = 0
        self.theory_found = False
        self.theories = []
        dirs = self.directions_str.copy()
        # list all theories
        for agent_id in range(self.n_objs):
            if self.args['infer_mapping']:
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
                                new_theory = dict(agent_id=agent_id,
                                                  input_mapping={0: dir0l, 1: dir1l, 2: dir2l, 3: dir3l},
                                                  input_reverse_mapping={dir0: 0, dir1: 1, dir2: 2, dir3: 3},
                                                  p_change=self.args['p_change'])
                                self.theories.append(new_theory)
            else:
                dir0, dir1, dir2, dir3 = dirs
                dir0l, dir1l, dir2l, dir3l = [s2l(dir) for dir in [dir0, dir1, dir2, dir3]]  # convert string forms into list forms
                new_theory = dict(agent_id=agent_id,
                                  input_mapping={0: dir0l, 1: dir1l, 2: dir2l, 3: dir3l},
                                  input_reverse_mapping={dir0: 0, dir1: 1, dir2: 2, dir3: 3},
                                  p_change = self.args['p_change'])

                self.theories.append(new_theory)
        self.theories = np.array(self.theories)
        self.probas = np.ones(self.n_theories) / self.n_theories  # uniform probability distribution over theories
        if self.args['biased_input_mapping']:
            self.probas[np.arange(0, self.n_theories, self.n_theories // 4)] *= 50
            self.probas /= self.probas.sum()
        self.prior_probas = self.probas
        self.consistency_record = [[]] * self.n_theories

    def reset_memory(self):
        self.actions = []  # memory of past actions
        self.obj_pos = dict(zip(range(self.n_objs), [[] for _ in range(self.n_objs)]))  # memory of past movements for each object


    def update_theory(self, prev_obs, new_obs, action):
        # store new info
        self.actions.append(action)
        for o_id in range(self.n_objs):
            self.obj_pos[o_id].append(new_obs['objects'][o_id])

        posteriors = self.compute_posteriors(prev_obs, new_obs, self.probas, action)
        self.probas = posteriors

        self.update_history_agent_probas()  # keep track of probabilities for each agent
        best_theory_id = np.argmax(self.probas)
        best_agent_id = self.theories[best_theory_id]['agent_id']
        data = self.get_smooth_agent_probas()

        if sum(self.probas)==0:
            self.history_agent_probas.pop(-1)
            self.reset_theories()
            return self.update_theory(prev_obs, new_obs, action)
        #deal w resetting
        if self.args['explicit_resetting']:
            if data.shape[0] > 10 and self.time_since_last_reset > 3:
                if data[:, best_agent_id][-1] < min(data[:, best_agent_id][-3], data[:, best_agent_id][-2]):  # if drop in belief about agent identity in smooth tracking
                    if self.args['print_status']:
                        print('Drop in the best theory smooth posterior, let\'s reset our theories')
                    self.history_agent_probas.pop(-1)
                    self.reset_theories()
                    return self.update_theory(prev_obs, new_obs, action)
        self.time_since_last_reset += 1
        if self.args['print_status']:
            self.print_top(self.theories, self.probas)
        if self.n_theories == 1:
            if self.args['print_status']:
                if not self.theory_found: print(f'We found the agent with probability 1: it\'s object {self.theories[0]["agent_id"]}, its action mapping is: {self.theories[0]["input_mapping"]}')
            self.theory_found = True
            return self.theories[0], 1
        else:
            theory_id = np.argmax(self.probas)
            return self.theories[theory_id], self.probas[theory_id]

    def update_history_agent_probas(self):
        probas = [0 for _ in range(self.n_objs)]
        for theory, proba in zip(self.theories, self.probas):
            probas[theory['agent_id']] += proba
        assert (np.sum(probas) - 1) < 1e-5
        self.history_agent_probas.append(probas)

    def get_mapping_probas(self):
        mapping_probs = np.zeros((len(self.directions), len(self.directions)))
        #for each action, what's the prob of each direction
        for action_idx in range(4):
            for i_theory, theory in enumerate(self.theories):
                #find in directions
                dir_idx = self.directions.index(theory['input_mapping'][action_idx])
                mapping_probs[action_idx][dir_idx] += self.probas[i_theory]
        return mapping_probs

    def print_top(self, theories, probs):
        probs = probs.copy()
        print("top theories:")
        for _ in range(min(len(probs),5)):
            id = np.argmax(probs)
            print("agent id: ", theories[id]['agent_id'], ", prob: ", probs[id], 'p_change:', theories[id]['p_change'])
            probs[id] = 0

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


    def compute_posteriors(self, prev_obs, new_obs, probas, action):
        # posterior of identity x noise is posterior of identity x posterior(noise | identity)
        # update parameters of the noise distribution for the current theory by incrementing alpha if observation is not consistent
        if self.args['hierarchical']:
            return self.compute_posteriors_hierarchical(prev_obs, new_obs, probas, action)

        # update the posterior for the agent's identity
        posteriors = np.zeros(self.n_theories)
        for i_theory, theory in enumerate(self.theories):
            likelihood = self.compute_likelihood(theory, prev_obs, new_obs, action)
            posterior = probas[i_theory] * (likelihood ** self.args['likelihood_weight'])
            posteriors[i_theory] = posterior
        if posteriors.sum() > 0:
            posteriors = posteriors / posteriors.sum()

        # make sure we don't get perfect confidence because that prevents any further learning
        #if np.max(posteriors) > 0.99:
        #    posteriors[np.argmax(posteriors)] = 0.99
        #    indexes = np.array([i for i in range(len(posteriors)) if i != np.argmax(posteriors)])
        #    posteriors[indexes] = 0.01 / len(indexes)
        return posteriors



    #modify noise to be discrete
    def compute_posteriors_hierarchical(self, prev_obs, new_obs, probas, action):
        p_change_by_theory = []
        #update noise assuming no change
        for i_theory, theory in enumerate(self.theories):
            obs_consistent = self.consistency_record[i_theory] + [int(self.is_agent_mvt_consistent(theory, prev_obs, new_obs, action))]
            new_consistency_record.append(obs_consistent)
            #this automatically takes into account possibility of change or no change
            #hchange_no_change_distribdistribow would we update the distrib if we knew there was no change at the last tpt?
            #i think rAlpha will be the same as rGamma for last tpt
            Alpha, rGamma, rAlpha, rBeta, JumpPost, Trans = ForwardBackward_BernoulliJump(np.array(obs_consistent)+1, self.args['p_change'], self.args['noise_values_discrete'],
                                                                                          self.args['noise_prior_discrete'], 'Backward')
            theory['p_change'] = JumpPost[-1]
            p_change_by_theory.append(theory['p_change'])
            # print(f'New pc = {JumpPost[-1]}')
            #last col of alpha is nans after first run
            no_change_distrib = Alpha[:,0,-1]
            if no_change_distrib.sum() > 0:
                no_change_distrib = no_change_distrib/no_change_distrib.sum()
            change_distrib = Alpha[:,1,-1]
            if change_distrib.sum() > 0:
                change_distrib = change_distrib/change_distrib.sum()
        p_change_by_theory = np.array(p_change_by_theory)
        #now compute probas of theories
        #first compute posterior if no change, based on noise estimate given no change
        no_change_posteriors = np.zeros(self.n_theories)
        for i_theory, theory in enumerate(self.theories):
            theory2 = theory.copy()
            likelihood = self.compute_likelihood(theory, prev_obs, new_obs, action)
            posterior = probas[i_theory] * (likelihood ** self.args['likelihood_weight'])
            no_change_posteriors[i_theory] = posterior

        #first compute posterior if change, based on noise estimate given change
        change_posteriors = np.zeros(self.n_theories)
        for i_theory, theory in enumerate(self.theories):
            theory2 = theory.copy()
            likelihood = self.compute_likelihood(theory2, prev_obs, new_obs, action)
            posterior = self.prior_probas[i_theory] * (likelihood ** self.args['likelihood_weight'])
            change_posteriors[i_theory] = posterior  
        #weighted sum of posteriors in the 2 cases
        nc = np.sum(change_posteriors * p_change_by_theory) + np.sum(no_change_posteriors* (1-p_change_by_theory))
        posteriors = ((change_posteriors * p_change_by_theory) + (no_change_posteriors * (1-p_change_by_theory)))/nc
        return posteriors


    def compute_likelihood(self, theory, prev_obs, new_obs, action):
        # probability of data (object positions) given theory and action
        # product of the probabilities of each of the observed movements
        agent_id = theory['agent_id']
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
            proba_movements.append(1)
        else:
            proba_movements.append(0)
        # move the agent in the map
        current_map[prev_pos[0], prev_pos[1]] = 0
        current_map[new_pos[0], new_pos[1]] = 4

        for obj_id, prev_pos, new_pos in zip(range(self.n_objs), prev_obj_pos, new_obj_pos):
            if obj_id != agent_id:
                mvt = new_pos - prev_pos  # observed movement
                if np.sum(np.abs(mvt)) > 0:  # actual movement
                    # the likelihood is the prior for that object and that movement
                    proba_movements.append(self.obj_direction_prior[obj_id][self.directions_str.index(l2s(mvt))])
                else: # if no actual movement, then it might be because the bot didn't move or because it was blocked
                    # let's sum the prior probabilities of each movement when these movements are blocked by collisions
                    probas_to_sum = [self.obj_direction_prior[obj_id][-1]]  # start with the proba of no movement
                    for dir in self.directions:
                        dir_prior = self.obj_direction_prior[obj_id][self.directions_str.index(l2s(dir))]
                        if dir_prior > 0:
                            if np.all(self.next_obj_pos(prev_pos, dir, current_map, False) == new_pos):  # if the expected movement results in new pos
                                probas_to_sum.append(dir_prior)
                    proba_movements.append(np.sum(probas_to_sum))
                # add movement of that object in the map
                current_map[prev_pos[0], prev_pos[1]] = 0
                current_map[new_pos[0], new_pos[1]] = 8
        return np.prod(proba_movements)

    def get_agent_probabilities(self, theories, probs):
        agent_probs = {}
        for i, t in enumerate(theories):
            id = t['agent_id']
            agent_probs[id] = agent_probs.get(id, 0) + probs[i]
        return agent_probs

    def get_action(self, obs, enforce_mode=None):
        # there are two modes of actions
        # mode 1: the agent tries to infer which object it is and what the action mapping is in an optimal way
        # mode 2: the agent moves towards the goal
        if enforce_mode is None:
            # decide whether to explore or exploit
            agent_probs = self.get_agent_probabilities(self.theories, self.probas)
            if sorted(agent_probs.items(), key=lambda x: x[1], reverse=True)[0][1] >= self.args['threshold']:
            # if np.max(self.probas)>self.args['threshold']:
                mode = 2
            else:
                mode = 1
        else:
            mode = enforce_mode
        # print(np.max(self.probas))

        if self.args['explore_randomly']:
            good_actions_explore = [0, 1, 2, 3]
            action_explore = np.random.choice(good_actions_explore)
        else:
            good_actions_explore, action_explore = self.explore(obs)

        if not self.args['explore_only'] and not enforce_mode==1:
            good_actions_exploit, action_exploit = self.exploit(obs)
        else:
            good_actions_exploit = []
            action_exploit = None
            mode = 1

        # if exploring and several actions are best, take the one advised by exploit
        if mode == 1:
            good_actions = set(good_actions_exploit).intersection(set(good_actions_explore))
            if len(good_actions) > 0:
                if self.args['print_status']:
                    print('explore and exploit')
                action = np.random.choice(sorted(good_actions))
            else:
                if self.args['print_status']:
                    print('explore')
                action = action_explore
        elif mode == 2:
            if self.args['print_status']:
                print('exploit')
            action = action_exploit

        else: raise ValueError
        return action

    def explore(self, prev_obs, look_ahead=False):
        if look_ahead:
            return self.explore_multiple(prev_obs)
        # for each action, compute expected information gain
        action_scores = []
        for action in range(4):
            # simulate possible observations given this theory and action
            # t_init = time.time()
            weighted_obs = self.simulate(prev_obs, action, self.theories, self.probas)
            # print(time.time() - t_init)
            # information gain for each possible observation, weighted by probability
            assert (np.isclose(sum(weighted_obs.values()), 1))
            # t_init = time.time()
            inputs = [(k, v, prev_obs, action, self.probas.copy()) for k, v in weighted_obs.items()]
            #TODO find a way to parallelize this
            # pool = Pool(10)
            # info_gains = pool.map(self.estimate_posteriors, inputs)
            # exp_info_gain = np.sum(info_gains)
            info_gains = [self.estimate_posteriors(input) for input in inputs]
            exp_info_gain = np.sum(info_gains)

            # print(time.time() - t_init)
            action_scores.append(exp_info_gain)
        max_score = np.max(action_scores)
        return np.argwhere(action_scores == max_score).flatten(), np.argmax(action_scores)

    def get_noise_mean(self, x, std=1):
        return ""

    def estimate_posteriors(self, inputs):
        obs_str, obs_prob, prev_obs, action, probas = inputs
        poss_obs = s2dict(obs_str)
        new_probas = self.compute_posteriors(prev_obs, poss_obs, probas, action)
        info_gain = information_gain(probas, new_probas)
        return info_gain * obs_prob

    def explore_multiple(self, prev_obs, n=2, sampling=True):
        # compute expected information gian for n>1 step action sequences
        actions = range(4)
        frontier = [([], dict2s(prev_obs), self.probas, 1)]
        scored_seqs = [frontier]
        for _ in range(n):
            new_frontier = []
            for action in actions:
                # simulate possible observations given this theory and action
                for (action_seq, prev_obs, theory_probas, prev_obs_proba) in frontier:
                    weighted_obs = self.simulate(s2dict(prev_obs), action, self.theories, theory_probas)
                    #update new frontier with action seq, obs
                    if sampling: #sample probable obs 
                        weighted_obs = list(weighted_obs.items())
                        sample_idxs = np.random.choice(np.arange(len(weighted_obs)), p=[x[1] for x in weighted_obs], size=5)
                        for idx in sample_idxs:
                            new_obs = weighted_obs[idx][0]
                            new_obs_proba = 1/len(sample_idxs)
                            new_theory_probas = self.compute_posteriors(s2dict(prev_obs), s2dict(new_obs), theory_probas, action)
                            new_frontier.append((action_seq + [action], new_obs, new_theory_probas, new_obs_proba * prev_obs_proba))
                    else:
                        for new_obs, new_obs_proba in weighted_obs.items(): #integrate over all possible obs
                            new_theory_probas = self.compute_posteriors(s2dict(prev_obs), s2dict(new_obs), theory_probas, action)
                            new_frontier.append((action_seq + [action], new_obs, new_theory_probas, new_obs_proba * prev_obs_proba))
            frontier = new_frontier
            scored_seqs.append(frontier)
        action_seq_scores = {}
        for frontier in scored_seqs: #list of frontiers at each seq length
            for (action_seq, prev_obs, theory_probas, prev_obs_proba) in frontier: #frontier holds action seqs and estimated resulting theory probabilities
                action_seq_str = str(action_seq)
                info_gain = self.information_gain(self.probas, theory_probas)
                action_seq_scores[action_seq_str] = action_seq_scores.get(action_seq_str, 0) + (prev_obs_proba * info_gain)
        action_seq_scores = list(action_seq_scores.items())
        #get action sequences with greatest expected information gain
        max_score = np.max([x[1] for x in action_seq_scores])
        good_seqs = [x[0] for x in action_seq_scores if x[1]==max_score]
        #seq length as a tie breaker
        min_length = np.min([len(seq) for seq in good_seqs])
        good_seqs = [seq for seq in good_seqs if len(seq)==min_length]
        good_actions = [eval(x)[0] for x in good_seqs]
        return good_actions, good_actions[0]


    def get_next_positions(self, obs, movements, agent_id):
        # given each object's intended movement, return object next positions and new map
        map = deepcopy(obs['map'])
        positions = np.full(len(movements), None)
        # update position of the agent
        prev_pos = obs['objects'][agent_id]
        next_pos = self.next_obj_pos(prev_pos, movements[agent_id], map, True)
        positions[agent_id] = next_pos
        map[prev_pos[0], prev_pos[1]] = 0
        map[next_pos[0], next_pos[1]] = 4
        # update positions of the bots
        for i, dir in enumerate(movements):
            if i != agent_id:
                prev_pos = obs['objects'][i]
                next_pos = self.next_obj_pos(prev_pos, dir, map, False)
                positions[i] = next_pos
                map[prev_pos[0], prev_pos[1]] = 0
                map[next_pos[0], next_pos[1]] = 8
        return map, positions

    def next_obj_pos(self, prev_pos, action_dir, current_map, agent):
        predicted_pos = prev_pos + action_dir
        if self.env.unwrapped.is_empty(predicted_pos, agent=agent, map=current_map):
           return predicted_pos
        else:
            return prev_pos

    def simulate(self, prev_obs, action, theories, probas):
        # given all theories about world and this action, what is the probability distribution over possible observations?
        weighted_obs = {}
        if self.args['simulation'] == 'exhaustive':
            # all possible observations given this action, weighted by probability
            for i_theory, theory in enumerate(theories):
                theory_prob = probas[i_theory]
                agent_id = theory['agent_id']
                action_dir = theory['input_mapping'][action]
                # obj positions are non-independent due to collisions
                poss_movements = []
                for obj_id, obj_pos in enumerate(prev_obs['objects']):
                    if obj_id==agent_id:
                        obj_poss_movements = [(action_dir, 1)]
                    else:
                        obj_poss_movements = []
                        for i_dir, dir_prob in enumerate(self.obj_direction_prior[obj_id]):
                            if dir_prob > 0:
                                dir = (self.directions + [0,0])[i_dir]
                                obj_poss_movements.append((dir, dir_prob))
                    poss_movements.append(obj_poss_movements)
                poss_movement_combinations = itertools.product(*poss_movements)
                # record expected obs given each possible combination of obj movements
                for movements_and_probs in poss_movement_combinations:
                    intended_movements = [x[0] for x in movements_and_probs]
                    prob = np.prod([x[1] for x in movements_and_probs])
                    map, positions = self.get_next_positions(prev_obs, intended_movements, agent_id)
                    obs = {'objects': positions, 'map': map, 'goal': prev_obs['goal']}
                    # obs weighted by theory prob * obs prob under theory
                    weighted_obs[dict2s(obs)] = weighted_obs.get(dict2s(obs), 0) + (prob * theory_prob)
        
        elif self.args['simulation'] == 'sampling':
            # sample observations (theory and bot movements) 
            theory_ids = np.random.choice(np.arange(len(theories)), p=probas, size=self.args['n_simulations'])
            prob = 1 / self.args['n_simulations']
            for theory_id in theory_ids:
                theory = theories[theory_id]
                agent_id = theory['agent_id']

                action_dir = theory['input_mapping'][action]
                intended_movements = []
                for obj_id, obj_pos in enumerate(prev_obs['objects']):
                    if obj_id == agent_id:
                        intended_movements.append(action_dir)
                    else:
                        dirs = self.directions + [[0, 0]]
                        i_dir = np.random.choice(np.arange(len(dirs)), p=self.obj_direction_prior[obj_id])
                        intended_movements.append(dirs[i_dir])
                # update agent and bot positions
                map, positions = self.get_next_positions(prev_obs, intended_movements, agent_id)
                obs = {'objects': positions, 'map': map, 'goal': prev_obs['goal']}
                weighted_obs[dict2s(obs)] = weighted_obs.get(dict2s(obs), 0) + prob
        else:
            raise NotImplementedError
        return weighted_obs


    def exploit(self, obs):
        # this implements a greedy strategy towards the goal, given assumptions about the agent identity and the input mapping
        # this only works in non-deceptive worlds (no obstacles)
        if self.n_theories == 1:  # if one theory, pick that one
            i_theory = 0
        else:  # if several theories, sample one according to the probabilities
            i_theory = np.argmax(self.probas)

        agent_id = self.theories[i_theory]['agent_id']
        reverse_mapping = self.theories[i_theory]['input_reverse_mapping']

        # compute direction between the agent and the goal
        agent_pos = obs['objects'][agent_id]
        vector_to_goal = obs['goal'] - agent_pos
        # assert np.sum(np.abs(vector_to_goal)) != 0, "the agent is already on the goal"
        directions = np.sign(vector_to_goal)

        good_actions = []  # there can be up to two good actions (if goal in diagonal)
        if directions[0] != 0:
            dir_to_go_str = l2s([directions[0], 0])
            good_actions.append(reverse_mapping[dir_to_go_str])
        if directions[1] != 0:
            dir_to_go_str = l2s([0, directions[1]])
            good_actions.append(reverse_mapping[dir_to_go_str])

        assert len(good_actions) > 0
        action = np.random.choice(good_actions)

        return good_actions, action

    @property
    def n_theories(self):
        return len(self.theories)

    def get_smooth_agent_probas(self, smooth=5):
        data = np.atleast_2d(np.array(self.history_agent_probas.copy()))
        if smooth is not None:
            smooth_data = np.zeros(data.shape)
            for i in range(data.shape[0]):
                smooth_data[i, :] = np.mean(data[max(0, i - smooth): i+1, :], axis=0)
            data = smooth_data
        return data

    def render(self, true_agent=None, smooth=5):
        data = np.atleast_2d(np.array(self.history_agent_probas.copy()))
        smooth_data = self.get_smooth_agent_probas(smooth=smooth)
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            for i, d in zip(range(data.shape[1]), data.T):
                self.ax.plot(d, c=COLORS[i],  label=f'{i}')
            for i, d in zip(range(data.shape[1]), smooth_data.T):
                self.ax.plot(d, linestyle='--', c=COLORS[i], label=f'{i} smoothed')
            if true_agent is not None:
                self.ax.scatter(data.shape[0] - 1, 1, c=COLORS[true_agent])
            plt.legend()
            plt.ylim([0, 1.05])
            plt.show(block=False)
        if true_agent is not None:
            self.ax.scatter(data.shape[0] - 1, 1, c=COLORS[true_agent])
        for i, d in zip(range(data.shape[1]), data.T):
            self.ax.plot(d, c=COLORS[i])
        for i, d in zip(range(data.shape[1]), smooth_data.T):
            self.ax.plot(d, linestyle='--', c=COLORS[i])
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

if __name__ == '__main__':
    inferselfNoiseless = InferSelfNoiseless()

import numpy as np
import itertools
import math
import scipy
from copy import deepcopy
import gym
import gym_gridworld
from scipy.stats import beta
import matplotlib.pyplot as plt
from hierarchical_scratch import ForwardBackward_BernoulliJump
from multiprocessing import Pool
import time
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
          '#7f7f7f', '#bcbd22', '#17becf']


ARGS = dict(n_objs=4,
            # what to infer
            infer_mapping=False,
            infer_switch=False,
            # priors
            biased_input_mapping=False,
            bias_bot_mvt='uniform', # static or uniform
            p_switch=0.1,
            # learning strategies and biases
            likelihood_weight=1,
            explicit_resetting=False,
            # exploration
            explore_only=False,  # if true, the agent only explores and the goal is removed from the env
            explore_randomly=False,
            simulation='sampling',  # exhaustive or sampling
            n_simulations=10,  # number of simulations if sampling
            # explore-exploit
            explore_exploit_threshold=0.5, # confidence threshold for agent id
            verbose=True,
            )

class InferSelf:
    def __init__(self, env, args):
        self.args = args
        self.n_objs = args['n_objs']
        self.prior_p_switch = args['p_switch']
        self.env = env
        assert 'noise' not in self.env.__str__()
        
        self.directions = [[-1, 0], [1, 0], [0, -1], [0, 1]] # up down left right
        self.directions_str = [l2s(l) for l in self.directions] # convert directions to string form

        if self.args['bias_bot_mvt'] == 'static':
            self.prior_npc_mvt = np.tile(np.append(np.zeros(len(self.directions)), 1), (self.args['n_objs'], 1))
        elif self.args['bias_bot_mvt'] == 'uniform':
            self.prior_npc_mvt = np.full((self.args['n_objs'], len(self.directions) + 1), 1 / (len(self.directions) + 1))
        else: raise NotImplementedError

        self.history_posteriors_over_agents = [[1 /  self.args['n_objs'] for _ in range( self.args['n_objs'])]]  # fill history with prior on agent identity
        self.history_posteriors_p_switch = []
        self.setup_theories()
        self.fig = None
        self.noise_mean_prior = 0.1
        self.forget_action_mappings 
        

    # # # # # # # # # # # # # # # #
    # Setting up inference
    # # # # # # # # # # # # # # # #
    def setup_theories(self):
        self.theories = []
        dirs = self.directions_str.copy()
        # list all theories
        for agent_id in range(self.args['n_objs']):
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
                                                  # noise_params_beta=self.args['noise_prior_beta'].copy(),
                                                  # noise_params_discrete=self.args['noise_prior_discrete'].copy(),
                                                  p_switch=self.args['p_switch'])
                                self.theories.append(new_theory)
            else:
                dir0, dir1, dir2, dir3 = dirs
                dir0l, dir1l, dir2l, dir3l = [s2l(dir) for dir in [dir0, dir1, dir2, dir3]]  # convert string forms into list forms
                new_theory = dict(agent_id=agent_id,
                                  input_mapping={0: dir0l, 1: dir1l, 2: dir2l, 3: dir3l},
                                  input_reverse_mapping={dir0: 0, dir1: 1, dir2: 2, dir3: 3},
                                  # noise_params_beta=self.args['noise_prior_beta'].copy(),
                                  # noise_params_discrete=self.args['noise_prior_discrete'].copy(),
                                  p_switch=self.args['p_switch'])

                self.theories.append(new_theory)
        self.theories = np.array(self.theories)
        self.initial_prior_over_theories = np.ones(self.n_theories) / self.n_theories  # uniform probability distribution over theories
        if self.args['biased_input_mapping']:
            self.initial_prior_over_theories[np.arange(0, self.n_theories, self.n_theories // 4)] *= 50
            self.initial_prior_over_theories /= self.initial_prior_over_theories.sum()
        self.reset_prior_over_theories()

    def reset_prior_over_theories(self):
        self.time_since_last_reset = 0
        self.current_posterior_over_theories = self.initial_prior_over_theories.copy()


    # # # # # # # # # # # # # # # #
    # Running inference
    # # # # # # # # # # # # # # # #
    def update_theory(self, prev_obs, new_obs, action):
        self.current_posterior_over_theories, p_switch = self.compute_posteriors(prev_obs, new_obs, self.current_posterior_over_theories, action)
        self.update_history_posterior_over_agents()
        self.history_posteriors_p_switch.append(p_switch)


        # track smooth posterior over theories and use drops to detect agent switch
        if self.args['explicit_resetting']:
            best_agent_id = self.get_best_theory()['agent_id']
            smooth_posterior_over_theories = self.get_smooth_posterior_over_theories()
            if smooth_posterior_over_theories.shape[0] > 10 and self.time_since_last_reset > 3:
                if smooth_posterior_over_theories[:, best_agent_id][-1] < min(smooth_posterior_over_theories[:, best_agent_id][-3], smooth_posterior_over_theories[:, best_agent_id][-2]):  # if drop in belief about agent identity in smooth tracking
                    if self.args['verbose']: print('Drop in the best theory smooth posterior, let\'s reset our theories')
                    self.history_posteriors_over_agents.pop(-1)
                    self.reset_prior_over_theories()
                    return self.update_theory(prev_obs, new_obs, action)
        self.time_since_last_reset += 1
        if self.args['verbose']: self.print_top(self.theories, self.current_posterior_over_theories)

    def update_history_posterior_over_agents(self):
        posterior_over_agents = [0 for _ in range( self.args['n_objs'])]
        for theory, proba in zip(self.theories, self.current_posterior_over_theories):
            posterior_over_agents[theory['agent_id']] += proba
        assert (np.sum(posterior_over_agents) - 1) < 1e-5
        self.history_posteriors_over_agents.append(posterior_over_agents)

    def compute_posteriors(self, prev_obs, new_obs, prior_over_theories, action):
        # posterior of identity x noise is posterior of identity x posterior(noise | identity)
        # update parameters of the noise distribution for the current theory by incrementing alpha if observation is not consistent
        likelihoods = np.array([self.compute_likelihood(theory, prev_obs, new_obs, action) for theory in self.theories])

        if self.args['infer_switch']:
            # get the matrix of non-diagonal elements
            NonDiag = np.ones((self.n_theories, self.n_theories))
            np.fill_diagonal(NonDiag, 0)
            # Compute the transition matrix (jump = non diagonal transitions).
            # Trans(i,j) is the probability to jump FROM theory i TO j
            # Hence, sum(Trans(i,:)) = 1
            Trans = NonDiag * self.initial_prior_over_theories.copy().reshape(1, -1)
            Trans = Trans / Trans.sum(axis=1).reshape(-1, 1)  # normalize values
            assert np.all(np.isclose(Trans.sum(axis=1), np.ones(Trans.shape[0])))  # check normalization
            # compute probabilities of theories and switch / no switch (given past observations)
            # prior already integrates past observations, so here we're just reweighting the prior for the switch vs no switch cases
            prob_theories_no_switch = (1 - self.prior_p_switch) * prior_over_theories
            prob_theories_switch = self.prior_p_switch * (Trans.T @ prior_over_theories)
            # normalize
            norm_cst = (prob_theories_switch + prob_theories_no_switch).sum()
            prob_theories_switch /= norm_cst
            prob_theories_no_switch /= norm_cst

            # compute proba of observing the new observation after switch / no switch given the theory after the switch
            prob_obs_switch = self.prior_p_switch * likelihoods  # no switch so the likelihood is the one under the previous theory
            # in case of a switch we need to reweight the likelihoods under the distribution of new theories
            # likelihood_given_switch = np.array([np.dot(likelihoods, Trans[i]) for i in range(self.n_theories)])
            # prob_obs_no_switch = self.prior_p_switch * likelihood_given_switch
            prob_obs_no_switch = (1 - self.prior_p_switch) * likelihoods
            # normalize
            norm_cst = (prob_obs_switch + prob_obs_no_switch).sum()
            prob_obs_no_switch /= norm_cst
            prob_obs_switch /= norm_cst
            #
            # prob_obs = prob_obs_no_switch + prob_obs_switch
            # prob_theories = prob_theories_switch + prob_theories_no_switch
            # posterior_over_theories = prob_obs * prob_theories
            # norm_cst = posterior_over_theories.sum()
            # posterior_over_theories /= norm_cst

            # compute posterior on p_switch for that step
            posterior_over_theories_with_switch = prob_theories_switch * prob_obs_switch / self.prior_p_switch
            posterior_over_theories_without_switch = prob_theories_no_switch * prob_obs_no_switch / (1 - self.prior_p_switch)
            norm_cst = (posterior_over_theories_without_switch + posterior_over_theories_with_switch).sum()
            posterior_p_switch = posterior_over_theories_with_switch.sum() / norm_cst
            posterior_over_theories = posterior_over_theories_without_switch + posterior_over_theories_with_switch
            posterior_over_theories / norm_cst
            # stop = 1
        else:
            posterior_p_switch = 0
            posterior_over_theories = prior_over_theories * (likelihoods ** self.args['likelihood_weight'])

        # # update the posterior over theories
        # posteriors = np.zeros(self.n_theories)
        # # print(posterior_p_switch)
        # for i_theory, theory in enumerate(self.theories):
        #     likelihood = self.compute_likelihood(theory, prev_obs, new_obs, action)
        #     posterior = ((1 - posterior_p_switch) * prior_over_theories[i_theory] + posterior_p_switch * self.initial_prior_over_theories[i_theory]) \
        #                 * (likelihood ** self.args['likelihood_weight'])
        #     posteriors[i_theory] = posterior
        # posterior_over_theories = posteriors.copy()

        # fix probabilities: normalize / reset / clip
        if posterior_over_theories.sum() > 0:
            posterior_over_theories = np.asarray(posterior_over_theories).astype('float64')
            posterior_over_theories = posterior_over_theories / posterior_over_theories.sum()  # normalize


        # make sure we don't get perfect confidence because that prevents any further learning
        if np.max(posterior_over_theories) > 0.99:
            posterior_over_theories[np.argmax(posterior_over_theories)] = 0.99
            indexes = np.array([i for i in range(len(posterior_over_theories)) if i != np.argmax(posterior_over_theories)])
            posterior_over_theories[indexes] = 0.01 / len(indexes)
            posterior_over_theories = np.asarray(posterior_over_theories).astype('float64')
            posterior_over_theories /= posterior_over_theories.sum()  # normalize
        posterior_over_theories = np.round(posterior_over_theories, 5)
        posterior_over_theories = fix_p(posterior_over_theories)
        if posterior_over_theories.sum() != 1:
            stop = 1
        return posterior_over_theories, posterior_p_switch

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
            proba_movements.append(1 - self.get_noise_mean(theory))
        else:
            proba_movements.append(self.get_noise_mean(theory))
        # move the agent in the map
        current_map[prev_pos[0], prev_pos[1]] = 0
        current_map[new_pos[0], new_pos[1]] = 4

        for obj_id, prev_pos, new_pos in zip(range( self.args['n_objs']), prev_obj_pos, new_obj_pos):
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
    # exploration
    # # # # # # # # # # # #

    def explore(self, prev_obs):
        # for each action, compute expected information gain
        action_scores = []
        for action in range(4):
            # simulate possible observations given this theory and action
            weighted_obs = self.simulate(prev_obs, action, self.theories, self.current_posterior_over_theories)
            # information gain for each possible observation, weighted by probability
            assert (np.isclose(sum(weighted_obs.values()), 1))
            inputs = [(k, v, prev_obs, action, self.current_posterior_over_theories.copy()) for k, v in weighted_obs.items()]
            info_gains = [self.estimate_info_gain(input) for input in inputs]
            exp_info_gain = np.sum(info_gains)
            action_scores.append(exp_info_gain)
        max_score = np.max(action_scores)
        return np.argwhere(action_scores == max_score).flatten()

    def estimate_info_gain(self, inputs):
        obs_str, obs_prob, prev_obs, action, probas = inputs
        poss_obs = s2dict(obs_str)
        new_probas, p_switche = self.compute_posteriors(prev_obs, poss_obs, probas, action)
        info_gain = information_gain(probas, new_probas)
        return info_gain * obs_prob

    def simulate(self, prev_obs, action, theories, probas):
        # given all theories about world and this action, what is the probability distribution over possible observations?
        weighted_obs = {}

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
                    #should sample this noise value?
                    if np.random.rand() < self.get_noise_mean(theory):
                        candidate_actions = sorted(set(range(5)) - set([action]))
                        effective_action = np.random.choice(candidate_actions)
                        if effective_action < 4:
                            action_dir = theory['input_mapping'][effective_action]
                        else:
                            action_dir = np.zeros(2)
                    else:
                        action_dir = theory['input_mapping'][action]
                    intended_movements.append(action_dir)
                else:
                    dirs = self.directions + [[0, 0]]
                    i_dir = np.random.choice(np.arange(len(dirs)), p=self.prior_npc_mvt[obj_id])
                    intended_movements.append(dirs[i_dir])
            # update agent and bot positions
            map, positions = self.get_next_positions(prev_obs, intended_movements, agent_id)
            obs = {'objects': positions, 'map': map, 'goal': prev_obs['goal']}
            weighted_obs[dict2s(obs)] = weighted_obs.get(dict2s(obs), 0) + prob

        return weighted_obs


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


    # # # # # # # # # # # #
    # exploitation
    # # # # # # # # # # # #

    def exploit(self, obs):
        # this implements a greedy strategy towards the goal, given assumptions about the agent identity and the input mapping
        # this only works in non-deceptive worlds (no obstacles)
        theory = self.get_best_theory()
        agent_id = theory['agent_id']
        reverse_mapping = theory['input_reverse_mapping']
        # compute direction between the agent and the goal
        agent_pos = obs['objects'][agent_id]
        vector_to_goal = obs['goal'] - agent_pos
        directions = np.sign(vector_to_goal)
        good_actions = []  # there can be up to two good actions (if goal in diagonal)
        if directions[0] != 0:
            dir_to_go_str = l2s([directions[0], 0])
            good_actions.append(reverse_mapping[dir_to_go_str])
        if directions[1] != 0:
            dir_to_go_str = l2s([0, directions[1]])
            good_actions.append(reverse_mapping[dir_to_go_str])
        assert len(good_actions) > 0
        return good_actions

    def get_action(self, obs, enforce_mode=None):
        # there are two modes of actions
        # mode explore: the agent tries to infer which object it is and what the action mapping is in an optimal way
        # mode exploit: the agent moves towards the goal
        if enforce_mode is None:
            # decide whether to explore or exploit
            posterior_over_agents = self.get_posterior_over_agents(self.theories, self.current_posterior_over_theories)
            if sorted(posterior_over_agents.items(), key=lambda x: x[1], reverse=True)[0][1] >= self.args['explore_exploit_threshold']:
                mode = "exploit"
            else:
                mode = "explore"
        else:
            mode = enforce_mode

        if mode == 'explore':
            if self.args['explore_randomly']:
                good_actions_explore = [0, 1, 2, 3]
            else:
                good_actions_explore = self.explore(obs)
            if self.args['explore_only']:
                action = np.random.choice(good_actions_explore)
                if self.args['verbose']: print('  explore')
            else:
                good_actions_exploit = self.exploit(obs)
                good_actions = set(good_actions_exploit).intersection(set(good_actions_explore))
                if len(good_actions) > 0:
                    if self.args['verbose']: print('  explore and exploit')
                    action = np.random.choice(sorted(good_actions))
                else:
                    if self.args['verbose']: print('  explore')
                    action = np.random.choice(good_actions_explore)
        elif mode == 'exploit':
            if self.args['verbose']: print('  exploit')
            good_actions_exploit = self.exploit(obs)
            action = np.random.choice(good_actions_exploit)
        else:
            raise NotImplementedError
        return action

    # # # # # # # # # # # #
    # utilities
    # # # # # # # # # # # #
    def get_noise_mean(self, theory):
        return self.noise_mean_prior

    def next_obj_pos(self, prev_pos, action_dir, current_map, agent):
        predicted_pos = prev_pos + action_dir
        if self.env.unwrapped.is_empty(predicted_pos, agent=agent, map=current_map):
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

    def get_best_theory(self, get_proba=False):
        theory_id = np.argmax(self.current_posterior_over_theories)
        if get_proba:
            return  self.theories[theory_id], self.current_posterior_over_theories[theory_id]
        else:
            return self.theories[theory_id]

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
        agent_probs = {}
        for i, t in enumerate(theories):
            id = t['agent_id']
            agent_probs[id] = agent_probs.get(id, 0) + probs[i]
        return agent_probs

    def print_top(self, theories, probs):
        probs = probs.copy()
        print("  Top theories:")
        for _ in range(min(len(probs), 5)):
            id = np.argmax(probs)
            print("    agent id: ", theories[id]['agent_id'], ", prob: ", probs[id], 'p_switch:', theories[id]['p_switch'])
            probs[id] = 0

    def get_mapping_probas(self):
        mapping_probs = np.zeros((len(self.directions), len(self.directions)))
        #for each action, what's the prob of each direction
        for action_idx in range(4):
            for i_theory, theory in enumerate(self.theories):
                #find in directions
                dir_idx = self.directions.index(theory['input_mapping'][action_idx])
                mapping_probs[action_idx][dir_idx] += self.current_posterior_over_theories[i_theory]
        return mapping_probs

    def render(self, true_agent=None, smooth=5):
        data = np.atleast_2d(np.array(self.history_posteriors_over_agents.copy()))

        smooth_data = self.get_smooth_posterior_over_theories(smooth=smooth)

        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            for i, d in zip(range(data.shape[1]), data.T):
                self.ax.plot(d, c=COLORS[i],  label=f'{i}')
            for i, d in zip(range(data.shape[1]), smooth_data.T):
                self.ax.plot(d, linestyle='--', c=COLORS[i], label=f'{i} smoothed')
            if true_agent is not None:
                self.ax.scatter(data.shape[0] - 1, 1, c=COLORS[true_agent])
            self.ax.plot(self.history_posteriors_p_switch, color='k', label='p_switch')
            plt.legend()
            plt.ylim([0, 1.05])
            plt.show(block=False)
        if true_agent is not None:
            self.ax.scatter(data.shape[0] - 1, 1, c=COLORS[true_agent])
        for i, d in zip(range(data.shape[1]), data.T):
            self.ax.plot(d, c=COLORS[i])
        self.ax.plot(self.history_posteriors_p_switch, color='k')
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

def fix_p( p):
    if p.sum() != 1.0:
        p = p*(1./p.sum())
    return p

if __name__ == '__main__':
    inferself = InferSelf()

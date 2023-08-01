import numpy as np
import scipy
from copy import deepcopy
import matplotlib.pyplot as plt
import heapq


#track goal for each possible self
#update theory as we go
#once we choose one, when choosing an action, expected reward is the same for each, go by steps to reward?
#min path to each goal?

#we should not be passing in env
#we should pass in map, in the way in which it's observable to the agent

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
          '#7f7f7f', '#bcbd22', '#17becf']

class InferSelf:
    def __init__(self, obs, args):
        self.args = args
        self.prior_p_switch = args['p_switch']
        self.n_objs = np.count_nonzero(obs['map'] == 8)#get from obs
        self.directions = [[-1, 0], [1, 0], [0, -1], [0, 1]] # up down left right
        self.directions_str = [l2s(l) for l in self.directions] # convert directions to string form

        if self.args['bias_bot_mvt'] == 'static':
            self.prior_npc_mvt = np.tile(np.append(np.zeros(len(self.directions)), 1), (self.n_objs, 1))
        elif self.args['bias_bot_mvt'] == 'uniform':
            self.prior_npc_mvt = np.full((self.n_objs, len(self.directions) + 1), 1 / (len(self.directions) + 1))
        else: raise NotImplementedError

        self.history_posteriors_over_agents = [[1 /  self.n_objs for _ in range( self.n_objs)]]  # fill history with prior on agent identity
        self.history_posteriors_p_switch = []
        self.setup_theories()
        self.fig = None
        self.noise_mean_prior = self.args['noise_prior']
        
        #right now, theories includes each self/action mapping pair
        #let's also have possible beliefs, which tracks for each theory, possible beliefs under that theory (ePOMDP)
        self.setup_poss_beliefs(obs)
        
    def setup_poss_beliefs(self, obs):
        self.poss_beliefs = [{}] * len(self.theories)
        for bel in self.poss_beliefs:
            goal_locs = np.transpose(np.nonzero(obs['map']==4))
            bel['goal_loc'] = [goal_locs, [1/len(goal_locs) for _ in goal_locs]]

    # # # # # # # # # # # # # # # #
    # Setting up inference
    # # # # # # # # # # # # # # # #
    def setup_theories(self):
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
        if self.args['biased_action_mapping']:
            self.initial_prior_over_theories[np.arange(0, self.n_theories, self.n_theories // 4)] *= self.args['biased_action_mapping_factor']
            self.initial_prior_over_theories /= self.initial_prior_over_theories.sum()
        self.initial_prior_over_theories = np.log(self.initial_prior_over_theories)
        self.reset_prior_over_theories()
        self.update_objs_attended_to()

    def update_objs_attended_to(self):
        if self.args['attention_bias']:
            if self.args['n_objs_attended_to'] == 0:
                if np.random.rand() < 0.9:
                     self.objs_attended_to = np.random.choice(range(self.n_objs),
                                                    p=np.exp(self.get_posterior_over_agents(self.theories, self.current_posterior_over_theories)),
                                                    size=1,
                                                    replace=False)
                else:
                    self.objs_attended_to = []
            else:
                if self.args['uniform_attention_bias']: 
                    probas = [1/self.n_objs] * self.n_objs
                else:
                    probas = np.exp(self.get_posterior_over_agents(self.theories, self.current_posterior_over_theories))
                self.objs_attended_to = np.random.choice(range(self.n_objs),
                                                    p=probas,
                                                    size=self.args['n_objs_attended_to'],
                                                    replace=False)
        else:
            self.objs_attended_to = range(self.n_objs)

    def reset_prior_over_theories(self):
        self.time_since_last_reset = 0
        self.current_posterior_over_theories = self.initial_prior_over_theories.copy()

    # # # # # # # # # # # # # # # #
    # Running inference
    # # # # # # # # # # # # # # # #
    def update_theory(self, prev_obs, new_obs, action):
        #update theory of self / aptness of ePOMDPs
        self.current_posterior_over_theories, p_switch = self.compute_posteriors(prev_obs, new_obs, self.current_posterior_over_theories, action)
        #update beliefs within dif ePOMDPs
        #self.poss_beliefs = self.compute_poss_beliefs(prev_obs, new_obs, self.current_posterior_over_theories, action)
        #after updating, forget!
        if self.args['infer_mapping'] and self.args['forget_action_mapping']:
            self.forget_action_mapping()

        self.update_history_posterior_over_agents()
        self.history_posteriors_p_switch.append(p_switch)

        self.update_objs_attended_to()

        self.time_since_last_reset += 1
        if self.args['verbose']: self.print_top(self.theories, np.exp(self.current_posterior_over_theories))

    def get_agent_theories(self, id):
        return [t for t in self.theories if t['agent_id']==id]

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
            theories = self.get_agent_theories(id)
            probs, agent_ids = self.get_agent_mapping_probs(id)
            init_probs = self.get_agent_mapping_init_probs(id)
            agent_prob = probs.sum()
            init_probs = init_probs / init_probs.sum() * agent_prob
            assert np.all(np.isclose(init_probs.sum(), agent_prob))
            final_probs = self.args['mapping_forgetting_factor'] * init_probs + \
            (1 - self.args['mapping_forgetting_factor']) * probs
            if agent_prob > 0:
                final_probs = final_probs / final_probs.sum() * agent_prob
            #agent_prob = np.logsumexp(probs)
            #init_probs = init_probs - (np.logsumexp(init_probs) + agent_prob)
            #assert np.all(np.isclose(np.logsumexp(init_probs), agent_prob))
            #final_probs = self.args['mapping_forgetting_factor'] * init_probs + \
            #(1 - self.args['mapping_forgetting_factor']) * probs
            #try:
            #    final_probs = final_probs / final_probs.sum() * agent_prob
            #    assert np.all(np.isclose(final_probs.sum(), agent_prob))
            #except:
            #    print(agent_prob)
            #    print(final_probs.sum())
            assert np.all(np.isclose(final_probs.sum(), agent_prob))
            self.current_posterior_over_theories[agent_ids] = final_probs
        # print(self.current_posterior_over_theories.sum())
        # assert self.current_posterior_over_theories.sum() == 1
#array([0.02281058, 0.12876011, 0.84842931])
    def update_history_posterior_over_agents(self):
        posterior_over_agents = [0 for _ in range( self.n_objs)]
        for theory, proba in zip(self.theories, self.current_posterior_over_theories):
            posterior_over_agents[theory['agent_id']] += proba
        assert (np.sum(posterior_over_agents) - 1) < 1e-5
        self.history_posteriors_over_agents.append(posterior_over_agents)

    def compute_poss_beliefs(self, prev_obs, new_obs, post_over_theories, action):
        #should update for each theory?
        for i, theory in enumerate(self.theories):
            poss_bel = self.poss_beliefs[i]
            [locs, prior] = poss_bel['goal_loc']
            #should pass in whole bel to lik, find joint lik of this happening
            #poss_bel might include beliefs abt many dif things, should sample to determine lik?
            #then, just compute prior * lik for each piece...
            #same lik function as for theory, but now assume one self
            

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
        if agent_id in self.objs_attended_to:
            if np.all(predicted_pos == new_pos):
                proba_movements.append(1 - self.get_noise_mean(theory))
            else:
                proba_movements.append(self.get_noise_mean(theory))
        # move the agent in the map
        current_map[prev_pos[0], prev_pos[1]] = 0
        current_map[new_pos[0], new_pos[1]] = 4


        for obj_id, prev_pos, new_pos in zip(range( self.n_objs), prev_obj_pos, new_obj_pos):
            if obj_id != agent_id:
                mvt = new_pos - prev_pos  # observed movement
                if obj_id in self.objs_attended_to:
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
        #return np.argwhere(action_scores >= max_score*0.95).flatten()

    def estimate_info_gain(self, inputs):
        obs_str, obs_prob, prev_obs, action, probas = inputs
        poss_obs = s2dict(obs_str)
        new_probas, p_switch = self.compute_posteriors(prev_obs, poss_obs, probas, action)
        info_gain = information_gain(probas, new_probas)
        
        #probas over agents instead of agent x action mapping:
        #agent_probas = get_posterior_over_agents(self.theories, probas)
        #new_agent_probas = get_posterior_over_agents(self.theories, new_probas)
        info_gain = information_gain(agent_probas, new_agent_probas)
        
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

    def exploit_prev(self, obs):
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

    def exploit(self, obs):
        # Define the heuristic function (Manhattan distance)
        def heuristic_dist(loc, goal_loc):
            return abs(loc[0] - goal_loc[0]) + abs(loc[1] - goal_loc[1])
        
        theory = self.get_best_theory()
        agent_id = theory['agent_id']
        reverse_mapping = theory['input_reverse_mapping']
        agent_pos = list(obs['objects'][agent_id])
        goal_loc = list(obs['goal'])

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
                dir_to_go_str = l2s([path[1][0] - path[0][0], path[1][1] - path[0][1]])
                return [reverse_mapping[dir_to_go_str]]
            #get possible neighbors
            poss_neighbors = [[current_loc[0]-1, current_loc[1]], [current_loc[0]+1, current_loc[1]],
                                [current_loc[0], current_loc[1]-1], [current_loc[0], current_loc[1]+1]]
            for neighbor in poss_neighbors:
                #check if we can move to this neighbor
                if not(self.is_empty(obs['map'], neighbor, agent=True)) or tuple(neighbor) in visited:
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
        return [0]


    def do_explore(self):
        posterior_over_agents = self.get_posterior_over_agents(self.theories, self.current_posterior_over_theories)
        #return(np.max(posterior_over_agents) < self.args['explore_exploit_threshold'])
        top = sorted(posterior_over_agents, reverse=True)[:2]
        return(top[0]/top[1] < 1.5)

    def get_action(self, obs, enforce_mode=None):
        # there are two modes of actions
        # mode explore: the agent tries to infer which object it is and what the action mapping is in an optimal way
        # mode exploit: the agent moves towards the goal
        if enforce_mode is None:
            # decide whether to explore or exploit
            if self.do_explore():
                mode = "explore"
            else:
                mode = "exploit"
        else:
            mode = enforce_mode

        if mode == 'explore':
            if self.args['explore_randomly']:
                good_actions_explore = [0, 1, 2, 3]
                action = np.random.choice(good_actions_explore)
            else:
                good_actions_explore = self.explore(obs)
            if self.args['explore_only']:
                action = np.random.choice(good_actions_explore)
                if self.args['verbose']: print('  explore')
            else:
                if not self.args['explore_randomly']:
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
        return action, mode
    
    # # # # # # # # # # # #
    # utilities
    # # # # # # # # # # # #
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

    #based off current beliefs about what the goal is
    def agent_at_goal(self, current_map):
        if self.mode == 'explore':
            return False
        else:
            return False
            #if self.poss_bel[]

    def is_empty(self, map, loc, agent=True):
        #can only move to goal if agent
        if agent: empty = [0, 3]
        else: empty = [0]
        #check for out of bounds
        if (int(loc[0]) < 0) or (int(loc[0]) >= np.shape(map)[0]) or (int(loc[1]) < 0) or (int(loc[1]) >= np.shape(map)[1]):
            return False
        return map[int(loc[0]), int(loc[1])] in empty
        

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
    @property
    def mode(self):
        posterior_over_agents = self.get_posterior_over_agents(self.theories, self.current_posterior_over_theories)
        #return(np.max(posterior_over_agents) < self.args['explore_exploit_threshold'])
        top = sorted(posterior_over_agents, reverse=True)[:2]
        if top[0]/top[1] < 1.5:
            return 'explore'
        else:
            return 'exploit'

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
                if i >= len(COLORS):
                    col = '#2ca02c'
                else:
                    col = COLORS[i]
                self.ax.plot(d, c=col,  label=f'{i}')
            #for i, d in zip(range(data.shape[1]), smooth_data.T):
            #    self.ax.plot(d, linestyle='--', c=COLORS[i], label=f'{i} smoothed')
            if true_agent is not None:
                self.ax.scatter(data.shape[0] - 1, 1, c=COLORS[true_agent])
            self.ax.plot(self.history_posteriors_p_switch, color='k', label='p_switch')
            plt.legend()
            plt.ylim([0, 1.05])
            plt.show(block=False)
        if true_agent is not None:
            self.ax.scatter(data.shape[0] - 1, 1, c=COLORS[true_agent])
        for i, d in zip(range(data.shape[1]), data.T):
            if i >= len(COLORS):
                col = '#2ca02c'
            else:
                col = COLORS[i]
            self.ax.plot(d, c=col)
        self.ax.plot(self.history_posteriors_p_switch, color='k')
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


import numpy as np
import itertools
import scipy
import gym
import gym_gridworld
from copy import deepcopy

class InferSelf:
    def __init__(self, env, args):
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
        self.reset_memory()
        self.reset_theories()

    def reset_theories(self):
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
                                                  input_reverse_mapping={dir0: 0, dir1: 1, dir2: 2, dir3: 3})
                                self.theories.append(new_theory)
            else:
                dir0, dir1, dir2, dir3 = dirs
                dir0l, dir1l, dir2l, dir3l = [s2l(dir) for dir in [dir0, dir1, dir2, dir3]]  # convert string forms into list forms
                new_theory = dict(agent_id=agent_id,
                                  input_mapping={0: dir0l, 1: dir1l, 2: dir2l, 3: dir3l},
                                  input_reverse_mapping={dir0: 0, dir1: 1, dir2: 2, dir3: 3})
                self.theories.append(new_theory)
        self.theories = np.array(self.theories)
        self.probas = np.ones(self.n_theories) / self.n_theories  # uniform probability distribution over theories
        if self.args['biased_input_mapping']:
            self.probas[np.arange(0, self.n_theories, self.n_theories // 4)] *= 1000
            self.probas /= self.probas.sum()

    def reset_memory(self):
        self.actions = []  # memory of past actions
        self.obj_pos = dict(zip(range(self.n_objs), [[] for _ in range(self.n_objs)]))  # memory of past movements for each object

    # likelihood: prob of new_obs from prev_obs given theory
    def update(self, action, prev_obs, new_obs, hypothetical=False):
        # store new info
        self.actions.append(action)
        for o_id in range(self.n_objs):
            self.obj_pos[o_id].append(new_obs['objects'][o_id])

        posteriors = np.zeros(self.n_theories)
        # run inference for the new datapoint
        to_keep = []  # we keep track of theories with proba > 0
        for i_theory, theory in enumerate(self.theories):
            likelihood = self.compute_likelihood(theory, prev_obs, new_obs, action)
            posterior = self.probas[i_theory] * likelihood
            posteriors[i_theory] = posterior
            if posterior > 0:
                to_keep.append(i_theory)
        to_keep = np.array(to_keep)

        if hypothetical:
            return posteriors / posteriors.sum()
        # print(posteriors)
        self.probas = posteriors

        # delete theories with proba = 0
        if len(to_keep) == 0:  # something weird happened (eg avatar switch)
            print('Past evidence is not consistent, let\'s reset the theories')
            self.reset_theories()
        else:
            if len(to_keep) == len(self.probas):
                if not self.theory_found: print(f'  new datapoint ingested, we didn\'t learn anything here')
            else:
                self.theories = self.theories[to_keep]
                self.probas = self.probas[to_keep]
                if not self.theory_found:  print(f'  new datapoint ingested, we now have {self.n_theories} theories')
            self.probas = self.probas / self.probas.sum()  # renormalize the probabilities

        if self.n_theories == 1:
            if not self.theory_found: print(f'We found the agent with probability 1: it\'s object {self.theories[0]["agent_id"]}, its action mapping is: {self.theories[0]["input_mapping"]}')
            self.theory_found = True
            return self.theories[0], 1
        else:
            theory_id = np.random.choice(np.arange(self.n_theories), p=self.probas)
            return self.theories[theory_id], self.probas[theory_id]

    def compute_posteriors(self, action, prev_obs, new_obs):
        posteriors = np.zeros(self.n_theories)
        # run inference for the new datapoint
        for i_theory, theory in enumerate(self.theories):
            likelihood = self.compute_likelihood(theory, prev_obs, new_obs, action)
            posterior = self.probas[i_theory] * likelihood
            posteriors[i_theory] = posterior
        return posteriors / posteriors.sum()

    def compute_likelihood(self, theory, prev_obs, new_obs, action):
        # probability of data given theory
        # product of the probability of observing each of the observed movements
        agent_id = theory['agent_id']
        action_dir = theory['input_mapping'][action]
        current_map = prev_obs['map'].copy()
        prev_obs = prev_obs['objects']
        new_obs = new_obs['objects']
        proba_movements = []  # list of probs for each movement

        # compute likelihood of agent movement
        prev_pos = prev_obs[agent_id]
        next_pos = new_obs[agent_id]
        predicted_pos = prev_pos + action_dir
        if not self.env.unwrapped.is_empty(predicted_pos, agent=True, map=current_map):
            predicted_pos = prev_pos
        if not np.all(predicted_pos == next_pos):
            return 0  # if theory cannot predict agent behavior, then likelihood is 0
        else:
            proba_movements.append(1)
        # move the agent in the map
        current_map[next_pos[0], next_pos[1]] = 4
        current_map[prev_pos[0], prev_pos[1]] = 0

        count_non_agent = 0
        for obj_id, obj_prev_pos, obj_new_pos in zip(range(self.n_objs), prev_obs, new_obs):
            if obj_id != agent_id:
                mvt = obj_new_pos - obj_prev_pos  # observed movement
                if np.sum(np.abs(mvt)) > 0:  # actual movement
                    # the likelihood is the prior for that object and that movement
                    proba_movements.append(self.obj_direction_prior[count_non_agent][self.directions_str.index(l2s(mvt))])
                else: # if no actual movement, then it might be because the bot didn't move or because it was blocked
                    # let's sum the prior probabilities of each movement when these movements are blocked by collisions
                    probas_to_sum = [self.obj_direction_prior[count_non_agent][-1]]  # start with the proba of no movement
                    for dir in self.directions:
                        if self.obj_direction_prior[count_non_agent][self.directions_str.index(l2s(dir))] > 0:
                            if not self.env.unwrapped.is_empty(obj_prev_pos + dir, agent=False, map=current_map):  # if the expected movement was possible, p=0
                                probas_to_sum.append(self.obj_direction_prior[count_non_agent][self.directions_str.index(l2s(dir))])
                    proba_movements.append(np.sum(probas_to_sum))
                # add movement of that object in the map
                current_map[obj_new_pos[0], obj_new_pos[1]] = 8
                current_map[obj_prev_pos[0], obj_prev_pos[1]] = 0
                count_non_agent += 1
        return np.prod(proba_movements)


    def get_action(self, obs, mode=None):
        # there are two modes of actions
        # mode 1: the agent tries to infer which object it is and what the action mapping is in an optimal way
        # mode 2: the agent moves towards the goal
        if mode is None:
            if self.n_theories == 1: mode = 2
            else: mode = 1

        good_actions_explore, action_explore = self.explore(obs)
        good_actions_exploit, action_exploit = self.exploit(obs)

        # if exploring and several actions are best, take the one advised by exploit
        if mode == 1:
            good_actions = set(good_actions_exploit).intersection(set(good_actions_explore))
            if len(good_actions) > 0:
                action = np.random.choice(sorted(good_actions))
            else:
                action = action_explore
        elif mode == 2:
            action = action_exploit

        else: raise ValueError
        return action

    def explore(self, prev_obs):
        # for each action, compute expected information gain
        action_scores = []
        for action in range(4):
            # simulate possible observations given this theory and action
            weighted_obs = self.simulate(prev_obs, action)
            # information gain for each possible observation, weighted by probability
            assert (np.isclose(sum(weighted_obs.values()), 1))
            exp_info_gain = 0
            # print(action)
            for obs_str, obs_prob in weighted_obs.items():
                poss_obs = s2dict(obs_str)
                new_probas = self.compute_posteriors(action, prev_obs, poss_obs)
                info_gain = self.information_gain(self.probas, new_probas)
                exp_info_gain += info_gain * obs_prob
            #     print(obs_str.split('objects')[1], obs_prob, info_gain * obs_prob)
            # print(exp_info_gain)
            action_scores.append(exp_info_gain)
        max_score = np.max(action_scores)
        return np.argwhere(action_scores == max_score).flatten(), np.argmax(action_scores)


    def information_gain(self, p0, p1):
        # errors in js distance with very small numbers
        p0 = [round(x,5) for x in p0]
        p1 = [round(x,5) for x in p1]
        return scipy.spatial.distance.jensenshannon(p0, p1)

    # given these actions and these objects, return directions of movemement
    # thinking we might want a function like this (to run in update as well?) bc movememnts may be non-independent
    # ie if two objects run into each other
    # and for cases where we want our theory to include information about how all objects move
    def get_next_positions(self, obs, movements, agent_id, map):
        positions = [None for _ in range(len(movements))]

        # update position of the agent
        prev_pos_agent = obs['objects'][agent_id]
        next_pos_agent = prev_pos_agent + movements[agent_id]
        if not self.env.unwrapped.is_empty(next_pos_agent, agent=True, map=map):
            next_pos_agent = prev_pos_agent
        else:
            # if agent moved, update the map
            map[next_pos_agent[0], next_pos_agent[1]] = 4
            map[prev_pos_agent[0], prev_pos_agent[1]] = 0
        positions[agent_id] = next_pos_agent

        # update positions of the bots
        for i, dir in enumerate(movements):
            if i != agent_id:
                prev_pos = obs['objects'][i]
                next_pos = prev_pos + dir
                if not self.env.unwrapped.is_empty(next_pos, agent=True, map=map):
                    next_pos = prev_pos
                else:
                    # update the map
                    map[next_pos[0], next_pos[1]] = 8
                    map[prev_pos[0], prev_pos[1]] = 0
                positions[i] = next_pos
        return map, positions

    # Given this theory and this action, what is the probability distribution over possible observations?
    def simulate(self, prev_obs, action):
        prev_obs = deepcopy(prev_obs)
        weighted_obs = {}
        if action == 1:
            stop = 1
        if self.args['simulation'] == 'exhaustive':
            # possible observations given this action, weighted by probability
            # init_time = time.time()
            for i_theory, theory in enumerate(self.theories):
                theory_prob = self.probas[i_theory]

                agent_id = theory['agent_id']
                action_dir = theory['input_mapping'][action]
                current_map = prev_obs['map']

                # not specified by theory: movements of other objects
                poss_movements = []
                for i, obj_pos in enumerate(prev_obs['objects']):
                    if i==agent_id:
                        obj_poss_movements = [(action_dir, 1)]
                    else:
                        obj_poss_movements = []
                        for dir_idx, p in enumerate(self.obj_direction_prior[i]):
                            if p > 0:
                                if dir_idx >= len(self.directions):
                                    dir = [0,0]
                                else:
                                    dir = self.directions[dir_idx]
                                obj_poss_movements.append((dir, p))
                    poss_movements.append(obj_poss_movements)
                poss_movements_and_probs = itertools.product(*poss_movements)

                for movements_and_probs in poss_movements_and_probs:
                    movements = [x[0] for x in movements_and_probs]
                    prob = np.prod([x[1] for x in movements_and_probs]) * theory_prob
                    assert(prob > 0)
                    map, positions = self.get_next_positions(deepcopy(prev_obs), movements, agent_id, current_map.copy())
                    obs = {'objects': positions.copy(), 'map': map, 'goal': prev_obs['goal']}
                    weighted_obs[dict2s(obs)] = weighted_obs.get(dict2s(obs), 0) + prob
            # print(time.time() - init_time)

        elif self.args['simulation'] == 'sampling':
            theory_ids = np.random.choice(np.arange(self.n_theories), p=self.probas, size=self.args['n_simulations'])
            prob = 1 / self.args['n_simulations']
            # init_time = time.time()
            for theory_id in theory_ids:
                theory = self.theories[theory_id]
                agent_id = theory['agent_id']
                action_dir = theory['input_mapping'][action]
                current_map = prev_obs['map'].copy()

                # sample agent pos
                agent_pos = prev_obs['objects'][agent_id]
                next_agent_pos = agent_pos + action_dir
                positions = [None for _ in range(self.n_objs)]
                if not self.env.unwrapped.is_empty(next_agent_pos, agent=True, map=current_map):
                    next_agent_pos = agent_pos
                else:
                    # update agent position on the map
                    current_map[next_agent_pos[0], next_agent_pos[1]] = 4
                    current_map[agent_pos[0], agent_pos[1]] = 0
                positions[agent_id] = next_agent_pos

                # sample bot pos
                bot_id = 0
                for obj_id, obj_pos in enumerate(prev_obs['objects']):
                    if obj_id != agent_id:
                        dirs = self.directions + [[0, 0]]
                        i_dir = np.random.choice(np.arange(len(dirs)), p=self.obj_direction_prior[bot_id])
                        next_pos = obj_pos + dirs[i_dir]
                        if not self.env.unwrapped.is_empty(next_pos, agent=False, map=current_map):
                            next_pos = obj_pos
                        else:
                            current_map[next_pos[0], next_pos[1]] = 8
                            current_map[obj_pos[0], obj_pos[1]] = 0
                        positions[obj_id] = next_pos
                        bot_id += 1
                obs = {'objects': positions.copy(), 'map': current_map.copy(), 'goal': prev_obs['goal']}
                weighted_obs[dict2s(obs)] = weighted_obs.get(dict2s(obs), 0) + prob
            # print(time.time() - init_time)
        else:
            raise NotImplementedError
        return weighted_obs


    def exploit(self, obs):
        # this implements a greedy strategy towards the goal, given assumptions about the agent identity and the input mapping
        # this only works in non-deceptive worlds (no obstacles)
        if self.n_theories == 1:  # if one theory, pick that one
            i_theory = 0
        else:  # if several theories, sample one according to the probabilities
            i_theory = np.random.choice(np.arange(self.n_theories), p=self.probas)

        agent_id = self.theories[i_theory]['agent_id']
        reverse_mapping = self.theories[i_theory]['input_reverse_mapping']

        # compute direction between the agent and the goal
        agent_pos = obs['objects'][agent_id]
        vector_to_goal = obs['goal'] - agent_pos
        assert np.sum(np.abs(vector_to_goal)) != 0, "the agent is already on the goal"
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

if __name__ == '__main__':
    inferself = InferSelf()

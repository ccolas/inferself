import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy
import gym
import gym_gridworld

ARGS = dict(n_objs=4)



class InferSelf:
    def __init__(self, args=ARGS):
        self.n_eobjs = args['n_objs']
        self.directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        # self.obj_direction_prior = np.tile(np.append(np.zeros(len(self.directions)), 1),(self.n_objs,1))
        self.obj_direction_prior = np.full((self.n_objs, len(self.directions)+1), 1/(len(self.directions)+1))
        self.reset_memory()
        self.reset_theories()

    def reset_theories(self):
        self.theory_found = False
        self.theories = []
        dirs = set([l2s(l) for l in self.directions]) # convert directions to string form
        for agent_id in range(self.n_objs):
            for dir0 in sorted(dirs):
                remaining_dirs0 = dirs - set([dir0])
                for dir1 in sorted(remaining_dirs0):
                    remaining_dirs1 = dirs - set([dir0, dir1])
                    for dir2 in sorted(remaining_dirs1):
                        remaining_dirs2 = dirs - set([dir0, dir1, dir2])
                        for dir3 in sorted(remaining_dirs2):
                            dir0l, dir1l, dir2l, dir3l = [s2l(dir) for dir in [dir0, dir1, dir2, dir3]]  # convert string forms into list forms
                            new_theory = dict(agent_id=agent_id,
                                              input_mapping={0: dir0l, 1: dir1l, 2: dir2l, 3: dir3l},
                                              input_reverse_mapping={dir0: 0, dir1: 1, dir2: 2, dir3: 3})
                            self.theories.append(new_theory)
        self.theories = np.array(self.theories)
        self.probas = np.ones(self.n_theories) / self.n_theories
        self.probas[np.arange(0, self.n_theories, self.n_theories // 4)] *= 5
        self.probas /= self.probas.sum()

    @property
    def n_theories(self):
        return len(self.theories)

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
        elif len(to_keep) == len(self.probas):
            if not self.theory_found: print(f'  new datapoint ingested, we didn\'t learn anything here')
        else:
            self.theories = self.theories[to_keep]
            self.probas = self.probas[to_keep]
            self.probas = self.probas / self.probas.sum()  # renormalize the probabilities
            if not self.theory_found:  print(f'  new datapoint ingested, we now have {self.n_theories} theories')

        if self.n_theories == 1:
            if not self.theory_found: print(f'We found the agent with probability 1: it\'s object {self.theories[0]["agent_id"]}, its action mapping is: {self.theories[0]["input_mapping"]}')
            self.theory_found = True
            return self.theories[0], 1
        else:
            theory_id = np.random.choice(np.arange(self.n_theories), p=self.probas)
            return self.theories[theory_id], self.probas[theory_id]

    def compute_likelihood(self, theory, prev_obs, new_obs, action):
        # probability of data given theory
        weighted_obs = self.simulate(prev_obs, theory, action)
        return weighted_obs.get(dict2s(new_obs), 0)

    def get_action(self, obs, mode=None):
        # there are two modes of actions
        # mode 1: the agent tries to infer which object it is and what the action mapping is in an optimal way
        # mode 2: the agent moves towards the goal
        if mode is None:
            if self.n_theories == 1: mode = 2
            else: mode = 1

        if mode == 1:
            good_actions, action = self.explore(obs)
        elif mode == 2:
            good_actions, action = self.exploit(obs)
        else: raise ValueError
        return action

    def explore(self, prev_obs):
        # for each action, compute expected information gain
        action_scores = []
        for action in range(4):
            weighted_obs = {}
            # possible observations given this action, weighted by probability
            for i_theory, theory in enumerate(self.theories):
                theory_prob = self.probas[i_theory]
                # simulate possible observations given this theory and action
                theory_weighted_obs = self.simulate(prev_obs, theory, action)
                for obs, obs_prob in theory_weighted_obs.items():
                    weighted_obs[obs] = weighted_obs.get(obs, 0) + (theory_prob * obs_prob)
            # information gain for each possible observation, weighted by probability
            assert (np.isclose(sum(weighted_obs.values()), 1))
            exp_info_gain = 0
            for obs_str, obs_prob in weighted_obs.items():
                poss_obs = s2dict(obs_str)
                new_probas = self.update(action, prev_obs, poss_obs, True)
                info_gain = self.information_gain(self.probas, new_probas)
                exp_info_gain += info_gain * obs_prob
            action_scores.append(exp_info_gain)
        return action_scores, np.argmax(action_scores)


    def information_gain(self, p0, p1):
        # errors in js distance with very small numbers
        p0 = [round(x,5) for x in p0]
        p1 = [round(x,5) for x in p1]
        return scipy.spatial.distance.jensenshannon(p0, p1)

    # given these actions and these objects, return directions of movemement
    # thinking we might want a function like this (to run in update as well?) bc movememnts may be non-independent
    # ie if two objects run into each other
    # and for cases where we want our theory to include information about how all objects move
    def get_next_positions(self, obs, movements):
        temp_map = []
        positions = []
        for i, dir in enumerate(movements):
            prev_pos = obs['objects'][i]
            if self.is_collision(prev_pos, dir, obs['map']):
                positions.append(prev_pos)
            else:
                positions.append(prev_pos + dir)
        # should check if any objects occupy same space, and if so randomly choose n-1 to move back
        """
        for i, p1 in enumerate(positions):
            for j, p2 in enumerate(positions):
                if i != j:
                    if p1==p2:
                        loser = scipy.stats.bernoulli(0.5)
                        positions[]
        """
        return positions


    # Given this theory and this action, what is the probability distribution over possible observations?
    def simulate(self, prev_obs, theory, action, sample=False):
        agent_id = theory['agent_id']
        action_dir = theory['input_mapping'][action]

        # not specified by theory: movements of other objects
        poss_movements = []
        for i, obj in enumerate(prev_obs['objects']):
            if i==agent_id:
                obj_poss_movements = [(action_dir, 1)]
            else:
                obj_poss_movements = []
                for dir_idx, p in enumerate(self.obj_direction_prior[i]):
                    if dir_idx >= len(self.directions):
                        dir = [0,0]
                    else:
                        dir = self.directions[dir_idx]
                    if p > 0:
                        obj_poss_movements.append((dir, p))
            poss_movements.append(obj_poss_movements)   
        poss_movements_and_probs = itertools.product(*poss_movements)
        
        weighted_obs = {}
        for movements_and_probs in poss_movements_and_probs:
            movements = [x[0] for x in movements_and_probs]
            prob = np.prod([x[1] for x in movements_and_probs])
            assert(prob > 0)
            positions = self.get_next_positions(prev_obs, movements)
            obs = {'objects': positions, 'map': prev_obs['map'], 'goal': prev_obs['goal']}
            weighted_obs[dict2s(obs)] = weighted_obs.get(dict2s(obs), 0) + prob
        # hypothetical obs weighted by probability
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

    def is_collision(self, obj_pos, action_dir, map):
        desired_pos = obj_pos + action_dir
        return map[desired_pos[0], desired_pos[1]] not in [0, 3]

def dict2s(d):
    d2 = {}
    d2['goal'] = list(d['goal'])
    d2['map'] = [list(row) for row in d['objects']]
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

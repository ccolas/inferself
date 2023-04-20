import numpy as np
import matplotlib.pyplot as plt

import gym
import gym_gridworld

ARGS = dict(n_objs=4)



class InferSelf:
    def __init__(self, args=ARGS):
        self.n_objs = args['n_objs']
        self.directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
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

    @property
    def n_theories(self):
        return len(self.theories)

    def reset_memory(self):
        self.actions = []  # memory of past actions
        self.obj_pos = dict(zip(range(self.n_objs), [[] for _ in range(self.n_objs)]))  # memory of past movements for each object

    def update(self, action, prev_obs, new_obs):
        # TODO need to adapt new_obs here
        # we need the position of each object
        # we need the map

        # store new info
        self.actions.append(action)
        for o_id in range(self.n_objs):
            self.obj_pos[o_id].append(new_obs['objects'][o_id])

        # run inference for the new datapoint
        to_keep = []  # we keep track of theories with proba > 0
        for i_theory, theory in enumerate(self.theories):
            obj_new_pos = new_obs['objects'][theory['agent_id']]
            obj_prev_pos = prev_obs['objects'][theory['agent_id']]
            action_dir = np.array(theory['input_mapping'][action])
            # in case of collision, the theory predicts no mvt
            if not self.is_collision(obj_prev_pos, action_dir, prev_obs['map']):
                predicted_pos = obj_prev_pos + action_dir
            else:
                predicted_pos = obj_prev_pos
            # likelihood is binary, either we observe what we expect (l=1) or we don't (l=0)
            likelihood = np.all(predicted_pos == obj_new_pos)
            posterior = self.probas[i_theory] * likelihood
            self.probas[i_theory] = posterior
            if posterior > 0:
                to_keep.append(i_theory)
        to_keep = np.array(to_keep)

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

    def explore(self, obs):
        # for each theory, we assume the theory and compare each of the four actions, their possible outcomes and compute the expected information gain resulting from taking
        # these actions
        action_scores = []
        for action in range(4):
            info_gains = []
            for i_theory, theory in enumerate(self.theories):
                agent_id = theory['agent_id']
                reverse_mapping = theory['action_reverse_mapping']
                mapping = theory['input_mapping']
                agent_pos = obs['objects'][agent_id]
                action_dir = theory['input_mapping'][action]
                # several things can happen here
                # either the theory is true, and
                # if self.is_collision(agent_pos, action_dir, obs['map']):


    def exploit(self, obs):
        # this implements a greedy strategy towards the goal, given assumptions about the agent identity and the input mapping
        # this only works in non-deceptive worlds (no obstacles)
        if self.n_theories == 1:  # if one theory, pick that one
            i_theory = 0
        else:  # if several theories, sample one according to the probabilities
            i_theory = np.random.choice(np.arange(self.n_theories), p=self.probas)

        agent_id = self.theories[i_theory]['agent_id']
        reverse_mapping = self.theories[i_theory]['input_reveerse_mapping']

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


def l2s(l):
    return f'{l[0]}_{l[1]}'

def s2l(s):
    return [int(ss) for ss in s.split('_')]

if __name__ == '__main__':
    inferself = InferSelf()

import os
import pickle
import numpy as np 
import pandas as pd
import json

def load_heuristic_data():
    agent_names = {'heuristic':'Proximity heuristic'}
    env_names = {"logic-v0_False": "Logic", "contingency-v0_False": "Contingency", 'contingency-shuffle-v0_False':'Switching Mappings', 'changeAgent-7-v0_False':'Switching Embodiments (7)', 'contingency_noisy-v0_False':'Noisy Contingency', 'contingency_noisy-v0_False2':'Noisy Contingency 2', 'contingency_less_chars-v0_False':'Contingency\n(2 characters)', 'contingency_more_chars-v0_False':'Contingency\n(6 characters)', 'contingency_8_chars-v0_False':'Contingency\n(8 characters)', 'changeAgent-10-v0_False':'Switching Embodiments (10)', "logic_u-v0_False": "Logic\ngoal uncertainty", "contingency_u-v0_False": "Contingency\ngoal uncertainty",'contingency_u-shuffle-v0_False':'Switching Mappings\ngoal uncertainty', 'changeAgent_u-7-v0_False':'Switching Embodiments (7)\ngoal uncertainty', 'changeAgent_u-10-v0_False':'Switching Embodiments (10)\ngoal uncertainty'}
    data = []
    seen_pairs = []
    # iterate thru exps in this file
    with open('heur_exp.pkl', 'rb') as f:
        exp_group = pickle.load(f)
    for exp_data in exp_group.values():
        # iterate thru envs for this exp
        for env in exp_data.keys():
            for (agent, agent_data) in exp_data[env].items():
                if (env, agent) in seen_pairs:
                    continue
                seen_pairs.append((env,agent))
                for seed in range(20):
                    seed = str(seed)
                    for level in range(40):
                        level = str(level)
                        #get steps until successs
                        level_data = agent_data[seed][level]
                        ind_success = np.argwhere(np.array(level_data['success'])).flatten()
                        if len(ind_success)==0:
                            steps = len(level_data['success'])
                        else:
                            steps = ind_success[0]
                        sf_steps = None
                        data.append({'env':env_names[env], 'agent':agent_names[agent], 'sf_steps':sf_steps, 'steps':steps, 'seed': seed, 'level':int(level)})
    df = pd.DataFrame.from_dict(data)
    df.to_csv('heuristic.csv')


def load_model_data():
    def arg_tup_to_agent_name(arg_tup):
        if arg_tup==(1,0):
            return 'Meta-ePOMDP'
        else:
            return 'Resource-limited meta-ePOMDP\n' + str(arg_tup)
    data = []
    path = 'model_exp.pkl'
    with open(path, 'rb') as f:
        exp_data = pickle.load(f)
    for env in exp_data.keys():
        for arg_tup in exp_data[env].keys():
            agent_data = exp_data[env][arg_tup]
            agent = arg_tup_to_agent_name(arg_tup)
            for seed in range(20):
                seed = str(seed)
                for level in range(40):
                    #get solving time
                    level_data = agent_data[seed][str(level)]
                    ind_success = np.argwhere(np.array(level_data['success'])).flatten()
                    if len(ind_success)==0:
                        steps = len(level_data['success'])
                    else:
                        steps = ind_success[0]
                    # get centering time
                    if env in ["Logic", "Contingency", "Switching Mappings"]:
                        true_self = level_data['true_self'][0]
                        # num steps = number of actions before they know. if it moves on the first action, that's 0
                        for t, char_probas in enumerate(level_data['all_self_probas']): # at 0, they're the same. at 1, maybe we know. in that case, return 0
                            if char_probas[true_self] == max(char_probas):
                                sorted_probas = sorted(char_probas, reverse=True)
                                if sorted_probas[0] >= 1.5*sorted_probas[1]:
                                    sf_steps = t-1
                                    break
                        if t==len(level_data['all_self_probas'])-1:
                            sf_steps = None
                    else:
                        sf_steps = None
                    data.append({'env':env, 'agent':agent, 'sf_steps':sf_steps, 'steps':steps, 'seed': seed, 'level':level})
    df = pd.DataFrame.from_dict(data)
    df.to_csv('model.csv')
    
load_model_data()
import numpy as np
from copy import deepcopy
import pickle
import os
import gym
from inferself import InferSelf
from heuristic import Heuristic
import sys
import random

n_seeds = 20
n_levels = 40

def get_args(env, agent):
    args = dict(max_steps=2,
                # what to infer
                infer_mapping=False,
                infer_switch=False,
                # priors
                biased_action_mapping=False,
                biased_action_mapping_factor=10,
                bias_bot_mvt='uniform',  # static or uniform
                p_switch=0.01,
                # learning strategies and biases
                likelihood_weight=1,
                explicit_resetting=False,
                noise_prior=0.01,
                avatar_noise=0.01,
                # exploration
                explore_only=False,  # if true, the agent only explores and the goal is removed from the env
                explore_randomly=False,
                simulation='sampling',  # exhaustive or sampling
                n_simulations=10,  # number of simulations if sampling
                uniform_attention_bias=False,
                attention_bias=False,
                mapping_forgetting_factor=None,
                forget_action_mapping=False,
                n_objs_attended_to=4,
                peripheral_attention_prob = 0.05,
                heuristic_noise = 0.1,
                is_heuristic=False,
                check_oob = False,
                conf_threshold=1.5,
                verbose=False,
                )
    if 'heuristic' in agent:
        args['is_heuristic'] = True
    if 'shuffle' not in env:
        args['infer_mapping'] = False
    else:
        args['infer_mapping'] = True
    if 'no_switch' in agent:
        args['infer_switch'] = False
    else:
        args['infer_switch'] = True

    if 'random_explo' in agent:
        args['explore_randomly'] = True

    if 'rand_attention_bias' in agent:
        args['uniform_attention_bias'] = True

    if 'attention_bias' in agent:
        args['attention_bias'] = True
        params = agent.split('_')[-1]
        args['n_objs_attended_to'] = int(params.split(',')[0])
        args['peripheral_attention_prob'] = int(params.split(',')[1])/100
        
    if 'forget_action_mapping' in agent:
        args['forget_action_mapping'] = True
        args['biased_action_mapping'] = True
    else:
        args['forget_action_mapping'] = False
        args['biased_action_mapping'] = False

    if 'prior_action_mapping' in agent:
        args['biased_action_mapping'] = True
    else:
        args['biased_action_mapping'] = False
    return args


def get_prob_of_true(s, true_agent, true_mapping):
    for i_theory, theory in enumerate(s.theories):
        found_it = False
        if theory['agent_id'] == true_agent:
            found_it = True
            for i, d in enumerate(true_mapping):
                if np.any(theory['input_mapping'][i] != d):
                    found_it = False
        if found_it:
            return s.current_posterior_over_theories[i_theory], s.get_noise_mean(s.theories[i_theory])


def run_agent_in_env(env_name, agent, threshold, keys, time_limit, ff):
    # run exp for this env/arg set
    args = get_args(env_name, agent)
    args['conf_threshold'] = threshold
    args['mapping_forgetting_factor'] = ff
    if 'changeAgent' in env_name:
        args['p_switch'] = 1/7
    else:
        args['p_switch'] = 0.01

    if 'noisy' in env_name:
        args['avatar_noise'] = 1/3
    else:
        args['avatar_noise'] = 0.01

    if args['is_heuristic'] and 'contingency' in env_name:
        args['check_oob'] = True
    
    env = gym.make(env_name)
    data = dict(zip(keys, [[] for _ in range(len(keys))]))
    prev_obs, prev_info, prev_obs_state = env.reset()
    args.update(n_objs=env.n_candidates)
    if args['is_heuristic']:
        inferself = Heuristic(obs=prev_obs_state,
                               args=args)
    else:
        inferself = InferSelf(obs=prev_obs_state,
                          args=args)
    previous_agent = None
    
    for t in range(time_limit):        
        mode=None
        change = False
        info = {}
        action=None
        action_mode = None
        obs_state = prev_obs_state
        done=False
        if t>0:
            action, action_mode = inferself.get_action(prev_obs_state, enforce_mode=mode)
            obs, rew, done, info, obs_state = env.step(action)
            inferself.update_theory(prev_obs_state, obs_state, action)
            # did the agent change?
            if previous_agent != env.unwrapped.agent_id and t > 0:
                change = True
            else:
                change = False
            previous_agent = env.unwrapped.agent_id

        if args['is_heuristic']:
            new_data = dict(tpt=t,
                        agent_change=change,
                        success=info.get("success", False),
                        obj_pos=info.get('semantic_state', {}).get("objects", 0),
                        map=info.get('semantic_state', {}).get("map", np.array([])).flatten(),
                        action=action,
                        true_self=env.unwrapped.agent_id,
                        all_self_probas=None,
                        mode=action_mode,
                        p_switch=None,
                        true_mapping=env.unwrapped.action_pos_dict,
                        all_mapping_probas=None,
                        true_theory_probas=None,
                        agent_found=None,
                        true_theory_noise_mean=None,
                        top_theory=None,
                        top_theory_proba=None)
        else:
            theory, theory_id, proba = inferself.get_best_theory(get_proba=True)
            # obs contains object positions, goal pos,
            # true agent id, predicted agent, prob of true agent, prob of true mapping, prob of top theory
            true_theory_prob, noise_mean = get_prob_of_true(inferself, env.unwrapped.agent_id, env.unwrapped.action_pos_dict)
        
            new_data = dict(tpt=t,
                        agent_change=change,
                        success=info.get("success", False),
                        obj_pos=info.get('semantic_state', {}).get("objects", 0),
                        map=info.get('semantic_state', {}).get("map", np.array([])).flatten(),
                        action=action,
                        mode=action_mode,
                        true_self=env.unwrapped.agent_id,
                        all_self_probas=inferself.history_posteriors_over_agents[-1],
                        p_switch=inferself.history_posteriors_p_switch[-1],
                        true_mapping=env.unwrapped.action_pos_dict,
                        all_mapping_probas=inferself.get_mapping_probas(),
                        true_theory_probas=true_theory_prob,
                        agent_found=true_theory_prob > 0.5,
                        true_theory_noise_mean=noise_mean,
                        top_theory=theory,
                        top_theory_proba=proba)  # which theory is correct? get prob of that theory
        
        for k in new_data.keys():
            data[k].append(new_data[k])
        prev_obs_state = deepcopy(obs_state)
        if done:
            break
    return data


def run_experiment(expe_name, envs, agents, threshold, ff, save_dir="output/", overwrite=False,
                   time_limit=10000):
    data_path = save_dir + expe_name + '.pkl'
    print(f'Running experiment {expe_name}, saving to {data_path}')

    keys = ['tpt', 'success', 'obj_pos', 'map', 'action', 'true_self', 'all_self_probas', 'true_mapping', 'all_mapping_probas', 'agent_found',
            'true_theory_probas', 'true_theory_noise_mean', 'top_theory', 'top_theory_proba', 'agent_change', 'p_switch', 'mode']
    
    # dict with all data
    data = dict()

    # load previous results
    if not overwrite:
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                
    if expe_name not in data.keys():
        data[expe_name] = dict()
        
    # loop over environments, agents and seeds
    for i_env, env_name in enumerate(envs):
        env_name_dict = env_name
        if env_name_dict not in data[expe_name].keys(): data[expe_name][env_name_dict] = dict()
        print(f"  Env: {env_name_dict} ({i_env + 1} / {len(envs)})")
        for i_agent, agent in enumerate(agents):
            print(f"    Agent: {agent} ({i_agent + 1} / {len(agents)})")
            if agent not in data[expe_name][env_name_dict].keys(): data[expe_name][env_name_dict][agent] = dict()
            
            data[expe_name][env_name_dict][agent]["args"] = get_args(env_name, agent)
            for s in range(n_seeds):
                print(f'      Seed {s + 1} / {n_seeds}')
                np.random.seed(s)
                random.seed(s)
                if str(s) not in data[expe_name][env_name_dict][agent].keys(): data[expe_name][env_name_dict][agent][str(s)] = dict()
                for l in range(n_levels):
                    print(f'     Level {l + 1} / {n_levels}')
                    data[expe_name][env_name_dict][agent][str(s)][str(l)] = run_agent_in_env(env_name, agent, threshold, keys, time_limit, ff)
                    level_data = data[expe_name][env_name_dict][agent][str(s)][str(l)]
                    print(np.argwhere(np.array(level_data['success'])).flatten())
                    with open(data_path, 'wb') as f:
                        pickle.dump(data, f)


if __name__ == '__main__':
    expe_name = "exp"
    envs = ['logic-v0', 'logic_u-v0', 'contingency-v0', 'contingency_u-v0', 'contingency_noisy-v0', 'contingency-shuffle-v0', 'contingency_u-shuffle-v0', 'contingency_less_chars-v0', 'contingency_more_chars-v0', 'contingency_8_chars-v0', 'changeAgent-7-v0', 'changeAgent_u-7-v0', 'changeAgent-10-v0']
    agents = ['base', 'forget_action_mapping_rand_attention_bias_1,5', 'heuristic']
    run_experiment(expe_name=expe_name, envs=envs, agents=agents, threshold=1.5, ff=0.4)


from copy import deepcopy
import pickle
import os
import gym
import gym_gridworld
from inferself import InferSelf
from foil_inferself import InferSelfFoil
# from inferself_noiseless import InferSelfNoiseless
import csv
import numpy as np

n_runs = 100

expe_name = 'comparison_exp'
envs = ['logic-v0', 'contingency-v0', 'contingency-shuffle-v0', 'changeAgent-7-v0']
for nm in ['logic', 'contingency', 'contingency-shuffle', 'changeAgent-7']:
    envs.append(nm + '-5-easy')
    envs.append(nm + '-5-hard')
    envs.append(nm + '-8-easy')
    envs.append(nm + '-8-hard')
    envs.append(nm + '-12-easy')
    envs.append(nm + '-12-hard')
agents = ['base', 'foil']
#agents = ['base', 'rand_attention_bias_1', 'forget_action_mapping_rand_attention_bias_1']
explore_exploit = [False]

def get_args(env, agent, explore_only=False):
    args = dict(n_objs=4,
                max_steps=2,
                # what to infer
                infer_mapping=False,
                infer_switch=False,
                # priors
                biased_action_mapping=False,
                biased_action_mapping_factor=100,
                bias_bot_mvt='uniform',  # static or uniform
                p_switch=0.01,
                # learning strategies and biases
                likelihood_weight=1,
                explicit_resetting=False,
                # noise_prior_beta=[1, 15],
                noise_prior=0.01,
                # exploration
                explore_only=False,  # if true, the agent only explores and the goal is removed from the env
                explore_randomly=False,
                simulation='sampling',  # exhaustive or sampling
                n_simulations=10,  # number of simulations if sampling
                uniform_attention_bias=False,
                attention_bias=False,
                mapping_forgetting_factor=0.25,
                forget_action_mapping=False,
                n_objs_attended_to=4,
                is_foil=False,
                # explore-exploit
                explore_exploit_threshold=0.5,  # confidence threshold for agent id
                verbose=False,
                )
    if 'foil' in agent:
        args['is_foil'] = True
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
        args['n_objs_attended_to'] = int(agent.split('_')[-1])

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


def run_agent_in_env(env_name, agent, explore_only, keys, time_limit):
    # run exp for this env/arg set
    args = get_args(env_name, agent, explore_only)
    if 'changeAgent' in env_name:
        args['p_switch'] = 1/7
    else:
        args['p_switch'] = 0.01

    env = gym.make(env_name)
    data = dict(zip(keys, [[] for _ in range(len(keys))]))
    if args['explore_only']:
        env.unwrapped.no_goal = True
    prev_obs, prev_info = env.reset()
    args.update(n_objs=env.n_candidates)
    if args['is_foil']:
         inferself = InferSelfFoil(env=env,
                               args=args)
    else:
        inferself = InferSelf(env=env,
                          args=args)
    previous_agent = None
    for t in range(time_limit):
        # print(t)
        mode=None
        if 'oneswitch' in env_name:
            if t < 30:
                mode = 1
        action, action_mode = inferself.get_action(prev_info['semantic_state'], enforce_mode=mode)
        obs, rew, done, info = env.step(action)
        inferself.update_theory(prev_info['semantic_state'], info['semantic_state'], action)

        # did the agent change?
        if previous_agent != env.unwrapped.agent_id and t > 0:
            change = True
        else:
            change = False
        previous_agent = env.unwrapped.agent_id


        if args['is_foil']:
            new_data = dict(tpt=t,
                        agent_change=change,
                        success=info["success"],
                        obj_pos=info['semantic_state']["objects"],
                        map=info['semantic_state']["map"].flatten(),
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
            theory, proba = inferself.get_best_theory(get_proba=True)
            # obs contains object positions, goal pos,
            # true agent id, predicted agent, prob of true agent, prob of true mapping, prob of top theory
            true_theory_prob, noise_mean = get_prob_of_true(inferself, env.unwrapped.agent_id, env.unwrapped.action_pos_dict)
        
            new_data = dict(tpt=t,
                        agent_change=change,
                        success=info["success"],
                        obj_pos=info['semantic_state']["objects"],
                        map=info['semantic_state']["map"].flatten(),
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
        prev_info = deepcopy(info)
        if done:
            break
    return data


def run_experiment(exp_name, envs, agents, explore_exploit, save_dir="output/", overwrite=False,
                   time_limit=200):
    data_path = save_dir + exp_name + '.pkl'
    print(f'Running experiment {exp_name}, saving to {data_path}')

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
    for explore_only in explore_exploit:
        for i_env, env_name in enumerate(envs):
            env_name_dict = env_name + '_' + str(explore_only)
            if env_name_dict not in data[expe_name].keys(): data[expe_name][env_name_dict] = dict()
            print(f"  Env: {env_name_dict} ({i_env + 1} / {len(envs)})")
            for i_agent, agent in enumerate(agents):
                print(f"    Agent: {agent} ({i_agent + 1} / {len(agents)})")
                if agent not in data[expe_name][env_name_dict].keys(): data[expe_name][env_name_dict][agent] = dict()
                
                data[expe_name][env_name_dict][agent]["args"] = get_args(env_name, agent, explore_only)
                for i in range(n_runs):
                    print(f'      Seed {i + 1} / {n_runs}')
                    #if str(i) not in data[expe_name][env_name_dict][agent].keys():
                    data[expe_name][env_name_dict][agent][str(i)] = run_agent_in_env(env_name, agent, explore_only, keys, time_limit)
                    with open(data_path, 'wb') as f:
                        pickle.dump(data, f)


if __name__ == '__main__':
    run_experiment(exp_name=expe_name, envs=envs, agents=agents, explore_exploit=explore_exploit, overwrite=False)


from copy import deepcopy
import pickle
import os
import gym
import gym_gridworld
from inferself import InferSelf
import csv
import numpy as np

# TODO: also track prob of correct action mapping?
#      slightly confusing bc we don't try to get this exactly right
# change explore so that we also figure out the action mapping?
# random exploration as one version


n_runs = 50

temp_noise = np.array([1, 1, 1., 1, 1])
discrete_noise_values = np.array([0, 0.05, 0.1, 0.15, 0.2])
proba_discrete_noise = temp_noise / sum(temp_noise)

expe_name = 'switch_frequency'

if expe_name == 'with_agent_change':
    explore_exploit = [True]
    env_list = ['changeAgent-noisy-v0', 'changeAgent-shuffle-noisy-10-v0']
    variants = ['base', 'explicit_resetter', 'current_focused_forgetter', 'hierarchical']
# elif expe_name == 'without_agent_change':
#     env_list = ['logic-v0', 'logic-noisy-v0', 'logic-shuffle-v0', 'logic-shuffle-noisy-v0',
#                 'contingency-v0', 'contingency-noisy-v0', 'contingency-shuffle-v0', 'contingency-shuffle-noisy-v0',]
#     variants = ['base']#, 'no_infer_mapping', 'biased_action_mapping', 'random_explo']
# elif expe_name == 'one_switch':
#     assert False
#     print("don't forget to cancel the goal!")
#     env_list = ['changeAgent-shuffle-noisy-oneswitch-v0']
#     variants = ['base', 'explicit_resetter', 'current_focused_forgetter', 'hierarchical']
elif expe_name == 'no_infer_mapping':
    explore_exploit = [True]
    env_list = ['logic-noisy-v0', 'contingency-noisy-v0']
    variants = ['no_infer_mapping']
elif expe_name == 'base':
    explore_exploit = [True]
    env_list = ['logic-shuffle-noisy-v0', 'contingency-shuffle-noisy-v0']
    variants = ['base']
elif expe_name == 'switch_frequency':
    # expe switch frequency
    explore_exploit = [True]
    env_list = ['changeAgent-noisy-7-v0', 'changeAgent-noisy-10-v0', 'changeAgent-noisy-15-v0',
                'changeAgent-shuffle-noisy-7-v0', 'changeAgent-shuffle-noisy-10-v0', 'changeAgent-shuffle-noisy-15-v0']
    variants = ['base', 'explicit_resetter', 'current_focused_forgetter', 'hierarchical']
elif expe_name == 'switch_frequency_false':
    # expe switch frequency
    explore_exploit = [False]
    env_list = ['changeAgent-7-v0', 'changeAgent-10-v0', 'changeAgent-15-v0',
                'changeAgent-noisy-7-v0', 'changeAgent-noisy-10-v0', 'changeAgent-noisy-15-v0',
                'changeAgent-shuffle-noisy-7-v0', 'changeAgent-shuffle-noisy-10-v0', 'changeAgent-shuffle-noisy-15-v0']
    variants = ['base', 'explicit_resetter', 'current_focused_forgetter', 'hierarchical']
elif expe_name == 'switch_frequency_no_noise_false':
    # expe switch frequency
    explore_exploit = [False]
    env_list = ['changeAgent-7-v0', 'changeAgent-10-v0', 'changeAgent-15-v0']
    variants = ['base_no_noise', 'explicit_resetter_no_noise', 'hierarchical_no_noise']
else:
    raise NotImplementedError



def get_args(variant, explore_only=False):
    args = dict(n_objs=4,
                biased_input_mapping=False,
                bias_bot_mvt='uniform',  # static or uniform
                simulation='sampling',  # exhaustive or sampling
                n_simulations=10,  # number of simulations if sampling
                infer_mapping=True,
                threshold=0.6,  # confidence threshold for agent id
                noise_prior_beta=[1, 15],
                noise_prior_discrete=proba_discrete_noise,
                noise_values_discrete=discrete_noise_values,
                forget_param=None,  # the smaller this is, the more forgetful we are when computing noise
                likelihood_weight=1,
                explicit_resetting=False,
                print_status=False,
                hierarchical=False,
                p_change=0.1,
                explore_only=explore_only,  # if true, the agent only explores and the goal is removed from the env
                explore_randomly=False
                )
    if variant == 'base':
        pass
    elif variant == 'no_infer_mapping':
        args['infer_mapping'] = False
    elif variant == 'explicit_resetter':
        args['explicit_resetting'] = True
    elif variant == 'current_focused_forgetter':
        args['likelihood_weight'] = 2
        args['forget_param'] = 5
    elif variant == 'hierarchical':
        args['hierarchical'] = True
    elif variant == 'random_explo':
        args['explore_randomly'] = True
    elif variant == 'biased_action_mapping':
        args['biased_input_mapping'] = True
    else:
        raise NotImplementedError

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
            return s.probas[i_theory], s.get_noise_mean(s.theories[i_theory])


def run_agent_in_env(env_name, agent, explore_only, keys, time_limit):
    # run exp for this env/arg set
    args = get_args(agent, explore_only)
    env = gym.make(env_name)
    data = dict(zip(keys, [[] for _ in range(len(keys))]))
    if args['explore_only']:
        env.unwrapped.no_goal = True
    prev_obs, prev_info = env.reset()
    args.update(n_objs=env.n_candidates)
    inferself = InferSelf(env=env, args=args)
    previous_agent = None
    for t in range(time_limit):
        # print(t)
        mode=None
        if 'oneswitch' in env_name:
            if t < 30:
                mode = 1
        action = inferself.get_action(prev_info['semantic_state'], enforce_mode=mode)
        obs, rew, done, info = env.step(action)
        theory, proba = inferself.update_theory(prev_info['semantic_state'], info['semantic_state'], action)

        # did the agent change?
        if previous_agent != env.unwrapped.agent_id and t > 0:
            change = True
        else:
            change = False
        previous_agent = env.unwrapped.agent_id

        # obs contains object positions, goal pos,
        # true agent id, predicted agent, prob of true agent, prob of true mapping, prob of top theory
        true_theory_prob, noise_mean = get_prob_of_true(inferself, env.unwrapped.agent_id, env.unwrapped.action_pos_dict)
        new_data = dict(tpt=t,
                        agent_change=change,
                        success=info["success"],
                        obj_pos=info['semantic_state']["objects"],
                        map=info['semantic_state']["map"].flatten(),
                        action=action,
                        true_self=env.unwrapped.agent_id,
                        all_self_probas=inferself.history_agent_probas[-1],
                        true_mapping=env.unwrapped.action_pos_dict,
                        all_mapping_probas=inferself.get_mapping_probas(),
                        true_theory_probas=true_theory_prob,
                        agent_found=true_theory_prob > 0.7,
                        true_theory_noise_mean=noise_mean,
                        top_theory=theory,
                        top_theory_proba=proba)  # which theory is correct? get prob of that theory
        for k in new_data.keys():
            data[k].append(new_data[k])
        prev_info = deepcopy(info)
        if done:
            break
    return data


def run_experiment(exp_name, envs, agents, explore_exploit, save_dir="/mnt/e85692fd-9cbc-4a8d-b5c5-9252bd9a34fd/Research/Scratch/inferself/data/experiments/", overwrite=False,
                   time_limit=60):
    data_path = save_dir + exp_name + '.pkl'
    print(f'Running experiment {exp_name}, saving to {data_path}')

    keys = ['tpt', 'success', 'obj_pos', 'map', 'action', 'true_self', 'all_self_probas', 'true_mapping', 'all_mapping_probas', 'agent_found',
            'true_theory_probas', 'true_theory_noise_mean', 'top_theory', 'top_theory_proba', 'agent_change']
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
                for i in range(n_runs):
                    print(f'      Seed {i + 1} / {n_runs}')
                    if str(i) not in data[expe_name][env_name_dict][agent].keys():
                        data[expe_name][env_name_dict][agent][str(i)] = run_agent_in_env(env_name, agent, explore_only, keys, time_limit)
                        with open(data_path, 'wb') as f:
                            pickle.dump(data, f)


if __name__ == '__main__':
    run_experiment(exp_name=expe_name, envs=env_list, agents=variants, explore_exploit=explore_exploit)


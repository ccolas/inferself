from copy import deepcopy

import gym
import gym_gridworld
from inferself import InferSelf
import csv
import numpy as np



#TODO: also track prob of correct action mapping?
#      slightly confusing bc we don't try to get this exactly right
#change explore so that we also figure out the action mapping?
#random exploration as one version


n_runs = 10
env_list = ['logic-v0', 'contingency-v0', 'changeAgent-v0', 'logic-shuffle-v0', 'contingency-shuffle-v0', 'changeAgent-shuffle-v0', 'logic-noisy-v0', 'contingency-noisy-v0', 'changeAgent-noisy-v0', 'logic-shuffle-noisy-v0', 'changeAgent-shuffle-noisy-v0', 'contingency-shuffle-noisy-v0']

args1 = dict(n_objs=4,
            biased_input_mapping=False,
            bias_bot_mvt='uniform', # static or uniform
            simulation='sampling',  # exhaustive or sampling
            n_simulations=50,  # number of simulations if sampling
            infer_mapping=True,
            threshold=0.9, # confidence threshold for agent id
            noise_prior_beta=[1, 15],
            noise_prior_discrete=np.full((21,), 1/21),
            noise_values_discrete= np.arange(21)/20,
            forget_param=None, #the smaller this is, the more forgetful we are when computing noise
            likelihood_weight=1,
            explicit_resetting=False,
            print_status=False,
            hierarchical=False,
            p_change=0.1
            )
args2 = dict(n_objs=4,
            biased_input_mapping=False,
            bias_bot_mvt='uniform', # static or uniform
            simulation='sampling',  # exhaustive or sampling
            n_simulations=50,  # number of simulations if sampling
            infer_mapping=True,
            threshold=0.9, # confidence threshold for agent id
            noise_prior_beta=[1, 15],
            noise_prior_discrete=np.full((21,), 1/21),
            noise_values_discrete= np.arange(21)/20,
            forget_param=None, #the smaller this is, the more forgetful we are when computing noise
            likelihood_weight=1,
            explicit_resetting=True,
            print_status=False,
            hierarchical=False,
            p_change=0.1
            )

args3 = dict(n_objs=4,
            biased_input_mapping=False,
            bias_bot_mvt='uniform', # static or uniform
            simulation='sampling',  # exhaustive or sampling
            n_simulations=50,  # number of simulations if sampling
            infer_mapping=True,
            threshold=0.9, # confidence threshold for agent id
            noise_prior_beta=[1, 15],
            noise_prior_discrete=np.full((21,), 1/21),
            noise_values_discrete= np.arange(21)/20,
            forget_param=None, #the smaller this is, the more forgetful we are when computing noise
            likelihood_weight=2,
            explicit_resetting=False,
            print_status=False,
            hierarchical=False,
            p_change=0.1
            )

args4 = dict(n_objs=4,
            biased_input_mapping=False,
            bias_bot_mvt='uniform', # static or uniform
            simulation='sampling',  # exhaustive or sampling
            n_simulations=50,  # number of simulations if sampling
            infer_mapping=True,
            threshold=0.9, # confidence threshold for agent id
            noise_prior_beta=[1, 15],
            noise_prior_discrete=np.full((21,), 1/21),
            noise_values_discrete= np.arange(21)/20,
            forget_param=5, #the smaller this is, the more forgetful we are when computing noise
            likelihood_weight=2,
            explicit_resetting=False,
            print_status=False,
            hierarchical=False,
            p_change=0.1
            )


agent_dict= dict(base=args1, explicit_resetter=args2, current_focused=args3, current_focused_forgetter=args4)



def get_prob_of_true(s, true_agent,true_mapping):
    for i_theory, theory in enumerate(s.theories):
        found_it = False
        if theory['agent_id'] == true_agent:
            found_it = True
            for i, d in enumerate(true_mapping):
                if np.any(theory['input_mapping'][i] != d):
                    found_it = False
        if found_it:
            return s.probas[i_theory]


with open('output/out2.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['env', 'agent_type', 'tpt', 'run', 'success', 'obj_pos', 'map', 'action', 'true_self', 'all_self_probas', 'true_mapping', 'all_mapping_probas', 'true_theory_probas', 'top_theory', 'top_theory_proba'])
            writer.writeheader()

for env_name in env_list:
    print(env_name)
    for agent, args in agent_dict.items():
        print(agent)
        d_list = []
        for i in range(n_runs):
            print(i)
            #run exp for this env/arg set
            env = gym.make(env_name)
            prev_obs, prev_info = env.reset()
            #env.render(None)
            args.update(n_objs=env.n_candidates)
            inferself = InferSelf(env=env,args=args)
            t = 0
            while True:
                action = inferself.get_action(env.semantic_state)
                #print('Action:', env.unwrapped.get_action_name(action))
                obs, rew, done, info = env.step(action)
                #env.render(None)
                theory, proba = inferself.update_theory(prev_info['semantic_state'], info['semantic_state'], action)
                #obs contains object positions, goal pos, 
                #true agent id, predicted agent, prob of true agent, prob of true mapping, prob of top theory
                true_theory_prob = get_prob_of_true(inferself, env.unwrapped.agent_id, env.unwrapped.action_pos_dict)
                d_list.append(dict(env=env_name, agent_type=agent, tpt=t, run=i,
                     success=info["success"],
                     obj_pos=info['semantic_state']["objects"],
                     map=info['semantic_state']["map"].flatten(),
                     action=action,
                     true_self=env.unwrapped.agent_id,
                     all_self_probas=inferself.history_agent_probas[-1],
                     true_mapping=env.unwrapped.action_pos_dict,
                     all_mapping_probas=inferself.get_mapping_probas(),
                     true_theory_probas = true_theory_prob, 
                     top_theory = theory,
                     top_theory_proba = proba)) #which theory is correct? get prob of that theory
                prev_obs = obs.copy()
                prev_info = deepcopy(info)
                t = t+1
                if done or t>300:
                    break
        with open('output/out.csv', 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=d_list[0].keys())
            writer.writerows(d_list)


#if __name__ == '__main__':
    

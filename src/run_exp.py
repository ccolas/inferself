from copy import deepcopy

import gym
import gym_gridworld
from inferself import InferSelf
import csv

n_runs = 10
env_list = ['logic-v0', 'contingency-v0', 'changeAgent-v0', 'logic-shuffle-v0', 'contingency-shuffle-v0', 'changeAgent-shuffle-v0', 'logic-noisy-v0', 'contingency-noisy-v0', 'changeAgent-noisy-v0', 'logic-shuffle-noisy-v0', 'changeAgent-shuffle-noisy-v0', 'contingency-shuffle-noisy-v0']

args1 = dict(n_objs=4,
            biased_input_mapping=False,
            bias_bot_mvt='uniform', # static or uniform
            simulation='sampling',  # exhaustive or sampling
            n_simulations=50,  # number of simulations if sampling
            infer_mapping=True,
            threshold=0.9, # confidence threshold for agent id
            beta_prior=[1, 15],
            forget_param=None, #the smaller this is, the more forgetful we are when computing noise
            likelihood_weight=1,
            explicit_resetting=False,
            print_status=False
            )
args2 = dict(n_objs=4,
            biased_input_mapping=False,
            bias_bot_mvt='uniform', # static or uniform
            simulation='sampling',  # exhaustive or sampling
            n_simulations=50,  # number of simulations if sampling
            infer_mapping=True,
            threshold=0.9, # confidence threshold for agent id
            beta_prior=[1, 15],
            forget_param=None, #the smaller this is, the more forgetful we are when computing noise
            likelihood_weight=1,
            explicit_resetting=True,
            print_status=False
            )

args3 = dict(n_objs=4,
            biased_input_mapping=False,
            bias_bot_mvt='uniform', # static or uniform
            simulation='sampling',  # exhaustive or sampling
            n_simulations=50,  # number of simulations if sampling
            infer_mapping=True,
            threshold=0.9, # confidence threshold for agent id
            beta_prior=[1, 15],
            forget_param=None, #the smaller this is, the more forgetful we are when computing noise
            likelihood_weight=2,
            explicit_resetting=False,
            print_status=False
            )

args4 = dict(n_objs=4,
            biased_input_mapping=False,
            bias_bot_mvt='uniform', # static or uniform
            simulation='sampling',  # exhaustive or sampling
            n_simulations=50,  # number of simulations if sampling
            infer_mapping=True,
            threshold=0.9, # confidence threshold for agent id
            beta_prior=[1, 15],
            forget_param=5, #the smaller this is, the more forgetful we are when computing noise
            likelihood_weight=2,
            explicit_resetting=False,
            print_status=False
            )


agent_dict= dict(base=args1, explicit_resetter=args2, current_focused=args3, current_focused_forgetter=args4)#, "forgetter": args2, "forgetter2": args3, "naive":args4}

with open('output/out.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['env', 'pred_agent', 'tpt', 'run', 'true_agent', 'success', 'obj_pos', 'map', 'agent_probas', 'action', 'top_theory', 'top_theory_proba', 'noise_beta_mean', 'noise_beta_params'])
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
                #true agent id
                d_list.append(dict(env=env_name, pred_agent=agent, tpt=t, run=i,
                     true_agent=env.unwrapped.agent_id, success=info["success"],
                     obj_pos=info['semantic_state']["objects"], map=info['semantic_state']["map"].flatten(), agent_probas=inferself.history_agent_probas[-1],
                     action=action, top_theory=theory, top_theory_proba = proba,
                     noise_beta_mean=inferself.get_beta_mean(theory, std=True), noise_beta_params=theory["beta_params"]))
                prev_obs = obs.copy()
                prev_info = deepcopy(info)
                t = t+1
                if done:
                    break
        with open('output/out.csv', 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=d_list[0].keys())
            writer.writerows(d_list)


#if __name__ == '__main__':
    

import os

import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
import pandas as pd
import json
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
          '#7f7f7f', '#bcbd22', '#17becf']
sns.color_palette("husl", 8)
save_dir = "../output/"
plot_dir = "plots/"


def plot_single_game(data, path):
    fig, ax = plt.subplots(figsize=(15, 7))
    # plot success
    ind_success = np.argwhere(np.array(data['success'])).flatten()
    if len(ind_success) > 0:
        plt.axvline(ind_success[0], ymin=0, ymax=1, color='red', label='success')
    # ind_inference = np.argwhere(np.array(data['true_theory_probas']) > 0.7).flatten()
    # if len(ind_inference) > 0:
    #     plt.axvline(ind_inference[0], ymin=0, ymax=1, color='blue', label='inference')
    all_self_probas = np.array(data['all_self_probas']).T
    n_agents = len(all_self_probas)
    n_steps = len(all_self_probas[0])
    for i, sp, c in zip(range(n_agents), all_self_probas, COLORS):
        plt.plot(sp, c=c, label=str(i))
    plt.plot(data['p_switch'], color='k', label='p_switch')
    plt.legend()
    plt.xlim([0, 60])
    plt.ylim([0, 1.05])
    plt.scatter(np.arange(n_steps), [1]*n_steps, c=[COLORS[agent_id] for agent_id in data['true_self']])
    plt.savefig(path)
    plt.close('all')

def plot_each_game():
    for expe_name in all_data.keys():
        for env in all_data[expe_name].keys():
            for agent in all_data[expe_name][env].keys():
                plot_path = plot_dir + env + '/' + agent + '/'
                os.makedirs(plot_path, exist_ok=True)
                for seed in all_data[expe_name][env][agent].keys():
                    data = all_data[expe_name][env][agent][seed]
                    filename = f'{env}__{agent}__{seed}.png'
                    plot_single_game(data, path=plot_path + filename)


#rn data organized as env, agent, seed
#we want seed col, agent col, env col

def get_exp_data():
    data = []
    folder_to_env_name = {'logic_game/':"logic", 'contingency_game/':"contingency", 'contingency_game_shuffled_1/':'contingency-shuffle', 'change_agent_game/':'changeAgent-7'}

    path = '../../../data/'
    for folder in ['logic_game/', 'contingency_game/', 'contingency_game_shuffled_1/', 'change_agent_game/']:
        #get last 80 levels of human play
        all_steps = []
        for fname in os.listdir(path + folder + 'human/'):
            if fname.startswith('.'):
                continue
            with open(path + folder + 'human/' + fname) as f:
                subj = json.load(f)
            all_steps = all_steps + subj['data']['steps'][20:]
        env_name = folder_to_env_name[folder]
        for (i,s) in enumerate(all_steps):
            data.append({'env':env_name, 'agent':'human_asymptote','seed':i,'success':s})
        
        #get all levels for self class
        all_steps = []
        for iter_folder in os.listdir(path + folder + 'self_class/'):
            if iter_folder.startswith('.'):
                continue
            for fname in os.listdir(path + folder + 'self_class/' + iter_folder):
                if fname.startswith('.'):
                    continue
                with open(path + folder + 'self_class/' + iter_folder + '/' + fname) as f:
                    iter = json.load(f)
                all_steps = all_steps + iter['data']['steps']
        env_name = folder_to_env_name[folder]
        for (i,s) in enumerate(all_steps):
            data.append({'env':env_name, 'agent':'hardcoded','seed':i,'success':s})
    df = pd.DataFrame.from_dict(data)
    df.to_csv('exp_data.csv')
    return data

def get_success_df(agents, envs):
    data = []
    for env in all_data.keys():
        if env.split('-v0_False')[0] not in envs:
            continue
        for agent in all_data[env].keys():
            if agent not in agents:
                continue
            for seed in all_data[env][agent].keys():
                if seed =="args":
                    continue
                ind_success = np.argwhere(np.array(all_data[env][agent][seed]['success'])).flatten()
                if len(ind_success) > 0:
                    success = ind_success[0]
                else:
                    print("here!")
                    print(env)
                    print(agent)
                    print(seed)
                    success = 150
                data.append({'env':env.split('-v0_False')[0],'agent':agent,'seed':seed,'success':success})
    df = pd.DataFrame.from_dict(data)
    df2 = pd.read_csv('exp_data.csv')
    df = pd.concat([df,df2], ignore_index = True)
    return df

def plot_mean_success(agents, envs):
    df = get_success_df(agents, envs)
    #rename for plotting
    new_agent_name = {'base':'base', 'hardcoded':'hardcoded', 'human_asymptote':'human asymptote', 'rand_attention_bias_1': 'attention limit 1 (uniform)',  'forget_action_mapping_rand_attention_bias_1': 'attention limit 1 (uniform), forget action mapping', 'attention_bias_1': 'attention limit 1 (posterior)',  'forget_action_mapping_attention_bias_1': 'attention limit 1 (posterior), forget action mapping'}
    df['agent'] = df['agent'].transform(lambda x: new_agent_name[x])
    #new_env_name = {"logic": "logic\n(only self moves)", "contingency": "contingency\n(all agents move)", 'contingency-shuffle':'contingency shuffle\n(all agents move, shuffled action mapping)', 'changeAgent-7':'switching embodiments\n(all agents move, self switches every 7 steps)'}
    new_env_name = {"logic": "logic", "contingency": "contingency", 'contingency-shuffle':'contingency shuffle', 'changeAgent-7':'switching embodiments'}
    df['env'] = df['env'].transform(lambda x: new_env_name[x])
    
    agent_order = ['base', 'hardcoded', 'human asymptote', 'attention limit 1 (uniform)', 'attention limit 1 (uniform), forget action mapping', 'attention limit 1 (posterior)', 'attention limit 1 (posterior), forget action mapping']#['base', 'hardcoded', 'human asymptote', 'forget_action_mapping_rand_attention_bias_1', '2_rand_attention_bias_1', '2_rand_forget_action_mapping_attention_bias_1', 'rand_attention_bias_1', 'rand_forget_action_mapping_attention_bias_1', "attention limit 1", "attention_limit_1_forget_action_mapping", 'attention limit 0', 'forget_action_mapping_attention_bias_0']#, 'attention_limit_1_prior_action_mapping', 'attention_limit_1_prior_action_mapping_and_noise']
    env_order = ["logic", "contingency", 'contingency shuffle', 'switching embodiments']
    sns.barplot(data=df, x='env', y='success', order = env_order, hue='agent', hue_order=agent_order, palette='hls', edgecolor='black')#['hardcoded' 'base', 'human_asymptote', "attention_limit_1", "attention_limit_1_forget_action_mapping", 'attention_limit_1_prior_action_mapping', 'attention_limit_1_prior_action_mapping_and_noise']
    plt.ylabel("Steps to success", fontweight="bold")
    plt.xlabel("Game type", fontweight="bold")
    plt.gcf().set_size_inches(10, 6)
    plt.savefig("plots/success3.png")
    plt.show()



if __name__ == "__main__":
    # compile all data
    all_data = {}
    for expe_name in ['final_exp.pkl']:#'all_data_50.pkl','forget_w_strong_prior.pkl', 'rand_attend_forget.pkl', 'rand_attend_2.pkl', 'rand_attend_2_2.pkl', 'rand_attend.pkl', 'test.pkl']:
        with open(save_dir + expe_name, 'rb') as f:
            temp = pickle.load(f)

        if len(temp.keys())==1:
            temp = temp[list(temp.keys())[0]]

        for env in temp.keys():
            all_data[env] = all_data.get(env, {})
            for agent in temp[env].keys():
                all_data[env][agent] = temp[env][agent]
    agents = ['base', 'rand_attention_bias_1', 'forget_action_mapping_rand_attention_bias_1', 'forget_action_mapping_attention_bias_1', 'attention_bias_1']#, 'rand_forget_action_mapping_attention_bias_1', '2_rand_forget_action_mapping_attention_bias_1', 'rand_attention_bias_1', '2_rand_attention_bias_1', 'attention_bias_0', 'base','forget_action_mapping_attention_bias_1', 'forget_action_mapping_attention_bias_0']
    envs = ['logic', 'contingency', 'contingency-shuffle', 'changeAgent-7']
    plot_mean_success(agents, envs)

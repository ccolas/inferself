#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
import pandas as pd
import scipy
import json
from matplotlib.ticker import FixedLocator, FixedFormatter

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
          '#7f7f7f', '#bcbd22', '#17becf']
sns.color_palette("husl", 8)
load_dir = '/Users/traceymills/Documents/cocosci_projects/self/inferself/src/output/'
plot_dir = "/Users/traceymills/Documents/cocosci_projects/self/inferself/src/analyses/plots/"



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

    path =  '/Users/traceymills/Documents/cocosci_projects/self/inferself/data/'
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
            data.append({'env':env_name + '-v0', 'agent':'human_asymptote','seed':i,'success':s})
        
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
            data.append({'env':env_name + '-v0', 'agent':'hardcoded','seed':i,'success':s})

        #get last 100 levels for RL agents
        for rl_agent in ['a2c_training', 'acer_training', 'dqn_training', 'ppo2_training', 'trpo_training', 'option_critic']:
            all_steps = []
            for seed_folder in os.listdir(path + folder + rl_agent + '/'):
                if seed_folder.startswith('.'):
                    continue
                #now open last 100 levels: train_3900.json
                if folder in ['logic_game/', 'contingency_game_shuffled_1/']:
                    num=1900
                else:
                    num=3900
                with open(path + folder + rl_agent + '/' + seed_folder + '/train_' + str(num) + '.json') as f:
                    iter = json.load(f)
                    all_steps = all_steps + iter['data']['steps']
            env_name = folder_to_env_name[folder]
            for (i,s) in enumerate(all_steps):
                data.append({'env':env_name + '-v0', 'agent':rl_agent,'seed':i,'success':s})

    df = pd.DataFrame.from_dict(data)
    df.to_csv('exp_data.csv')
    return data

def get_success_df(agents, envs):
    data = []
    for env in all_data.keys():
        if env.split('_False')[0] not in envs:
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
                    print(len(all_data[env][agent][seed]['success']))
                    print("here!", env, agent, seed)
                    success = 200
                
                if 'mode' in all_data[env][agent][seed].keys():
                    ind_exploit = np.argwhere(np.array(all_data[env][agent][seed]['mode'])=="exploit").flatten()
                    if len(ind_exploit)>0:
                        exploit = ind_exploit[0]
                    else:
                        exploit = None
                else:
                    exploit = None

                data.append({'env':env.split('_False')[0],'agent':agent,'seed':seed,'first_exploit':exploit, 'success':success})
    df = pd.DataFrame.from_dict(data)
    df2 = pd.read_csv('/Users/traceymills/Documents/cocosci_projects/self/inferself/src/analyses/exp_data.csv')
    df2["first_exploit"] = None
    df = pd.concat([df,df2], ignore_index = True)
    df = df[df["agent"].isin(agents)]
    return df

def plot_mean_success(agents, envs):
    df = get_success_df(agents, envs)
    #rename for plotting
    new_agent_name = {'base':'Meta-ePOMDP', 'foil':'Heuristic model', 'base_optimal':'implicit_base', 'base_prev':'base_prev', 'hardcoded':'hardcoded', 'human_asymptote':'Human asymptote', 'rand_attention_bias_1': 'attention limit 1',  'forget_action_mapping_rand_attention_bias_1': 'Resource rational meta-ePOMDP', 'attention_bias_1': 'attention limit 1 (posterior)',  'forget_action_mapping_attention_bias_1': 'attention limit 1 (posterior), forget action mapping'}
    df['agent'] = df['agent'].transform(lambda x: new_agent_name.get(x, x))
    #new_env_name = {"logic": "logic\n(only self moves)", "contingency": "contingency\n(all agents move)", 'contingency-shuffle':'contingency shuffle\n(all agents move, shuffled action mapping)', 'changeAgent-7':'switching embodiments\n(all agents move, self switches every 7 steps)'}
    new_env_name = {"logic-v0": "Logic", "contingency-v0": "Contingency", 'contingency-shuffle-v0':'Switching Mappings', 'changeAgent-7-v0':'Switching Embodiments'}
    df['env'] = df['env'].transform(lambda x: new_env_name[x])
    agent_order = [new_agent_name.get(a, a) for a in agents]#['base', 'human asymptote', 'attention limit 1', 'attention limit 1, forget action mapping', 'foil']#, 'attention limit 1 (posterior)', 'attention limit 1 (posterior), forget action mapping']#['base', 'hardcoded', 'human asymptote', 'forget_action_mapping_rand_attention_bias_1', '2_rand_attention_bias_1', '2_rand_forget_action_mapping_attention_bias_1', 'rand_attention_bias_1', 'rand_forget_action_mapping_attention_bias_1', "attention limit 1", "attention_limit_1_forget_action_mapping", 'attention limit 0', 'forget_action_mapping_attention_bias_0']#, 'attention_limit_1_prior_action_mapping', 'attention_limit_1_prior_action_mapping_and_noise']
    env_order = ["Logic", "Contingency", 'Switching Mappings', 'Switching Embodiments']
    sns.barplot(data=df, x='env', y='success',hue='agent', hue_order=agent_order, palette='hls', edgecolor='black', order = env_order)
    plt.ylabel("Average no. steps to complete level", fontweight="bold", fontsize=15)
    #max_val = max(df.groupby(['env', 'agent'])["success"].mean())
    plt.xticks(fontsize=13)
    plt.xlabel("Game type", fontweight="bold", fontsize=15)
    plt.legend(title="")#, title_fontweight='bold')
    plt.gcf().set_size_inches(10, 6)
    plt.savefig(plot_dir + "success_bar.png")
    plt.show()

def plot_mean_success_bar_break(agents, envs):
    df = get_success_df(agents, envs)
    #rename for plotting
    #new_agent_name = {'base':'self representing agent', 'foil':'heuristic', 'base_optimal':'implicit_base', 'base_prev':'base_prev', 'hardcoded':'hardcoded', 'human_asymptote':'human asymptote', 'rand_attention_bias_1': 'attention limit 1',  'forget_action_mapping_rand_attention_bias_1': 'resource rational self representing agent', 'attention_bias_1': 'attention limit 1 (posterior)',  'forget_action_mapping_attention_bias_1': 'attention limit 1 (posterior), forget action mapping'}
    new_agent_name = {'base':'Meta-ePOMDP', 'foil':'Heuristic model', 'base_optimal':'implicit_base', 'base_prev':'base_prev', 'hardcoded':'hardcoded', 'human_asymptote':'Human asymptote', 'rand_attention_bias_1': 'attention limit 1',  'forget_action_mapping_rand_attention_bias_1': 'Resource rational meta-ePOMDP', 'attention_bias_1': 'attention limit 1 (posterior)',  'forget_action_mapping_attention_bias_1': 'attention limit 1 (posterior), forget action mapping'}
    df['agent'] = df['agent'].transform(lambda x: new_agent_name.get(x, x))
    new_env_name = {"logic-v0": "Logic", "contingency-v0": "Contingency", 'contingency-shuffle-v0':'Switching Mappings', 'changeAgent-7-v0':'Switching Embodiments'}
    df['env'] = df['env'].transform(lambda x: new_env_name[x])
    agent_order = [new_agent_name.get(a, a) for a in agents]#['base', 'human asymptote', 'attention limit 1', 'attention limit 1, forget action mapping', 'foil']#, 'attention limit 1 (posterior)', 'attention limit 1 (posterior), forget action mapping']#['base', 'hardcoded', 'human asymptote', 'forget_action_mapping_rand_attention_bias_1', '2_rand_attention_bias_1', '2_rand_forget_action_mapping_attention_bias_1', 'rand_attention_bias_1', 'rand_forget_action_mapping_attention_bias_1', "attention limit 1", "attention_limit_1_forget_action_mapping", 'attention limit 0', 'forget_action_mapping_attention_bias_0']#, 'attention_limit_1_prior_action_mapping', 'attention_limit_1_prior_action_mapping_and_noise']
    env_order = ["Logic", "Contingency", 'Switching Mappings', 'Switching Embodiments']
    
    fig, (ax_top, ax_bottom) = plt.subplots(ncols=1, nrows=2, sharex=True, gridspec_kw={'hspace':0.1})

    sns.barplot(data=df, x='env', y='success',hue='agent', hue_order=agent_order, order = env_order, palette='viridis', edgecolor='black', ax=ax_top)#, errorbar="se") #order = env_order, 
    sns.barplot(data=df, x='env', y='success',hue='agent', hue_order=agent_order, order = env_order, palette='viridis', edgecolor='black', ax=ax_bottom)#, errorbar="se")
    ax_top.set_ylim(bottom=250)  
    ax_bottom.set_ylim(0,70)

    sns.despine(ax=ax_bottom)
    sns.despine(ax=ax_top, bottom=True)
    ax_top.tick_params(bottom=False)
    ax_top.set(xlabel=None, ylabel=None)
    ax_bottom.set(xlabel=None, ylabel=None)
    #add diagonal lines
    ax = ax_top
    d = .015  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax2 = ax_bottom
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    ax_bottom.legend_.remove()
    plt.xticks(fontsize=12)
    fig.text(0.5, 0.01, 'Game type', ha='center', fontsize=15, fontweight="bold")
    fig.text(0.06, 0.5, 'Average no. steps to complete level', va='center', fontsize=15, fontweight="bold", rotation='vertical')
    ax_top.legend(title="")
    plt.gcf().set_size_inches(11, 5.8)
    #plt.savefig(plot_dir + "success_bar2.png")
    plt.show()

def plot_mean_success_line(agents, envs):
    df = get_success_df(agents, envs)
    #rename for plotting
    new_agent_name = {'base':'Meta-ePOMDP', 'foil':'Heuristic model', 'base_optimal':'implicit_base', 'base_prev':'base_prev', 'hardcoded':'hardcoded', 'human_asymptote':'Human asymptote', 'rand_attention_bias_1': 'attention limit 1',  'forget_action_mapping_rand_attention_bias_1': 'Resource rational meta-ePOMDP', 'attention_bias_1': 'attention limit 1 (posterior)',  'forget_action_mapping_attention_bias_1': 'attention limit 1 (posterior), forget action mapping'}
    df['agent'] = df['agent'].transform(lambda x: new_agent_name.get(x, x))
    #new_env_name = {"logic": "logic\n(only self moves)", "contingency": "contingency\n(all agents move)", 'contingency-shuffle':'contingency shuffle\n(all agents move, shuffled action mapping)', 'changeAgent-7':'switching embodiments\n(all agents move, self switches every 7 steps)'}
    new_env_name = {"logic-v0": "Logic", "contingency-v0": "Contingency", 'contingency-shuffle-v0':'Switching Mappings', 'changeAgent-7-v0':'Switching Embodiments'}
    df['env'] = df['env'].transform(lambda x: new_env_name[x])
    agent_order = [new_agent_name.get(a, a) for a in agents]#['base', 'human asymptote', 'attention limit 1', 'attention limit 1, forget action mapping', 'foil']#, 'attention limit 1 (posterior)', 'attention limit 1 (posterior), forget action mapping']#['base', 'hardcoded', 'human asymptote', 'forget_action_mapping_rand_attention_bias_1', '2_rand_attention_bias_1', '2_rand_forget_action_mapping_attention_bias_1', 'rand_attention_bias_1', 'rand_forget_action_mapping_attention_bias_1', "attention limit 1", "attention_limit_1_forget_action_mapping", 'attention limit 0', 'forget_action_mapping_attention_bias_0']#, 'attention_limit_1_prior_action_mapping', 'attention_limit_1_prior_action_mapping_and_noise']
    env_order = ["Logic", "Contingency", 'Switching Mappings', 'Switching Embodiments']#agent_order = ['base', 'hardcoded', 'human asymptote', 'attention limit 1 (uniform)', 'attention limit 1 (uniform), forget action mapping', 'attention limit 1 (posterior)', 'attention limit 1 (posterior), forget action mapping']#['base', 'hardcoded', 'human asymptote', 'forget_action_mapping_rand_attention_bias_1', '2_rand_attention_bias_1', '2_rand_forget_action_mapping_attention_bias_1', 'rand_attention_bias_1', 'rand_forget_action_mapping_attention_bias_1', "attention limit 1", "attention_limit_1_forget_action_mapping", 'attention limit 0', 'forget_action_mapping_attention_bias_0']#, 'attention_limit_1_prior_action_mapping', 'attention_limit_1_prior_action_mapping_and_noise']
    #df['env_idx'] = df.apply(lambda x: env_order.index(x.env), axis=1)
    sns.catplot(data=df, x='env', y='success', hue='agent',  kind="point", palette='hls',  legend=False)#['hardcoded' 'base', 'human_asymptote', "attention_limit_1", "attention_limit_1_forget_action_mapping", 'attention_limit_1_prior_action_mapping', 'attention_limit_1_prior_action_mapping_and_noise']
    for line in plt.gca().lines:
        line.set_alpha(0.6) 
    for marker_collection in plt.gca().collections:
        marker_collection.set_alpha(0.6)
    #for mark in plt.gca().markers:
    #    mark.set_alpha(0.8) 
    plt.ylabel("Average no. steps to complete level", fontweight="bold", fontsize=15)
    #max_val = max(df.groupby(['env', 'agent'])["success"].mean())
    plt.xticks(fontsize=11)
    plt.xlabel("Game type", fontweight="bold", fontsize=15)
    plt.legend(title="")#, title_fontweight='bold')
    plt.gcf().set_size_inches(8, 6)
    plt.savefig(plot_dir + "success_line.png")
    plt.show()

def plot_mean_success_line_break(agents, envs):
    df = get_success_df(agents, envs)
    #rename for plotting
    new_agent_name = {'base':'Meta-ePOMDP', 'foil':'Heuristic model', 'base_optimal':'implicit_base', 'base_prev':'base_prev', 'hardcoded':'hardcoded', 'human_asymptote':'Human asymptote', 'rand_attention_bias_1': 'attention limit 1',  'forget_action_mapping_rand_attention_bias_1': 'Resource rational meta-ePOMDP', 'attention_bias_1': 'attention limit 1 (posterior)',  'forget_action_mapping_attention_bias_1': 'attention limit 1 (posterior), forget action mapping'}
    df['agent'] = df['agent'].transform(lambda x: new_agent_name.get(x, x))
    #new_env_name = {"logic": "logic\n(only self moves)", "contingency": "contingency\n(all agents move)", 'contingency-shuffle':'contingency shuffle\n(all agents move, shuffled action mapping)', 'changeAgent-7':'switching embodiments\n(all agents move, self switches every 7 steps)'}
    new_env_name = {"logic-v0": "Logic", "contingency-v0": "Contingency", 'contingency-shuffle-v0':'Switching Mappings', 'changeAgent-7-v0':'Switching Embodiments'}
    df['env'] = df['env'].transform(lambda x: new_env_name[x])
    agent_order = [new_agent_name.get(a, a) for a in agents]#['base', 'human asymptote', 'attention limit 1', 'attention limit 1, forget action mapping', 'foil']#, 'attention limit 1 (posterior)', 'attention limit 1 (posterior), forget action mapping']#['base', 'hardcoded', 'human asymptote', 'forget_action_mapping_rand_attention_bias_1', '2_rand_attention_bias_1', '2_rand_forget_action_mapping_attention_bias_1', 'rand_attention_bias_1', 'rand_forget_action_mapping_attention_bias_1', "attention limit 1", "attention_limit_1_forget_action_mapping", 'attention limit 0', 'forget_action_mapping_attention_bias_0']#, 'attention_limit_1_prior_action_mapping', 'attention_limit_1_prior_action_mapping_and_noise']
    env_order = ["Logic", "Contingency", 'Switching Mappings', 'Switching Embodiments']#agent_order = ['base', 'hardcoded', 'human asymptote', 'attention limit 1 (uniform)', 'attention limit 1 (uniform), forget action mapping', 'attention limit 1 (posterior)', 'attention limit 1 (posterior), forget action mapping']#['base', 'hardcoded', 'human asymptote', 'forget_action_mapping_rand_attention_bias_1', '2_rand_attention_bias_1', '2_rand_forget_action_mapping_attention_bias_1', 'rand_attention_bias_1', 'rand_forget_action_mapping_attention_bias_1', "attention limit 1", "attention_limit_1_forget_action_mapping", 'attention limit 0', 'forget_action_mapping_attention_bias_0']#, 'attention_limit_1_prior_action_mapping', 'attention_limit_1_prior_action_mapping_and_noise']
    #df['env_idx'] = df.apply(lambda x: env_order.index(x.env), axis=1)
    
    fig, (ax_top, ax_bottom) = plt.subplots(ncols=1, nrows=2, sharex=True, gridspec_kw={'hspace':0.1})
    sns.pointplot(data=df, x='env', y='success', hue='agent',  palette='hls', ax=ax_top) #order = env_order, 
    sns.pointplot(data=df, x='env', y='success', hue='agent',  palette='hls', ax=ax_bottom)
    ax_top.set_ylim(bottom=250)   # those limits are fake
    ax_bottom.set_ylim(0,70)
    sns.despine(ax=ax_bottom)
    sns.despine(ax=ax_top, bottom=True)
    ax_top.tick_params(bottom=False)
    ax_top.set(xlabel=None, ylabel=None)
    ax_bottom.set(xlabel=None, ylabel=None)
    #add diagonal lines
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    kwargs.update(transform=ax_bottom.transAxes)  # switch to the bottom axes
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.legend_.remove()
    
    for line in ax_top.lines:
        line.set_alpha(0.6) 
    for line in ax_bottom.lines:
        line.set_alpha(0.6) 
    for marker_collection in ax_top.collections:
        marker_collection.set_alpha(0.6)
    for marker_collection in ax_bottom.collections:
        marker_collection.set_alpha(0.6)
    ax_top.legend(title="")
    fig.text(0.5, 0.04, 'Game type', ha='center', fontsize=15, fontweight="bold")
    fig.text(0.04, 0.5, 'Average no. steps to complete level', va='center', fontsize=15, fontweight="bold", rotation='vertical')
    plt.gcf().set_size_inches(8, 6)
    plt.savefig(plot_dir + "success_line.png")
    plt.show()


def get_game_type(env):
    if "contingency-shuffle" in env:
        return "contingency-shuffle"
    elif "contingency" in env:
        return "contingency"
    elif "logic" in env:
        return "logic"
    elif "changeAgent" in env:
        return "changeAgent"

def get_game_subtype(game_type, env):
    if game_type=="changeAgent":
        return env.split(game_type)[1][3:]
    else:
        return env.split(game_type)[1][1:]

def plot_mean_success_grid(agents, envs):
    df = get_success_df(agents, envs)
    df["temp"] = df["env"]
    df["game_type"] = df.apply(lambda row: get_game_type(row["env"]), axis=1)
    df["env"] = df.apply(lambda row: get_game_subtype(row["game_type"], row["env"]), axis=1)
    new_game_name = {"logic": "logic", "contingency": "contingency", 'contingency-shuffle':'contingency shuffle', 'changeAgent':'switching embodiments'}
    df['game_type'] = df['game_type'].transform(lambda x: new_game_name[x])
    #agent_order = ['base', 'foil']
    #env_order = ["logic", "contingency", 'contingency shuffle', 'switching embodiments']
    g = sns.FacetGrid(df, col="game_type", col_wrap=2, sharex=False)
    g.map_dataframe(sns.barplot, x="env", y="success", hue="agent",
          order=["5-easy", "5-hard", "8-easy","8-hard", "12-easy", "12-hard"],
          estimator="mean", palette='hls', edgecolor='black',
          errorbar="se").add_legend()
    for item, ax in g.axes_dict.items():
        ax.set_title(item, weight="bold", size=12)
        ax.set_xlabel('')
    plt.gcf().set_size_inches(10, 8)
    #plt.savefig(plot_dir + "generalize.png")
    plt.show()


def time_to_exploit(envs):
    df = get_success_df(['base'], envs)
    df["temp"] = df["env"]
    df["game_type"] = df.apply(lambda row: get_game_type(row["env"]), axis=1)
    df["env"] = df.apply(lambda row: get_game_subtype(row["game_type"], row["env"]), axis=1)
    new_game_name = {"logic": "logic", "contingency": "contingency", 'contingency-shuffle':'contingency shuffle', 'changeAgent':'switching embodiments'}
    df['game_type'] = df['game_type'].transform(lambda x: new_game_name[x])

    #now, we want idx of first 'exploit'
    #agent_order = ['base', 'foil']
    #env_order = ["logic", "contingency", 'contingency shuffle', 'switching embodiments']
    g = sns.FacetGrid(df, col="game_type", col_wrap=2, sharex=False)
    g.map_dataframe(sns.barplot, x="env", y="first_exploit",
          order=["v0", "5-easy", "5-hard", "8-easy","8-hard", "12-easy", "12-hard"],
          estimator="mean", palette='hls', edgecolor='black',
          errorbar="se").add_legend()
    for item, ax in g.axes_dict.items():
        ax.set_title(item, weight="bold", size=12)
        ax.set_xlabel('')
    plt.gcf().set_size_inches(10, 8)
    #plt.savefig(plot_dir + "exploit.png")
    plt.show()

def print_stats(agents, envs):
    df = get_success_df(agents, envs)
    df_base = df[df["agent"]=="base"]
    df_human = df[df["agent"]=="human_asymptote"]
    df_rr = df[df["agent"]=="forget_action_mapping_rand_attention_bias_1"]
    df_heur = df[df["agent"]=="foil"]
    
    
    mean_df = df.groupby(['env', 'agent'], as_index=False)["success"].mean()
    #print(mean_df.head(20))
    print(mean_df.columns)
    print(scipy.stats.pearsonr(mean_df[mean_df["agent"]=="human_asymptote"]["success"], mean_df[mean_df["agent"]=="forget_action_mapping_rand_attention_bias_1"]["success"]))
    print(scipy.stats.pearsonr(mean_df[mean_df["agent"]=="human_asymptote"]["success"], mean_df[mean_df["agent"]=="base"]["success"]))
    print(scipy.stats.pearsonr(mean_df[mean_df["agent"]=="human_asymptote"]["success"], mean_df[mean_df["agent"]=="foil"]["success"]))
    print(scipy.stats.pearsonr(mean_df[mean_df["agent"]=="base"]["success"], mean_df[mean_df["agent"]=="forget_action_mapping_rand_attention_bias_1"]["success"]))


def load_data(exp_names):
    all_data = {}
    for expe_name in exp_names:
        with open(load_dir + expe_name, 'rb') as f:
            temp = pickle.load(f)
        if len(temp.keys())==1:
            temp = temp[list(temp.keys())[0]]
        for env in temp.keys():
            all_data[env] = all_data.get(env, {})
            for agent in temp[env].keys():
                all_data[env][agent] = temp[env][agent]
    return all_data

if __name__ == "__main__": 
    envs = ['logic-v0', 'contingency-v0', 'contingency-shuffle-v0', 'changeAgent-7-v0'] 
    #all_data = load_data(['comparison_exp.pkl'])
    agents = ['base', 'foil']
    """
    for nm in ['logic', 'contingency', 'contingency-shuffle', 'changeAgent-7']:
        envs.append(nm + '-5-easy')
        envs.append(nm + '-5-hard')
        envs.append(nm + '-8-easy')
        envs.append(nm + '-8-hard')
        envs.append(nm + '-12-easy')
        envs.append(nm + '-12-hard')
    """
    #plot_mean_success_grid(agents, envs)
    #time_to_exploit(envs)
    
    
    all_data = load_data(['exp.pkl'])#['general_exp.pkl', 'foil_exp.pkl'])
    #agents = ['base', 'foil', 'human_asymptote', 'forget_action_mapping_rand_attention_bias_1', 'a2c_training', 'acer_training', 'dqn_training', 'ppo2_training', 'trpo_training', 'option_critic']#, 'rand_attention_bias_1', 'forget_action_mapping_rand_attention_bias_1']
    agents = ['human_asymptote', 'base', 'forget_action_mapping_rand_attention_bias_1', 'foil']
    
    #print_stats(agents, envs)
    plot_mean_success_bar_break(agents, envs)
    #plot_mean_success_line(agents, envs)
    #plot_mean_success_line_break(agents, envs)





#2 with threshold 0.95
    #but this makes the agent way too good in contingency
#3 with ==
    #but this makes the switch ones way too bad
#4 with == and ratio threshold for explore at 1.5
#can also increase noise estimate

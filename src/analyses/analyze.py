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
from statannotations.Annotator import Annotator
import matplotlib.ticker as ticker
from statsmodels.stats.power import TTestIndPower

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
          '#7f7f7f', '#bcbd22', '#17becf']
sns.color_palette("husl", 8)
load_dir = '/Users/traceymills/Dropbox (MIT)/cocosci_projects/self/inferself/src/output/'
load_dir = '../output/'
plot_dir = "/Users/traceymills/Dropbox (MIT)/cocosci_projects/self/inferself/src/analyses/plots/"
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

def get_exp_data(mean=True):
    data = []
    folder_to_env_name = {'logic_game/':"logic", 'contingency_game/':"contingency", 'contingency_game_shuffled_1/':'contingency-shuffle', 'change_agent_game/':'changeAgent-7'}
    extended_vsn = {"logic":"logicExtended-v0", "contingency":'contingencyExtended-v0', 'contingency-shuffle':'contingencyExtended-shuffle-v0', 'changeAgent-7':'changeAgentExtended-7-v0'}
    path =  '/Users/traceymills/Dropbox (MIT)/cocosci_projects/self/data/'
    for folder in ['logic_game/', 'contingency_game/', 'contingency_game_shuffled_1/', 'change_agent_game/', ]:
        #get last 80 levels of human play
        all_steps = []
        for fname in os.listdir(path + folder + 'human/'):
            if fname.startswith('.'):
                continue
            with open(path + folder + 'human/' + fname) as f:
                subj = json.load(f)
            if mean:
                all_steps = all_steps + [np.mean(subj['data']['steps'][20:])]
            else:
                all_steps = all_steps + subj['data']['steps'][20:]
        env_name = folder_to_env_name[folder]
        for (i,s) in enumerate(all_steps):
            data.append({'env':env_name + '-v0', 'agent':'human_asymptote','seed':i,'success':s})
        
        all_steps = []
        for fname in os.listdir(path + folder + 'human_extended/'):
            if fname.startswith('.'):
                continue
            with open(path + folder + 'human_extended/' + fname) as f:
                subj = json.load(f)
            if mean:
                all_steps = all_steps + [np.mean(subj['data']['steps'][20:])]
            else:
                all_steps = all_steps + subj['data']['steps'][20:]
        env_name = folder_to_env_name[folder]
        for (i,s) in enumerate(all_steps):
            data.append({'env':extended_vsn[env_name], 'agent':'human_asymptote','seed':i,'success':s})
        
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
    df.to_csv('exp_data_means.csv')
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
                    success = 10000
                
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
    df2 = pd.read_csv('/Users/traceymills/Dropbox (MIT)/cocosci_projects/self/inferself/src/analyses/exp_data_means.csv')
    df2["first_exploit"] = None
    df = pd.concat([df,df2], ignore_index = True)
    df = df[df["agent"].isin(agents)]
    df = df[df["env"].isin(envs)]
    return df

def plot_mean_success(agents, envs):
    df = get_success_df(agents, envs)
    #rename for plotting
    new_agent_name = {'base':'Meta-ePOMDP', 'foil':'Heuristic model', 'base_optimal':'implicit_base', 'base_prev':'base_prev', 'hardcoded':'hardcoded', 'human_asymptote':'Human asymptote', 'rand_attention_bias_1': 'attention limit 1',  'forget_action_mapping_rand_attention_bias_1': 'Resource rational meta-ePOMDP', 'attention_bias_1': 'attention limit 1 (posterior)',  'forget_action_mapping_attention_bias_1': 'attention limit 1 (posterior), forget action mapping'}
    df['agent'] = df['agent'].transform(lambda x: new_agent_name.get(x, x))
    #new_env_name = {"logic": "logic\n(only self moves)", "contingency": "contingency\n(all agents move)", 'contingency-shuffle':'contingency shuffle\n(all agents move, shuffled action mapping)', 'changeAgent-7':'switching embodiments\n(all agents move, self switches every 7 steps)'}
    new_env_name = {"logic-v0": "Logic", "logic_u-v0": "Logic\nmult. goals", "contingency-v0": "Contingency", "contingency_u-v0": "Contingency\nmult. goals", 'contingency-shuffle-v0':'Switch Mappings', 'contingency_u-shuffle-v0':'Switch Mappings\nmult. goals.', 'changeAgent-7-v0':'Switch Embodiments', 'changeAgent_u-7-v0':'Switch Embodiments\nmult. goals', "logicExtended-v0": "Logic (5)", 'contingencyExtended-v0':"Contingency (5)", 'contingencyExtended-shuffle-v0':"Switching Mappings (5)", 'changeAgentExtended-7-v0':'Switching Embodiments (5)'}
    df['env'] = df['env'].transform(lambda x: new_env_name[x])
    agent_order = [new_agent_name.get(a, a) for a in agents]#['base', 'human asymptote', 'attention limit 1', 'attention limit 1, forget action mapping', 'foil']#, 'attention limit 1 (posterior)', 'attention limit 1 (posterior), forget action mapping']#['base', 'hardcoded', 'human asymptote', 'forget_action_mapping_rand_attention_bias_1', '2_rand_attention_bias_1', '2_rand_forget_action_mapping_attention_bias_1', 'rand_attention_bias_1', 'rand_forget_action_mapping_attention_bias_1', "attention limit 1", "attention_limit_1_forget_action_mapping", 'attention limit 0', 'forget_action_mapping_attention_bias_0']#, 'attention_limit_1_prior_action_mapping', 'attention_limit_1_prior_action_mapping_and_noise']
    env_order = [new_env_name.get(e, e) for e in envs]
    ax = sns.barplot(data=df, x='env', y='success',hue='agent', hue_order=agent_order, palette='viridis', edgecolor='black', order = env_order)
    plt.ylabel("Average no. steps to complete level", fontweight="bold", fontsize=15)
    #max_val = max(df.groupby(['env', 'agent'])["success"].mean())
    plt.xticks(fontsize=9)
    plt.xlabel("Game type", fontweight="bold", fontsize=15)
    
    pairs = []
    pvalues = []
    print(pd.unique(df["agent"]))
    to_compare = [('Human asymptote', 'Meta-ePOMDP'), ('Human asymptote', 'Resource rational meta-ePOMDP')] #('Human asymptote', 'Heuristic model')]
    for env in env_order:
        for pair in to_compare:
            pairs.append([(env, pair[0]), (env, pair[1])])
            val = scipy.stats.ttest_ind(df[(df['agent']==pair[0]) & (df['env']==env)]["success"], df[(df['agent']==pair[1]) & (df['env']==env)]["success"], equal_var=False)[1]
            if val < 0.0001:
                pvalues.append("****")
            elif val < 0.001:
                pvalues.append("***")
            elif val < 0.01:
                pvalues.append("**")
            elif val < 0.05:
                pvalues.append("*")
            else:
                pvalues.append('p=' + "{:.2f}".format(val).lstrip('0'))
            print(pvalues)
    params = {'data':df, 'x':'env', 'y':'success', 'hue':'agent', 'hue_order':agent_order, 'order':env_order, 'palette':'viridis', 'edgecolor':'black'}
    ann = Annotator(ax, pairs, **params)
    ann.set_custom_annotations(pvalues)
    ann.annotate()
    
    plt.legend(title="")#, title_fontweight='bold')
    plt.gcf().set_size_inches(20, 6)
    plt.savefig(plot_dir + "success_bar.png")
    plt.show()

def plot_mean_success_bar_break(agents, envs):
    df = get_success_df(agents, envs)
    #rename for plotting
    #new_agent_name = {'base':'Self-representing agent', 'foil':'Heuristic', 'human_asymptote':'Humans',  'forget_action_mapping_rand_attention_bias_1': 'Resource-rational self-representing agent'}
    new_agent_name = {'base':'Meta-ePOMDP', 'foil':'Heuristic model', 'base_optimal':'implicit_base', 'base_prev':'base_prev', 'hardcoded':'hardcoded', 'human_asymptote':'Human asymptote', 'rand_attention_bias_1': 'attention limit 1',  'forget_action_mapping_rand_attention_bias_1': 'Resource rational meta-ePOMDP', 'attention_bias_1': 'attention limit 1 (posterior)',  'forget_action_mapping_attention_bias_1': 'attention limit 1 (posterior), forget action mapping'}
    df['agent'] = df['agent'].transform(lambda x: new_agent_name.get(x, x))
    new_env_name = {"logic-v0": "Logic", "logic_u-v0": "Logic\nmult. goals", "contingency-v0": "Contingency", "contingency_u-v0": "Contingency\nmult. goals", 'contingency-shuffle-v0':'Switching Mappings', 'contingency_u-shuffle-v0':'Switching Mappings\nmult. goals.', 'changeAgent-7-v0':'Switching Embodiments', 'changeAgent_u-7-v0':'Switching Embodiments\nmult. goals', "logicExtended-v0": "Logic (5)", 'contingencyExtended-v0':"Contingency (5)", 'contingencyExtended-shuffle-v0':"Switching Mappings (5)", 'changeAgentExtended-7-v0':'Switching Embodiments (5)'}
    df['env'] = df['env'].transform(lambda x: new_env_name.get(x,x))
    
    #change success for Heuristic Contingency to be mean of success
    df.loc[(df['agent']=='Heuristic model') & (df['env']=='Contingency'), 'success'] = np.mean(df[(df['agent']=='Heuristic model') & (df['env']=='Contingency')]['success'])
    df.loc[(df['agent']=='Heuristic model') & (df['env']=='Contingency (5)'), 'success'] = np.mean(df[(df['agent']=='Heuristic model') & (df['env']=='Contingency (5)')]['success'])
    df.loc[(df['agent']=='Heuristic model') & (df['env']=='Switching Mappings (5)'), 'success'] = np.mean(df[(df['agent']=='Heuristic model') & (df['env']=='Switching Mappings (5)')]['success'])
    
    agent_order = [new_agent_name.get(a, a) for a in agents]#['base', 'human asymptote', 'attention limit 1', 'attention limit 1, forget action mapping', 'foil']#, 'attention limit 1 (posterior)', 'attention limit 1 (posterior), forget action mapping']#['base', 'hardcoded', 'human asymptote', 'forget_action_mapping_rand_attention_bias_1', '2_rand_attention_bias_1', '2_rand_forget_action_mapping_attention_bias_1', 'rand_attention_bias_1', 'rand_forget_action_mapping_attention_bias_1', "attention limit 1", "attention_limit_1_forget_action_mapping", 'attention limit 0', 'forget_action_mapping_attention_bias_0']#, 'attention_limit_1_prior_action_mapping', 'attention_limit_1_prior_action_mapping_and_noise']
    env_order = [new_env_name.get(e, e) for e in envs]
    #print(env_order)
    factor = 2.788
    fig, (ax_top, ax_bottom) = plt.subplots(ncols=1, nrows=2, sharex=True, gridspec_kw={'hspace':0.05, 'height_ratios': [1, factor]})

    sns.barplot(data=df, x='env', y='success',hue='agent', hue_order=agent_order, order = env_order, palette='viridis', edgecolor='black', ax=ax_top)#, errorbar="se") #order = env_order, 
    sns.barplot(data=df, x='env', y='success',hue='agent', hue_order=agent_order, order = env_order, palette='viridis', edgecolor='black', ax=ax_bottom)#, errorbar="se")
    ax_top.set_ylim(bottom=405)#1300)
    ax_bottom.set_ylim(0,100)#70)#130)

    #add annotations
    """
    pairs = []
    pvalues = []
    to_compare = [('Human asymptote', 'Meta-ePOMDP'), ('Human asymptote', 'Resource rational meta-ePOMDP'), ('Human asymptote', 'Heuristic model')]
    for env in env_order:
        for pair in to_compare:
            pairs.append([(env, pair[0]), (env, pair[1])])
            val = scipy.stats.ttest_ind(df[(df['agent']==pair[0]) & (df['env']==env)]["success"], df[(df['agent']==pair[1]) & (df['env']==env)]["success"], equal_var=False)[1]
            if val < 0.0001:
                pvalues.append("****")
            elif val < 0.001:
                pvalues.append("***")
            elif val < 0.01:
                pvalues.append("**")
            elif val < 0.05:
                pvalues.append("*")
            else:
                pvalues.append('p=' + "{:.2f}".format(val).lstrip('0'))
            print(pvalues)
    params = {'data':df, 'x':'env', 'y':'success', 'hue':'agent', 'hue_order':agent_order, 'order':env_order, 'palette':'viridis', 'edgecolor':'black'}
    ann = Annotator(ax_top, pairs, **params)
    ann.set_custom_annotations(pvalues)
    ann.annotate()
    """
    sns.despine(ax=ax_bottom)
    sns.despine(ax=ax_top, bottom=True)
    ax_top.tick_params(bottom=False)
    ax_top.set(xlabel=None, ylabel=None)
    ax_bottom.set(xlabel=None, ylabel=None)
    #ax_top.yaxis.set_major_locator(ticker.MultipleLocator(base=10))
    #add diagonal lines
    ax = ax_top
    
    d = .015  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-factor*d, +factor*d), **kwargs)        # top-left diagonal
    
    ax2 = ax_bottom
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    plt.xticks(fontsize=12)
    fig.text(0.5, 0.01, 'Game type', ha='center', fontsize=15, fontweight="bold")
    fig.text(0.06, 0.5, 'Average no. steps to complete level', va='center', fontsize=15, fontweight="bold", rotation='vertical')
   
    ax_top.legend_.remove()
    ax_bottom.legend(title="", loc='center left', bbox_to_anchor=(0.03, 0.7))
    plt.gcf().set_size_inches(12, 7)
    plt.savefig(plot_dir + "success_bar2.png")
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

def print_vars(agents, envs):
    df = get_success_df(agents, envs)
    df_base = df[df["agent"]=="base"]
    df_human = df[df["agent"]=="human_asymptote"]
    df_rr = df[df["agent"]=="forget_action_mapping_rand_attention_bias_1"]
    df_heur = df[df["agent"]=="foil"]
    stds = {}
    steps = {}
    path =  '/Users/traceymills/Documents/cocosci_projects/self/data/'
    folder_to_env_name = {'logic_game/':"logic", 'contingency_game/':"contingency", 'contingency_game_shuffled_1/':'contingency-shuffle', 'change_agent_game/':'changeAgent-7'}
    for folder in ['logic_game/', 'contingency_game/', 'contingency_game_shuffled_1/', 'change_agent_game/']:
        env_name = folder_to_env_name[folder]
        stds[env_name] = []
        steps[env_name] = []
        for fname in os.listdir(path + folder + 'human/'):
            if fname.startswith('.'):
                continue
            with open(path + folder + 'human/' + fname) as f:
                subj = json.load(f)
            stds[env_name] = stds[env_name] + [np.std(subj['data']['steps'][20:])]
            steps[env_name] = steps[env_name] + [subj['data']['steps'][20:]]
    for env in pd.unique(df["env"]):
        print(env)
        base_succ = df_base[df_base["env"]==env]["success"]
        heur_succ = df_heur[df_heur["env"]==env]["success"]
        hum_succ = df_human[df_human["env"]==env]["success"]
        rr_succ = df_rr[df_rr["env"]==env]["success"]
        print(np.var(base_succ))
        print("inter-avg std")
        print(np.std(hum_succ))
        print("mean of within subj stds")
        for i in range(len(stds[env[:-3]])):   
            print(stds[env[:-3]][i])
            print(np.std(steps[env[:-3]][i]))
            print(steps[env[:-3]][i])
        print(len(stds[env[:-3]]))
        print(np.mean(stds[env[:-3]]))
        print(np.var(rr_succ))
        print(np.var(heur_succ))
        print(np.var(hum_succ) / np.var(rr_succ))
    

def print_stats(agents, envs):
    df = get_success_df(agents, envs)
    df_base = df[df["agent"]=="base"]
    df_human = df[df["agent"]=="human_asymptote"]
    df_rr = df[df["agent"]=="forget_action_mapping_rand_attention_bias_1"]
    df_heur = df[df["agent"]=="foil"]
    for env in pd.unique(df["env"]):
        print(env)
        base_succ = df_base[df_base["env"]==env]["success"]
        heur_succ = df_heur[df_heur["env"]==env]["success"]
        hum_succ = df_human[df_human["env"]==env]["success"]
        rr_succ = df_rr[df_rr["env"]==env]["success"]
        #print(scipy.stats.mannwhitneyu(base_succ, hum_succ))
        print(scipy.stats.ttest_ind(base_succ, hum_succ, equal_var=False))
        #print(scipy.stats.mannwhitneyu(rr_succ, hum_succ))
        print(scipy.stats.ttest_ind(rr_succ, hum_succ, equal_var=False))
        #print(scipy.stats.mannwhitneyu(heur_succ, hum_succ))
        print(scipy.stats.ttest_ind(heur_succ, hum_succ, equal_var=False))
    
    mean_df = df.groupby(['env', 'agent'], as_index=False)["success"].mean()
    print("correlations:")
    print(scipy.stats.pearsonr(mean_df[mean_df["agent"]=="human_asymptote"]["success"], mean_df[mean_df["agent"]=="forget_action_mapping_rand_attention_bias_1"]["success"]))
    print(scipy.stats.pearsonr(mean_df[mean_df["agent"]=="human_asymptote"]["success"], mean_df[mean_df["agent"]=="base"]["success"]))
    print(scipy.stats.pearsonr(mean_df[mean_df["agent"]=="human_asymptote"]["success"], mean_df[mean_df["agent"]=="foil"]["success"]))
    print(scipy.stats.pearsonr(mean_df[mean_df["agent"]=="base"]["success"], mean_df[mean_df["agent"]=="forget_action_mapping_rand_attention_bias_1"]["success"]))

#probability of rejecting null hyp if it should be rejected, 1 - P(false negative)
def power_analaysis():
    #based on what we know of human variability,
    #how many people do we need to run to see sig. differences bt normal game type and GU game type
    #we know, for each game type:
    #human variance
    #model variance
    #num steps model took 1
    #num steps model took 2
    
    #logic game

    for env in ["logic", "contingency", "contingency-shuffle", "changeAgent-7"]:
        print("--------------------")
        print(env)
        df1 = get_success_df(["human_asymptote", "forget_action_mapping_rand_attention_bias_1"], [env + "-v0"])
        human1 = df1[df1["agent"]=="human_asymptote"]["success"]
        model1 = df1[df1["agent"]=="forget_action_mapping_rand_attention_bias_1"]["success"]
        hum_var_1 = np.var(human1)
        model_var_1 = np.var(model1)
        model_mean_1 = np.mean(model1)
        human_mean_1 = np.mean(model1)
        
        if env == 'contingency-shuffle':
            env = 'contingency_u-shuffle'
        elif env == 'changeAgent-7':
            env = 'changeAgent_u-7'
        else:
            env = env + "_u"
        df2 = get_success_df(["human_asymptote", "forget_action_mapping_rand_attention_bias_1"], [env + "-v0"])
        #human2 = df[df["agent"]=="human_asymptote"]["success"]
        model2 = df2[df2["agent"]=="forget_action_mapping_rand_attention_bias_1"]["success"]
        #hum_var_2 = np.var(human2)
        model_var_2 = np.var(model2)
        model_mean_2 = np.mean(model2)
        #human_mean_2 = np.mean(model2)
        
        #using model mean 2 and human var 1, how many participants do we need to get significant difference from 
        # estimate sample size via power analysis
        # parameters for power analysis
        scale = (model_var_2 ** 0.5)/(model_var_1 ** 0.5)
        est_std = (hum_var_1**0.5)
        print(est_std)
        print(model_mean_2 - human_mean_1)
        #est_std = hum_var_1 + model_var_2) **0.5
        effect = (model_mean_2 - human_mean_1)/est_std #difference between the two means divided by the standard deviation
        print(effect)
        alpha = 0.05
        power = 0.8
        # perform power analysis
        analysis = TTestIndPower()
        result = analysis.solve_power(effect_size=float(effect), power=float(power), nobs1=float(len(human1)), alpha=float(alpha), ratio=None,  alternative='two-sided')
        print('Ratio: %.3f' % result)
        result = analysis.solve_power(effect_size=float(effect), power=float(power), nobs1=None, alpha=float(alpha), ratio=1.0, alternative='two-sided')
        print('Sample size: %.3f' % result)

    analysis = TTestIndPower()
    result = analysis.solve_power(effect_size=float(0.65), power=float(0.8), nobs1=float(20), alpha=float(0.05), ratio=None,  alternative='two-sided')
    print(result)

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

def compare_data(agents, envs):
    df = get_success_df(agents, envs)
    df_new = pd.read_csv('/Users/traceymills/Dropbox (MIT)/cocosci_projects/self/game_app/data/success.csv')
    df_new["agent"] = "humans_new"
    df = pd.concat([df, df_new])
    
    new_agent_name = {'base':'Meta-ePOMDP', 'foil':'Heuristic model', 'base_optimal':'implicit_base', 'base_prev':'base_prev', 'hardcoded':'hardcoded', 'human_asymptote':'Human og', 'humans_new':'Human new', 'rand_attention_bias_1': 'attention limit 1',  'forget_action_mapping_rand_attention_bias_1': 'Resource rational meta-ePOMDP', 'attention_bias_1': 'attention limit 1 (posterior)',  'forget_action_mapping_attention_bias_1': 'attention limit 1 (posterior), forget action mapping'}
    df['agent'] = df['agent'].transform(lambda x: new_agent_name.get(x, x))
    new_env_name = {"logic-v0": "Logic", "logic_u-v0": "Logic\nmult. goals", "contingency-v0": "Contingency", "contingency_u-v0": "Contingency\nmult. goals", 'contingency-shuffle-v0':'Switch Mappings', 'contingency_u-shuffle-v0':'Switch Mappings\nmult. goals.', 'changeAgent-7-v0':'Switch Embodiments', 'changeAgent_u-7-v0':'Switch Embodiments\nmult. goals', "logicExtended-v0": "Logic (5)", 'contingencyExtended-v0':"Contingency (5)", 'contingencyExtended-shuffle-v0':"Switching Mappings (5)", 'changeAgentExtended-7-v0':'Switching Embodiments (5)'}
    df['env'] = df['env'].transform(lambda x: new_env_name[x])
    agent_order = ["Human og", "Human new", "Resource rational meta-ePOMDP"]
    env_order = [new_env_name.get(e, e) for e in envs]
    ax = sns.barplot(data=df, x='env', y='success',hue='agent', hue_order=agent_order, palette='viridis', edgecolor='black', order = env_order)
    plt.ylabel("Average no. steps to complete level", fontweight="bold", fontsize=15)
    #max_val = max(df.groupby(['env', 'agent'])["success"].mean())
    plt.xticks(fontsize=9)
    plt.xlabel("Game type", fontweight="bold", fontsize=15)
    
    pairs = []
    pvalues = []
    print(pd.unique(df["agent"]))
    to_compare = [('Human og', 'Human new'), ('Human new', 'Resource rational meta-ePOMDP')] #('Human asymptote', 'Heuristic model')]
    for env in env_order:
        for pair in to_compare:
            pairs.append([(env, pair[0]), (env, pair[1])])
            val = scipy.stats.ttest_ind(df[(df['agent']==pair[0]) & (df['env']==env)]["success"], df[(df['agent']==pair[1]) & (df['env']==env)]["success"], equal_var=False)[1]
            if val < 0.0001:
                pvalues.append("****")
            elif val < 0.001:
                pvalues.append("***")
            elif val < 0.01:
                pvalues.append("**")
            elif val < 0.05:
                pvalues.append("*")
            else:
                pvalues.append('p=' + "{:.2f}".format(val).lstrip('0'))
            print(pvalues)
    params = {'data':df, 'x':'env', 'y':'success', 'hue':'agent', 'hue_order':agent_order, 'order':env_order, 'palette':'viridis', 'edgecolor':'black'}
    ann = Annotator(ax, pairs, **params)
    ann.set_custom_annotations(pvalues)
    ann.annotate()
    
    plt.legend(title="")#, title_fontweight='bold')
    plt.gcf().set_size_inches(10, 6)
    #plt.savefig(plot_dir + "success_bar.png")
    plt.show()
    
    
    
    



if __name__ == "__main__": 
    #get_exp_data(mean=True)
    envs = ['logic-v0', 'logic_u-v0', 'contingency-v0', 'contingency_u-v0', 'contingency-shuffle-v0', 'contingency_u-shuffle-v0', 'changeAgent-7-v0', 'changeAgent_u-7-v0']
    #envs = ['logic-v0', 'contingency-v0', 'contingency-shuffle-v0', 'changeAgent-7-v0']
    #envs = ["logicExtended-v0", 'contingencyExtended-v0', 'contingencyExtended-shuffle-v0', 'changeAgentExtended-7-v0']
   
    """
    all_data = load_data(['comparison_exp.pkl'])
    for nm in ['logic', 'contingency', 'contingency-shuffle', 'changeAgent-7']:
        envs.append(nm + '-5-easy')
        envs.append(nm + '-5-hard')
        envs.append(nm + '-8-easy')
        envs.append(nm + '-8-hard')
        envs.append(nm + '-12-easy')
        envs.append(nm + '-12-hard')
    """
    all_data = load_data(['exp.pkl', 'goal_uncertainty.pkl'])#,'exp.pkl's, 'goal_uncertainty2.pkl', 'extended_exp.pkl', 'extended_exp_foil.pkl'])#['general_exp.pkl', 'foil_exp.pkl'])
    agents = ['human_asymptote', 'forget_action_mapping_rand_attention_bias_1', 'base']
    compare_data(agents, envs)
    #power_analaysis()
    #agents = ['human_asymptote', 'forget_action_mapping_rand_attention_bias_1', 'base', 'foil']
    #agents = ['human_asymptote', 'forget_action_mapping_rand_attention_bias_1', 'base']#[ 'foil']
    
    #plot_mean_success(agents, envs)





#2 with threshold 0.95
    #but this makes the agent way too good in contingency
#3 with ==
    #but this makes the switch ones way too bad
#4 with == and ratio threshold for explore at 1.5
#can also increase noise estimate

import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import seaborn as sns
import pandas as pd
import scipy
import json
from matplotlib.ticker import FixedLocator, FixedFormatter
#from statannotations.Annotator import Annotator
import matplotlib.ticker as ticker
#from statsmodels.stats.power import TTestIndPower

asymptote = 35
model_data_path = '/Users/traceymills/Dropbox (MIT)/cocosci_projects/self/inferself/src/output/'
new_human_data_path = '/Users/traceymills/Dropbox (MIT)/cocosci_projects/self/inferself_app/data/'


#compare humans and rr model in change agent uncertainty game
def compare_one(env):
    def custom2(y, **kwargs):
        ym = y[asymptote:].mean()
        plt.axhline(ym, alpha=0.5)
        plt.annotate(f"mean: {ym:.1f}", xy=(1,900), color="blue",
                    xycoords=plt.gca().get_yaxis_transform(), ha="right")
    def custom1(y, **kwargs):
        ym = y.mean()
        plt.axhline(ym, alpha=0.5, color="green")
        plt.annotate(f"mean: {ym:.1f}", xy=(1,800), color="green",
                    xycoords=plt.gca().get_yaxis_transform(), ha="right")
    env_name_dict = {'change_agent_u': "changeAgent_u-7-v0_False"}
    model_data = load_data(['uncertainty_exp.pkl'])[env_name_dict[env]]['forget_action_mapping_rand_attention_bias_1']
    model_steps = [np.argwhere(np.array(model_data[str(i)]['success'])).flatten()[0] for i in range(100)]  
    with open(new_human_data_path + 'trials.json') as f:
        trials = json.load(f)
    trials = [t for t in trials if "test" not in t["subject_id"]]
    trials = [t for t in trials if t["game_type"]==env]
    plot_d = {"subj_steps":[],"model_steps":[],"level":[],"subj":[]}
    for i, subj in enumerate(trials):
        subj_steps = subj["game_data"]["steps"]
        n = len(subj_steps)
        plot_d["subj_steps"] = plot_d["subj_steps"] + subj_steps
        plot_d["model_steps"] = plot_d["model_steps"] + model_steps[:n]
        plot_d["subj"] = plot_d["subj"] + [i]*n
        plot_d["level"] = plot_d["level"] + list(range(n))
        
    #plot grid of rr model on each, then each subj
    td = pd.DataFrame.from_dict(plot_d)
    plot_d = {"subj_steps":[],"model_steps":[],"level":[],"subj":[]}
    g = sns.FacetGrid(td, col="subj", ylim=(0,1000), col_wrap=5)
    g.map_dataframe(sns.lineplot, x="level", y="subj_steps")
    g = g.map(custom2, 'subj_steps')
    #g.map(lambda y, **kw: plt.axhline(y[20:].mean()), 'subj_steps', alpha=0.5)
    g.map_dataframe(sns.lineplot, x="level", y="model_steps", color='green', alpha=0.5) 
    g = g.map(custom1, 'model_steps')
    g.set_xlabels('level')
    g.set_ylabels('no. steps to success')
    #g.map(lambda y, **kw: plt.axhline(y.mean(), color="green"), 'model_steps', alpha=0.5) 
    plt.show()


def compare_all(agents, envs, latest):
    df = get_success_df(agents, envs)
    if latest:
        df_new = pd.read_csv(new_human_data_path + 'success_latest.csv')
    else:
        df_new = pd.read_csv(new_human_data_path + 'success.csv')
    df_new["agent"] = "humans_new"
    df = pd.concat([df, df_new]) 
    new_agent_name = {'base':'Meta-ePOMDP', 'foil':'Heuristic model', 'base_optimal':'implicit_base', 'base_prev':'base_prev', 'hardcoded':'hardcoded', 'human_asymptote':'Human og', 'humans_new':'Human new', 'rand_attention_bias_1': 'attention limit 1',  'forget_action_mapping_rand_attention_bias_1': 'Resource rational meta-ePOMDP', 'attention_bias_1': 'attention limit 1 (posterior)',  'forget_action_mapping_attention_bias_1': 'attention limit 1 (posterior), forget action mapping'}
    df['agent'] = df['agent'].transform(lambda x: new_agent_name.get(x, x))
    new_env_name = {"logic-v0": "Logic", "logic_u-v0": "Logic\ngoal uncertainty", "contingency-v0": "Contingency", "contingency_u-v0": "Contingency\ngoal uncertainty", 'contingency-shuffle-v0':'Switch Mappings', 'contingency_u-shuffle-v0':'Switch Mappings\ngoal uncertainty', 'changeAgent-7-v0':'Switch Embodiments', 'changeAgent_u-7-v0':'Switch Embodiments\ngoal uncertainty', "logicExtended-v0": "Logic (5)", 'contingencyExtended-v0':"Contingency (5)", 'contingencyExtended-shuffle-v0':"Switching Mappings (5)", 'changeAgentExtended-7-v0':'Switching Embodiments (5)'}
    df['env'] = df['env'].transform(lambda x: new_env_name[x])
    agent_order = ["Human og", "Human new", "Resource rational meta-ePOMDP", "Meta-ePOMDP"]
    env_order = [new_env_name.get(e, e) for e in envs]
    ax = sns.barplot(data=df, x='env', y='success',hue='agent', hue_order=agent_order, palette='viridis', edgecolor='black', order = env_order)
    plt.ylabel("Average no. steps to complete level", fontweight="bold", fontsize=15)
    #max_val = max(df.groupby(['env', 'agent'])["success"].mean())
    plt.xticks(fontsize=9)
    plt.xlabel("Game type", fontweight="bold", fontsize=15)
    plt.legend(title="")#, title_fontweight='bold')
    plt.gcf().set_size_inches(17, 6)
    #plt.savefig(plot_dir + "pilot.png")
    plt.show()



  
def get_exp_data(mean=True):
    data = []
    folder_to_env_name = {'logic_game/':"logic", 'contingency_game/':"contingency", 'contingency_game_shuffled_1/':'contingency-shuffle', 'change_agent_game/':'changeAgent-7'}
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
                all_steps = all_steps + [np.mean(subj['data']['steps'][asymptote:])]
            else:
                all_steps = all_steps + subj['data']['steps'][asymptote:]
        env_name = folder_to_env_name[folder]
        for (i,s) in enumerate(all_steps):
            data.append({'env':env_name + '-v0', 'agent':'human_asymptote','seed':i,'success':s})
    df = pd.DataFrame.from_dict(data)
    df.to_csv('exp_data_means.csv')
    return data


def load_data(names):
    all_data = {}
    for expe_name in names:
        with open(model_data_path + expe_name, 'rb') as f:
            temp = pickle.load(f)
        if len(temp.keys())==1:
            temp = temp[list(temp.keys())[0]]
        for env in temp.keys():
            all_data[env] = all_data.get(env, {})
            for agent in temp[env].keys():
                all_data[env][agent] = temp[env][agent]
    return all_data
def get_success_df(agents, envs):  
    all_data = load_data(['uncertainty_exp.pkl', 'exp.pkl'])
    data = []
    temp = []  
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
                    assert(False)
                temp.append(success)
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
    df2 = pd.DataFrame.from_dict(get_exp_data(mean=True))
    #df2 = pd.read_csv('/Users/traceymills/Dropbox (MIT)/cocosci_projects/self/inferself/src/analyses/exp_data_means.csv')
    
    df2["first_exploit"] = None
    df = pd.concat([df,df2], ignore_index = True)
    df = df[df["agent"].isin(agents)]
    df = df[df["env"].isin(envs)]
    return df

  

if __name__ == "__main__": 
    #compare_humans_and_rr_cau()
    envs = ['logic-v0', 'contingency-v0', 'contingency-shuffle-v0', 'changeAgent-7-v0']
    envs += ['logic_u-v0', 'contingency_u-v0', 'contingency_u-shuffle-v0', 'changeAgent_u-7-v0']
    agents = ['human_asymptote', 'forget_action_mapping_rand_attention_bias_1', 'base', 'foil']
    
    #compare_all(agents, envs,True)
    compare_one('change_agent_u')

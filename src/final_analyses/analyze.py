import os
import matplotlib.pyplot as plt
import pickle
import numpy as np 
import seaborn as sns
import pandas as pd
import scipy
import json
from statannotations.Annotator import Annotator


asymptote = 35
model_data_path = '/Users/traceymills/Dropbox (MIT)/cocosci_projects/self/inferself/src/output/'
human_data_path = '/Users/traceymills/Dropbox (MIT)/cocosci_projects/self/data/'
human_data_path_centering = '/Users/traceymills/Dropbox (MIT)/cocosci_projects/self/probabilisticSelf/stats/'
plot_dir = "/Users/traceymills/Dropbox (MIT)/cocosci_projects/self/inferself/src/analyses/plots/"


def get_human_data():
    data = []
    folder_to_env_name = {'logic_game/':"Logic", 'contingency_game/':"Contingency", 'contingency_game_shuffled_1/':'Switching Mappings', 'change_agent_game/':'Switching Embodiments'}
    for folder, env_name in folder_to_env_name.items():
        for fname in os.listdir(human_data_path + folder + 'human/'):
            if fname.startswith('.'):
                continue
            with open(human_data_path + folder + 'human/' + fname) as f:
                subj = json.load(f)
            subj_mean = np.mean(subj['data']['steps'][asymptote:])
            data.append({'env':env_name, 'agent':'Human', 'steps':subj_mean})
    df = pd.DataFrame.from_dict(data)
    return df

def get_human_data_centering():
    env_names = {'contingency_game':"Contingency", 'contingency_game_shuffled_1':'Switching Mappings',}
    df = pd.DataFrame()
    for env in ["contingency_game", "contingency_game_shuffled_1"]:
        env_data = pd.read_csv(human_data_path_centering + 'self_orienting_' + env + '.csv')
        env_data = env_data[(env_data["level"]>=asymptote) & (env_data["level"]<100)]
        #take mean by participant
        env_data = env_data.groupby('participant', as_index=False).mean()
        env_data = env_data[['self_finding_steps', 'steps', 'participant']]
        env_data = env_data.rename(columns={"self_finding_steps": "sf_steps"})
        env_data['env'] = env_names[env]
        env_data['agent'] = 'Human'
        df = pd.concat([df, env_data])
    return df

def get_model_data():
    agent_names = {'base':'Meta-ePOMDP', 'foil':'Heuristic', 'forget_action_mapping_rand_attention_bias_1': 'Resource rational meta-ePOMDP'}
    env_names = {"logic-v0_False": "Logic", "contingency-v0_False": "Contingency", 'contingency-shuffle-v0_False':'Switching Mappings', 'changeAgent-7-v0_False':'Switching Embodiments'}
    with open(model_data_path + 'exp.pkl', 'rb') as f:
        exp_data = pickle.load(f)
    data = []
    for env in exp_data.keys():
        for agent in exp_data[env].keys():
            for seed in exp_data[env][agent].keys():
                if seed =="args":
                    continue
                #get steps until successs
                level_data = exp_data[env][agent][seed]
                ind_success = np.argwhere(np.array(level_data['success'])).flatten()
                steps = ind_success[0]
                #get steps until centered
                if env_names[env] in ["Contingency", "Switching Mappings"] and agent_names[agent] in ['Meta-ePOMDP', 'Resource rational meta-ePOMDP']:
                    true_self = level_data['true_self'][0]
                    for t, char_probas in enumerate(level_data['all_self_probas']):
                        if char_probas[true_self] == max(char_probas):
                            sorted_probas = sorted(char_probas, reverse=True)
                            if sorted_probas[0] > 1.5*sorted_probas[1]:
                                sf_steps = t-1
                                break
                    if t==len(level_data['all_self_probas'])-1:
                        sf_steps = None
                else:
                    sf_steps = None
                data.append({'env':env_names[env], 'agent':agent_names[agent], 'sf_steps':sf_steps, 'steps':steps})
                
    df = pd.DataFrame.from_dict(data)
    return df


def plot_performance(df, save=False, annotate=False):
    agent_order = ['Human', 'Resource rational meta-ePOMDP', 'Meta-ePOMDP', 'Heuristic']
    env_order = ['Logic', 'Contingency', 'Switching Mappings', 'Switching Embodiments']
    ax = sns.barplot(data=df, x='env', y='steps',hue='agent', hue_order=agent_order, palette='viridis', edgecolor='black', order = env_order, alpha=0.9)
    plt.ylabel("Average no. steps to complete level", fontweight="bold", fontsize=15)
    plt.xticks(fontsize=12)
    plt.xlabel("Game type", fontweight="bold", fontsize=15)
    if annotate:
        pairs = []
        pvalues = []
        to_compare = [('Humans', 'Meta-ePOMDP'), ('Humans', 'Resource rational meta-ePOMDP'), ('Humans', 'Heuristic model')]
        for env in env_order:
            for pair in to_compare:
                pairs.append([(env, pair[0]), (env, pair[1])])
                val = scipy.stats.ttest_ind(df[(df['agent']==pair[0]) & (df['env']==env)]["steps"], df[(df['agent']==pair[1]) & (df['env']==env)]["steps"], equal_var=False)[1]
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
        params = {'data':df, 'x':'env', 'y':'steps', 'hue':'agent', 'hue_order':agent_order, 'order':env_order, 'palette':'viridis', 'edgecolor':'black'}
        ann = Annotator(ax, pairs, **params)
        ann.set_custom_annotations(pvalues)
        ann.annotate()
    plt.legend(title="")
    plt.gcf().set_size_inches(12, 6)
    if save:
        plt.savefig(plot_dir + "performance.png", dpi=1000)
    plt.show()


def stats_performance(df):
    df_human = df[df["agent"]=="Human"]
    df_meta = df[df["agent"]=="Meta-ePOMDP"]
    df_rr = df[df["agent"]=="Resource rational meta-ePOMDP"]
    df_heur = df[df["agent"]=="Heuristic"]
    print("t tests:")
    for env in pd.unique(df["env"]):
        print("game type: " + env)
        human_steps = df_human[df_human["env"]==env]["steps"]
        meta_steps = df_meta[df_meta["env"]==env]["steps"]
        rr_steps = df_rr[df_rr["env"]==env]["steps"]
        heur_steps = df_heur[df_heur["env"]==env]["steps"]
        print("human, meta: " + str(scipy.stats.ttest_ind(meta_steps, human_steps, equal_var=False)))
        print("human, rr: "+ str(scipy.stats.ttest_ind(rr_steps, human_steps, equal_var=False)))
        print("human, heuristic: "+ str(scipy.stats.ttest_ind(heur_steps, human_steps, equal_var=False)))
    mean_human_steps = df_human.groupby(['env'], as_index=False)["steps"].mean()["steps"]
    mean_meta_steps = df_meta.groupby(['env'], as_index=False)["steps"].mean()["steps"]
    mean_rr_steps = df_rr.groupby(['env'], as_index=False)["steps"].mean()["steps"]
    mean_heur_steps = df_heur.groupby(['env'], as_index=False)["steps"].mean()["steps"]
    print("\ncorrelations:")
    print("human, meta: "+ str(scipy.stats.pearsonr(mean_human_steps, mean_meta_steps)))
    print("human, rr: "+ str(scipy.stats.pearsonr(mean_human_steps, mean_rr_steps)))
    print("human, heuristic: "+ str(scipy.stats.pearsonr(mean_human_steps, mean_heur_steps)))


#want to look at num steps before finding correct avatar
def plot_centering(df, save=False):
    df = df[df["agent"] != "Heuristic"]
    env_order = ['Contingency', 'Switching Mappings']
    df['agent_before'] = df['agent'] + ", before selection"
    df['agent_after'] = df['agent'] + ", after selection"
    ax = sns.barplot(data=df, x='env', y='steps', hue='agent_after', order=env_order, palette='viridis', edgecolor='black', alpha=0.5)
    sns.barplot(data=df, x='env', y='sf_steps', hue='agent_before', order=env_order, palette='viridis', edgecolor='black',  alpha=0.9)
    plt.xticks(fontsize=12)
    plt.xlabel('Game type', fontsize=15, fontweight="bold")
    plt.ylabel('Average no. steps', fontsize=15, fontweight="bold")
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3,0,4,1,5,2]
    plt.legend(handles=[handles[i] for i in order], labels=[labels[i] for i in order], title="", loc='upper left')
    plt.gcf().set_size_inches(12, 6)
    if save:
        plt.savefig(plot_dir + "centering.png", dpi=1000)
    plt.show()
    
    
def stats_centering(df, hum_prev_df):
    print("original vs new human performance data:")
    for env in ["Contingency", "Switching Mappings"]:
        print("game type: " + env)
        steps_new = df[(df["env"] == env) & (df["agent"] == "Human")]["steps"]
        steps_prev = hum_prev_df[(hum_prev_df["env"] == env) & (hum_prev_df["agent"] == "Human")]["steps"]
        print("og mean: " + str(np.mean(steps_prev)))
        print("new mean: " + str(np.mean(steps_new)))
        print(scipy.stats.ttest_ind(steps_new, steps_prev))

    print("\nhuman vs model centering data:")
    for env in ["Contingency", "Switching Mappings"]:
        print("game type: " + env)
        sf_steps_human = df[(df["env"] == env) & (df["agent"] == "Human")]["sf_steps"]
        sf_steps_rr = df[(df["env"] == env) & (df["agent"] == 'Resource rational meta-ePOMDP')]["sf_steps"]
        sf_steps_human = [x for x in sf_steps_human if not np.isnan(x)]
        sf_steps_rr = [x for x in sf_steps_rr if not np.isnan(x)]
        print("human mean: " + str(np.mean(sf_steps_human)))
        print(np.var(sf_steps_human))
        print("rr mean: " + str(np.mean(sf_steps_rr)))
        print(np.var(sf_steps_rr))
        print(scipy.stats.ttest_ind(sf_steps_human, sf_steps_rr, equal_var=True))


if __name__ == "__main__": 
    with open(model_data_path + 'exp.pkl', 'rb') as f:
        exp_data = pickle.load(f)
    df_human = get_human_data()
    df_model = get_model_data()
    df = pd.concat([df_human, df_model], ignore_index=True)
    #plot_performance(df)
    stats_performance(df)
    df_human_centering = get_human_data_centering()
    df_centering = pd.concat([df_human_centering, df_model], ignore_index=True)
    #plot_centering(df_centering)
    stats_centering(df_centering, df_human)

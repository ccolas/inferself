import os
import matplotlib.pyplot as plt
import pickle
import numpy as np 
import seaborn as sns
import pandas as pd
import scipy
import json
from statannotations.Annotator import Annotator
#import rpy2.robjects as robjects
#from rpy2.robjects import r, pandas2ri
#from rpy2.robjects.packages import importr
#pandas2ri.activate()


asymptote = 35
model_data_path = '/Users/traceymills/Dropbox (MIT)/cocosci_projects/self/inferself/src/output/'
human_data_path = '/Users/traceymills/Dropbox (MIT)/cocosci_projects/self/data/'
human_data_path_centering = '/Users/traceymills/Dropbox (MIT)/cocosci_projects/self/probabilisticSelf/stats/'
plot_dir = "/Users/traceymills/Dropbox (MIT)/cocosci_projects/self/inferself/src/analyses/plots/"


def get_human_data():
    data = []
    folder_to_env_name = {'logic_game/':"Logic", 'contingency_game/':"Contingency", 'contingency_game_shuffled_1/':'Switching Mappings', 'change_agent_game/':'Switching Embodiments (7)'}
    for folder, env_name in folder_to_env_name.items():
        i=0
        for fname in os.listdir(human_data_path + folder + 'human/'):
            if fname.startswith('.'):
                continue
            with open(human_data_path + folder + 'human/' + fname) as f:
                subj = json.load(f)
            for (level, level_steps) in enumerate(subj['data']['steps'][asymptote:]):
                data.append({'env':env_name, 'agent':'Humans', 'steps':level_steps, 'seed':i, 'level':level})
            i=i+1
    df = pd.DataFrame.from_dict(data)
    print(df.head())
    df = df.groupby(['agent', 'env', 'seed'], as_index=False)['steps'].mean()
    print(df.head())
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
        env_data['agent'] = 'Humans'
        df = pd.concat([df, env_data])
    return df

def get_model_data(path):
    mode=1
    agent_names = {'base':'Meta-ePOMDP', 'foil':'Proximity heuristic', 'forget_action_mapping_rand_attention_bias_1': 'Resource-limited meta-ePOMDP'}
    env_names = {"logic-v0_False": "Logic", "contingency-v0_False": "Contingency", 'contingency-shuffle-v0_False':'Switching Mappings', 'changeAgent-7-v0_False':'Switching Embodiments (7)', 'contingency_noisy-v0_False':'Noisy Contingency', 'contingency_less_chars-v0_False':'Contingency\n(2 characters)', 'contingency_more_chars-v0_False':'Contingency\n(6 characters)', 'contingency_8_chars-v0_False':'Contingency\n(8 characters)', 'changeAgent-10-v0_False':'Switching Embodiments (10)', "logic_u-v0_False": "Logic\ngoal uncertainty", "contingency_u-v0_False": "Contingency\ngoal uncertainty",'contingency_u-shuffle-v0_False':'Switching Mappings\ngoal uncertainty', 'changeAgent_u-7-v0_False':'Switching Embodiments (7)\ngoal uncertainty', 'changeAgent_u-10-v0_False':'Switching Embodiments (10)\ngoal uncertainty'}
    first=True
        
    for path in [model_data_path + 'test_new_rr.pkl', model_data_path + 'test.pkl', model_data_path + 'test8.pkl', model_data_path + 'test_new_rr_last.pkl', model_data_path + 'new_games.pkl']:
        exp_data = {}
        with open(path, 'rb') as f:
            temp = pickle.load(f)
        if path != (model_data_path + 'exp'):
            temp = temp[list(temp.keys())[0]]
        for env in temp.keys():
            if path not in [model_data_path + 'test.pkl', model_data_path + 'test_new_rr.pkl']:
                if env == 'contingency_noisy-v0_False':
                    continue
            exp_data[env] = temp[env]
    
        #add 8 chars
        #with open(model_data_path + 'test8.pkl', 'rb') as f:
        #    temp = pickle.load(f)
        #temp = temp[list(temp.keys())[0]]
        #exp_data['contingency_8_chars-v0_False'] = temp['contingency_8_chars-v0_False']
        #replace noisy w correct prior
        #with open(model_data_path + 'test.pkl', 'rb') as f:
        #    temp = pickle.load(f)
        #temp = temp[list(temp.keys())[0]]
        #exp_data['contingency_noisy-v0_False'] = temp['contingency_noisy-v0_False']
        data = []
        for env in exp_data.keys():
            for agent in exp_data[env].keys():
                for seed in exp_data[env][agent].keys():
                    if seed =="args":
                        continue
                    if mode==1:
                        for level in exp_data[env][agent][seed].keys():
                            #get steps until successs
                            level_data = exp_data[env][agent][seed][level]
                            ind_success = np.argwhere(np.array(level_data['success'])).flatten()
                            
                            if len(ind_success)==0:
                                steps = 10000
                            else:
                                steps = ind_success[0]
                            #get steps until centered
                            if env_names[env] in ["Contingency", "Switching Mappings"] and agent_names.get(agent, agent) in ['Meta-ePOMDP', 'Resource-limited meta-ePOMDP']:
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
                            data.append({'env':env_names[env], 'agent':agent_names.get(agent, agent), 'sf_steps':sf_steps, 'steps':steps, 'seed': seed, 'level':level})
                    else:
                        #get steps until successs
                        level=""
                        level_data = exp_data[env][agent][seed]
                        ind_success = np.argwhere(np.array(level_data['success'])).flatten()
                        
                        if len(ind_success)==0:
                            steps = 10000
                        else:
                            steps = ind_success[0]
                        #get steps until centered
                        if env_names[env] in ["Contingency", "Switching Mappings"] and agent_names[agent] in ['Meta-ePOMDP', 'Resource-limited meta-ePOMDP']:
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
                        data.append({'env':env_names[env], 'agent':agent_names[agent], 'sf_steps':sf_steps, 'steps':steps, 'seed': seed, 'level':level})
                        
                        
            df = pd.DataFrame.from_dict(data)
            if first:
                all_df = df
                first=False
            else:
                all_df = pd.concat([all_df, df])
    return all_df


def plot_performance(df, save=False, annotate=False):
    agent_order = ['Humans', 'Resource-limited meta-ePOMDP', 'Meta-ePOMDP']#, 'Proximity heuristic']
    agent_order = ['Humans', 'Resource-limited meta-ePOMDP', 'forget_action_mapping_rand_attention_bias_1,10', 'forget_action_mapping_rand_attention_bias_1,25', 'forget_action_mapping_rand_attention_bias_1,50', 'forget_action_mapping_rand_attention_bias_1,75', 'Meta-ePOMDP']#, 'Proximity heuristic']
    #env_order = ['Logic', 'Contingency', 'Switching Mappings', 'Switching Embodiments (7)', 'Switching Embodiments (10)', 'Logic\ngoal uncertainty', 'Contingency\ngoal uncertainty', 'Switching Mappings\ngoal uncertainty', 'Switching Embodiments (7)\ngoal uncertainty', 'Switching Embodiments (10)\ngoal uncertainty']
    #env_order = ['Switching Mappings', 'Switching Embodiments (7)', 'Switching Embodiments (10)', 'Switching Mappings\ngoal uncertainty', 'Switching Embodiments (7)\ngoal uncertainty', 'Switching Embodiments (10)\ngoal uncertainty']
    env_order = ['Logic', 'Contingency', 'Switching Mappings', 'Switching Embodiments (7)', 'Switching Embodiments (10)', 'Contingency\ngoal uncertainty', 'Switching Mappings\ngoal uncertainty', 'Noisy Contingency', 'Contingency\n(2 characters)', 'Contingency\n(6 characters)', 'Contingency\n(8 characters)']
    #within each agent type, within each seed, average across levels
    #df = df.groupby(['agent', 'env', 'seed'], as_index=False)['steps'].mean()
    #ax = sns.boxplot(data=df, x='env', y='steps',hue='agent', hue_order=agent_order, palette='viridis', order = env_order, showfliers=False)
    plt.figure(figsize=(18, 6))
    ax = sns.barplot(data=df, x='env', y='steps',hue='agent', hue_order=agent_order, palette='viridis', edgecolor='black', order = env_order, dodge=2.0, alpha=0.9, errorbar='sd')#ci=95)
    plt.ylabel("Average no. steps to complete level", fontweight="bold", fontsize=15)
    plt.xticks(fontsize=7)
    plt.xlabel("Game type", fontweight="bold", fontsize=15)
    if annotate:
        pairs = []
        pvalues = []
        to_compare = [('Humans', 'Meta-ePOMDP'), ('Humans', 'Resource-limited meta-ePOMDP'), ('Humans', 'Proximity heuristic')]
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
    #plt.gcf().set_size_inches(12, 6)
    #plt.figure(figsize=(16, 6))
    #if save:
    plt.savefig(plot_dir + "temp3.png", dpi=1000)
    plt.show()




def stats_performance(df):
    BayesFactor = importr('BayesFactor')
    
    df_human = df[df["agent"]=="Humans"]
    df_meta = df[df["agent"]=="Meta-ePOMDP"]
    df_rr = df[df["agent"]=="Resource-limited meta-ePOMDP"]
    df_heur = df[df["agent"]=="Proximity heuristic"]
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
    print("\nbayes factor tests:")
    for env in pd.unique(df["env"]):
        print("game type: " + env)        
        robjects.globalenv["human"] = df_human[df_human["env"]==env]
        robjects.globalenv["meta"] = df_meta[df_meta["env"]==env]
        robjects.globalenv["rr"] = df_rr[df_rr["env"]==env]
        robjects.globalenv["heur"] = df_heur[df_heur["env"]==env]
        print("human, meta: ")
        r('print(ttestBF(x=human$steps, y=meta$steps))')
        print("human, rr: ")
        r('print(ttestBF(x=human$steps, y=rr$steps))')
        print("human, heuristic: ")
        r('print(ttestBF(x=human$steps, y=heur$steps))')
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
    df = df[df["agent"] != "Proximity heuristic"]
    env_order = ['Contingency', 'Switching Mappings']
    df['agent_before'] = df['agent'] + ", before selection"
    df['agent_after'] = df['agent'] + ", after selection"
    ax = sns.barplot(data=df, x='env', y='steps', hue='agent_after', order=env_order, palette='viridis', edgecolor='black', alpha=0.5, ci=95)
    sns.barplot(data=df, x='env', y='sf_steps', hue='agent_before', order=env_order, palette='viridis', edgecolor='black',  alpha=0.9, ci=95)
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
        steps_new = df[(df["env"] == env) & (df["agent"] == "Humans")]["steps"]
        steps_prev = hum_prev_df[(hum_prev_df["env"] == env) & (hum_prev_df["agent"] == "Humans")]["steps"]
        print("og mean: " + str(np.mean(steps_prev)))
        print("new mean: " + str(np.mean(steps_new)))
        print(scipy.stats.ttest_ind(steps_new, steps_prev))

    print("\nhuman vs model centering data:")
    for env in ["Contingency", "Switching Mappings"]:
        print("game type: " + env)
        sf_steps_human = df[(df["env"] == env) & (df["agent"] == "Humans")]["sf_steps"]
        sf_steps_rr = df[(df["env"] == env) & (df["agent"] == 'Resource-limited meta-ePOMDP')]["sf_steps"]
        sf_steps_human = [x for x in sf_steps_human if not np.isnan(x)]
        sf_steps_rr = [x for x in sf_steps_rr if not np.isnan(x)]
        print("human mean: " + str(np.mean(sf_steps_human)))
        print(np.var(sf_steps_human))
        print("rr mean: " + str(np.mean(sf_steps_rr)))
        print(np.var(sf_steps_rr))
        print(scipy.stats.ttest_ind(sf_steps_human, sf_steps_rr, equal_var=True))


def plot_temp(df):
    print(df.head())
    print(pd.unique(df["agent"]))
    df = df[df["agent"]=="Proximity heuristic"]
    g = sns.FacetGrid(df, col="env", col_wrap=3, height=4)
    g.map(plt.hist, "steps")
    #sns.histplot(data=df, x='env', y='steps')
    """
    agent_order = ['Humans', 'Resource-limited meta-ePOMDP', 'Meta-ePOMDP', 'Proximity heuristic']
    env_order = ['Logic', 'Contingency', 'Switching Mappings', 'Switching Embodiments (7)', 'Switching Embodiments (10)']
    ax = sns.barplot(data=df, x='env', y='steps',hue='agent', hue_order=agent_order, palette='viridis', edgecolor='black', order = env_order, alpha=0.9, ci=95)
    plt.ylabel("Average no. steps to complete level", fontweight="bold", fontsize=15)
    plt.xticks(fontsize=12)
    plt.xlabel("Game type", fontweight="bold", fontsize=15)
   
    plt.legend(title="")
    plt.gcf().set_size_inches(12, 6)
    if save:
        plt.savefig(plot_dir + "performance.png", dpi=1000)
    """
    plt.show()

if __name__ == "__main__": 
    #with open(model_data_path + 'exp.pkl', 'rb') as f:
    #    exp_data = pickle.load(f)
    #with open(model_data_path + 'new_heur_exp.pkl', 'rb') as f:
    #    exp_data = pickle.load(f)
    #path = model_data_path + 'test8.pkl'
    path = model_data_path + 'test_new_rr.pkl'
    #path = model_data_path + 'exp.pkl'
    df_human = get_human_data()
    df_model = get_model_data(path)
    df = pd.concat([df_human, df_model], ignore_index=True)
    #plot_temp(df)
    plot_performance(df, save=False)
    #stats_performance(df)
    #df_human_centering = get_human_data_centering()
    #df_centering = pd.concat([df_human_centering, df_model], ignore_index=True)
    #plot_centering(df_centering, save=False)
    #stats_centering(df_centering, df_human)

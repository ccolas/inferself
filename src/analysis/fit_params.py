import matplotlib.pyplot as plt
import pickle
import numpy as np 
import seaborn as sns
import pandas as pd
import scipy
import json
import ast

first_level = 16
last_level = 40
n_seeds = 20

env_names = {"logic-v0_False": "Logic", "contingency-v0_False": "Contingency", 'contingency-shuffle-v0_False':'Switching Mappings', 'changeAgent-7-v0_False':'Switching Embodiments (7)', 'contingency_noisy-v0_False':'Noisy Contingency', 'contingency_less_chars-v0_False':'Contingency\n(2 characters)', 'contingency_more_chars-v0_False':'Contingency\n(6 characters)', 'contingency_8_chars-v0_False':'Contingency\n(8 characters)', 'changeAgent-10-v0_False':'Switching Embodiments (10)', "contingency_u-v0_False": "Contingency\ngoal uncertainty",'contingency_u-shuffle-v0_False':'Switching Mappings\ngoal uncertainty'}
all_games = list(env_names.values())


def init_model_data():
    with open('../model_output/grid_runs.pkl', 'rb') as f:
        runs = pickle.load(f)
    all_arg_tups = list(runs['Switching Mappings'].keys())
    runs2 = {}
    for env in runs.keys():
        runs2[env] = {}
        for arg_tup in all_arg_tups:
            if arg_tup in runs[env].keys():
                runs2[env][str(arg_tup)] = runs[env][arg_tup]
            else:
                if 'Switching Mappings' in env:
                    print(env)
                    print(arg_tup)
                # now get a dif arg tup in keys with same first val
                swaps = [k for k in list(runs[env].keys()) if k[0]==arg_tup[0]]
                if len(swaps)!=0: 
                    runs2[env][str(arg_tup)] = runs[env][swaps[0]]
    for env in runs2.keys():
        todo = []
        for att in [.05 * i for i in range(21)]:
            for ff in [.05 * j for j in range(21)]:
                att = round(att,3)
                ff = round(ff, 3)
                if str((att, ff)) not in runs2[env].keys():
                    todo.append((att,ff))
        print(todo)  
    with open('../model_output/grid_runs.json', 'w') as f:
        json.dump(runs2, f)

def load_model_data():
    with open('../model_output/grid_runs.json') as f:
        runs_temp = json.load(f)
    runs = {}
    for env in runs_temp.keys():
        runs[env] = {}
        for tup_str in runs_temp[env].keys():
            runs[env][ast.literal_eval(tup_str)] = runs_temp[env][tup_str]
    return runs

def load_human_data():
    with open('../../human_data/study2/data.json') as f:
        trials = json.load(f)
    env_dict = {"logic":"Logic", "contingency":"Contingency",  "contingency_u":"Contingency\ngoal uncertainty", "shuffle_keys":'Switching Mappings', 'shuffle_keys_u':'Switching Mappings\ngoal uncertainty', "change_agent":'Switching Embodiments (7)',"change_agent_10":'Switching Embodiments (10)', "contingency_noisy":"Noisy Contingency", "contingency_2":"Contingency\n(2 characters)", "contingency_6":"Contingency\n(6 characters)", "contingency_8":"Contingency\n(8 characters)"}
    data = []
    max_steps=150   
    for gt in env_dict.keys():
        subjects = [t for t in trials if t["game_type"]==gt]
        for (i,subj) in enumerate(subjects):
            next_idx = False
            for (level, steps) in enumerate(subj["game_data"]["steps"]):
                if next_idx:
                    next_idx = steps > max_steps
                    continue
                next_idx = steps > max_steps
                data.append({'env':env_dict[gt], 'agent':'Humans','level':level, 'seed':i,'steps':steps, 'subj_id':subj["subject_id"]})
    df = pd.DataFrame.from_dict(data)
    df = df[df["level"]<=last_level]
    df = df[df["level"]>=first_level]
    df = df.groupby(['agent', 'env', 'seed'], as_index=False)['steps'].mean()
    return df

def compute_game_ll(human_data, model_data, gt, arg_tup):
    # for this game only, ll of human data given certain params
    human_data = human_data[human_data["env"]==gt]["steps"] #list of len 20 (n_participants)
    model_data = model_data[gt][arg_tup]
    # now lets draw human data from this distrib
    ll = scipy.stats.norm(model_data['mean'], model_data['sd']).logpdf(human_data).sum()
    return ll

# get ll of human data under these args
def compute_total_ll(human_data, model_data, arg_tup):
    ll = 0
    for gt in all_games:
        ll += compute_game_ll(human_data, model_data, gt, arg_tup)
    return ll

def get_all_param_combos(model_data):
    env_data = model_data['Switching Mappings']
    arg_tups = list(env_data.keys())
    return arg_tups

def get_best_params(human_train, model_data):
    param_scores = {}
    for arg_tup in get_all_param_combos(model_data):
        param_scores[arg_tup] = compute_total_ll(human_train, model_data, arg_tup)
    # return params w max score
    return sorted(param_scores.items(),key=lambda x: x[1], reverse=True)[0]

def get_best_params_game(gt, human_train, model_data):
    param_scores = {}
    for arg_tup in get_all_param_combos(model_data): #second val in param only matters for mapping
        param_scores[arg_tup] = compute_game_ll(human_train, model_data, gt, arg_tup)
    return sorted(param_scores.items(), key=lambda x: x[1], reverse=True)[0]

# split within each game type
def split_human_data(human_data):
    # select without replacement
    def random_half(group):
        return group.sample(frac=0.5)
    # apply the random_half function to each group within the env column
    df1 = human_data.groupby('env', group_keys=False).apply(random_half)
    # the other half is in df2
    df2 = human_data.drop(df1.index)
    assert(len(df1)==len(df2))
    return df1, df2

# one crossval run where we get the best params and their ll for a particular train/test split
def crossval_run(human_data, model_data):
    human_train, human_test = split_human_data(human_data)
    (fit_params, ll_train) = get_best_params(human_train, model_data)
    ll_test_rr = compute_total_ll(human_test, model_data, fit_params)
    ll_test_opt = compute_total_ll(human_test, model_data, (1,0))
    return (fit_params, ll_test_rr, ll_test_opt)

# one crossval run for a particular game type only
def crossval_run_game(human_data, model_data, gt):
    human_train, human_test = split_human_data(human_data)
    (fit_params, ll_train) = get_best_params_game(gt, human_train, model_data)
    ll_test_rr = compute_game_ll(human_test, model_data, gt, fit_params)
    ll_test_opt = compute_game_ll(human_test, model_data, gt, (0.05,0.4))
    return (fit_params, ll_test_rr, ll_test_opt)

def run_crossval_by_game():
    hd = load_human_data()
    md = load_model_data()
    param_list = []
    dif_list = []
    ll_rr_list = []
    ll_opt_list = []
    gt_list = []
    for gt in all_games:
        if gt!='Switching Mappings':
            continue
        tp_param_list = []
        tp_diff = []
        print(gt)
        for run in range(10):
            (fit_params, ll_test_rr, ll_test_opt) = crossval_run_game(hd, md, gt)
            print("ll for top on test:", ll_test_rr)
            print("ll for global on test:", ll_test_opt)
            param_list.append(fit_params)
            tp_param_list.append(fit_params)
            dif_list.append(ll_test_rr-ll_test_opt)
            tp_diff.append(ll_test_rr-ll_test_opt) #fit vs global, pos means fit is better
            ll_rr_list.append(ll_test_rr)
            ll_opt_list.append(ll_test_opt)
            gt_list.append(gt)
        counts = dict()
        for i in tp_param_list:
            counts[i] = counts.get(i, 0) + 1
        print(sorted(counts.items(), key=lambda x: x[1]))
        print(np.mean(tp_diff))
        print(scipy.stats.bootstrap((tp_diff,), np.mean))
    df = pd.DataFrame({'dif':dif_list, 'll_rr':ll_rr_list, 'll_opt':ll_opt_list, 'params':param_list, 'gt':gt_list})
    print(pd.unique(df["gt"]))
    ax = sns.pointplot(data=df, x='dif', y='gt', errorbar=('ci', 95), capsize=0.1, palette='viridis', size=0.1)
    plt.legend()
    plt.show()
    
    
def run_crossval_all_games():
    hd = load_human_data()
    md = load_model_data()
    param_list = []
    dif_list = []
    ll_rr_list = []
    ll_opt_list = []
    for run in range(1000):
        (fit_params, ll_test_rr, ll_test_opt) = crossval_run(hd, md)
        param_list.append(fit_params)
        dif_list.append(ll_test_rr-ll_test_opt)
        ll_rr_list.append(ll_test_rr)
        ll_opt_list.append(ll_test_opt)
    counts = dict()
    for i in param_list:
        counts[i] = counts.get(i, 0) + 1
    print(counts)
    print(sorted(counts.items(), key=lambda x: x[1]))
    df = pd.DataFrame({'dif':dif_list, 'height':[1]*len(dif_list), 'll_rr':ll_rr_list, 'll_opt':ll_opt_list, 'params':param_list})
    df.to_csv('ll_dif.csv')
    df = pd.read_csv('ll_dif.csv')
    differences = df['dif']
    sns.histplot(data=df, y='dif')
    plt.show()
    # Plot the data with confidence intervals
    ax = sns.pointplot(data=df, x='dif', errorbar=('ci', 95), capsize=0.1, palette='viridis')
    ax.set_xlim(15200,15700)
    ax.set_yticks([])
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    ax.set_xlabel("LL_resource-limited - LL_optimal", fontweight="bold")
    # Show the plot
    plt.tight_layout()
    plt.gcf().set_size_inches(4, 2.5)
    plt.show()
    print(np.mean(differences))
    print(scipy.stats.bootstrap((differences,), np.mean))                
                


def get_param_plot(model_data):   
    def filter_fn(row):
        tup = eval(row['params'])
        return tup[0]<0.46 and tup[1]>0.14 and tup[1]<0.71
    try:
        df = pd.read_csv('param_scores.csv')
    except:
        human_data = load_human_data()
        param_dicts = []
        
        for x in range(1000):
            human_train, human_test = split_human_data(human_data)
            param_scores = {}
            for arg_tup in get_all_param_combos(model_data):
                param_scores[arg_tup] = compute_total_ll(human_train, model_data, arg_tup)
            param_dicts.append(param_scores)
        df = pd.DataFrame([(key, val, run) for run, d in enumerate(param_dicts) for key, val in d.items()],
                            columns=['params', 'LL', 'run'])
        df.to_csv('param_scores.csv', index=False)
        df = pd.read_csv('param_scores.csv')

    df = df[df.apply(filter_fn, axis=1)]

    ordered_tups = sorted(pd.unique(df['params']))
    ax = sns.barplot(x='params', y='LL', order=ordered_tups, data=df)
    for item in ax.get_xticklabels():
        item.set_rotation(90)
    df["att"] = df.apply(lambda row: eval(row['params'])[0], axis=1)
    df["ff"] = df.apply(lambda row: eval(row['params'])[1], axis=1)
    df.drop(columns=['params'])
    df = df.groupby(['ff', 'att'], as_index=False)['LL'].mean()
    # Create a pivot table
    pivot_df = df.pivot(index='att', columns='ff', values='LL')
    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.0f', cbar_kws={'label': 'Log likelihood of human solving times'})
    
    square_mask = np.zeros_like(pivot_df, dtype=bool)
    square_mask[1, 5] = True 
    for i in range(np.size(square_mask, axis=0)):
        for j in range(np.size(square_mask, axis=1)):
            if square_mask[i, j]:
                rect = plt.Rectangle((j, i), 1, 1, linewidth=2, edgecolor='black', facecolor='none')
                plt.gca().add_patch(rect)
    
    plt.xlabel('$\it{ff}$', fontweight='bold', fontsize=16)
    plt.ylabel('$\it{P(att)}$', fontweight='bold', fontsize=16)
    plt.savefig('../../figs/param_grid.png', dpi=1000)
    plt.show()


if __name__=="__main__":  
    # best fitting params
    md = load_model_data()
    get_param_plot(md)
    
    
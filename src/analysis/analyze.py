import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import scipy
import statsmodels.api as sm
import os
import json

plot_path = '../../figs/'
data_path = '../model_output/'
human_data_path = '../../human_data/'

og_arg_tup = (0, 0.2)
new_arg_tup = (0.05, 0.4)
og_level_bounds = (35, 100)
new_level_bounds = (16, 40)


def scatter_solving_new():
    """
    Generate scatter plots for solving times between humans and all models
    """
    
    X_plot = np.linspace(-100,100,10)
    Y_plot = X_plot
    
    df_human = load_new_human_solving_data()
    df_model = pd.read_csv(data_path + 'model.csv')
    df = pd.concat([df_human, df_model], ignore_index=True)
    df = df[(df["level"]>=new_level_bounds[0]) & (df["level"]<=new_level_bounds[1])]
    df["steps"] = np.minimum(df["steps"], 151)
    df = df.groupby(['agent', 'env', 'seed'], as_index=False)['steps'].mean()
    df = df.groupby(['agent', 'env'], as_index=False)['steps'].mean()
    df['agent'] = df['agent'].replace('Resource-limited meta-ePOMDP\n'+str(new_arg_tup), 'Resource-limited meta-ePOMDP')
    df['env'] = df['env'].replace('Switching Embodiments (7)', 'Switching Embodiments')
    df['env'] = df['env'].replace('Switching Embodiments (10)', 'Switching Embodiments (every 10)')
    df['env'] = df['env'].replace('Contingency\n(2 characters)', 'Contingency2')
    df['env'] = df['env'].replace('Contingency\n(6 characters)', 'Contingency6')
    df['env'] = df['env'].replace('Contingency\n(8 characters)', 'Contingency8')
    df['env'] = df['env'].replace('Switching Mappings\ngoal uncertainty', 'Switching Mappings + goal uncertainty')
    df['env'] = df['env'].replace('Contingency\ngoal uncertainty', 'Contingency + goal uncertainty')

    agent1 = 'Humans'
    agent2 = 'Resource-limited meta-ePOMDP'
    agent3 = 'Meta-ePOMDP'

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    # Scatter plot for types 1 and 2
    df_agent1 = df[df['agent'] == agent1]
    df_agent2 = df[df['agent'] == agent2]
    df_agent3 = df[df['agent'] == agent3]
    print(scipy.stats.pearsonr(df_agent1["steps"], df_agent2["steps"]))
    print(scipy.stats.pearsonr(df_agent1["steps"], df_agent3["steps"]))
    df_comparison1 = pd.merge(df_agent1, df_agent2, on='env', suffixes=('_'+agent1, '_'+agent2))
    df_comparison2 = pd.merge(df_agent1, df_agent3, on='env', suffixes=('_'+agent1, '_'+agent3))
    
    scatter = sns.regplot(x='steps_'+agent2, y='steps_'+agent1, label="env", data=df_comparison1, ax=axs[0], color=sns.color_palette('viridis', 3)[1])
    scatter.plot(X_plot, Y_plot, color='black', linestyle='dashed', alpha=0.5)
    for line in range(0, df_comparison1.shape[0]):
        e = df_comparison2['env'].iloc[line]
        d = {'Switching Mappings + goal uncertainty':0.9, 'Noisy Contingency':1.5, 'Contingency + goal uncertainty':0.3, 'Contingency2':0.5, 'Contingency6':-0.6, 'Switching Mappings': 1.5, 'Switching Embodiments': 0.5}
        vsh = d.get(e, 0)
        scatter.text(df_comparison1['steps_'+agent2].iloc[line]+0.3, df_comparison1['steps_'+agent1].iloc[line]+vsh, df_comparison1['env'].iloc[line], verticalalignment='top', horizontalalignment='left', fontsize=10, color='black', fontweight='medium')
    r2 = scipy.stats.pearsonr(df_agent1["steps"], df_agent2["steps"])[0]**2
    #scatter.text(68, 3, r'$\mathbf{R^2}$=' + f'{r2:.2f}',  size='large', color='black', fontweight='bold') 
    scatter.text(60, 3, r'$\mathit{R^2}=' + f'{r2:.2f}' + '$', fontsize=16, color='black')
 
    scatter = sns.regplot(x='steps_'+agent3, y='steps_'+agent1, label="env", data=df_comparison2, ax=axs[1], color=sns.color_palette('viridis', 3)[2])
    scatter.plot(X_plot, Y_plot, color='black', linestyle='dashed', alpha=0.5)
    for line in range(0, df_comparison2.shape[0]):
        e = df_comparison2['env'].iloc[line]
        d = {'Switching Mappings + goal uncertainty':0.5, 'Noisy Contingency':1.0, 'Contingency + goal uncertainty':0.5, 'Contingency8':-0.4, 'Contingency6':-0.2, 'Switching Embodiments': 0.5, 'Switching Mappings':1.0}
        vsh = d.get(e, 0)
        scatter.text(df_comparison2['steps_'+agent3].iloc[line]+0.6, df_comparison2['steps_'+agent1].iloc[line]+vsh, df_comparison2['env'].iloc[line], verticalalignment='top', horizontalalignment='left',  fontsize=10, color='black', fontweight='medium')
    r2 = scipy.stats.pearsonr(df_agent1["steps"], df_agent3["steps"])[0]**2
    #scatter.text(68, 3, r'$\mathbf{R^2}$=' + f'{r2:.2f}',  size='large', color='black', fontweight='bold')
    scatter.text(60, 3, r'$\mathit{R^2}=' + f'{r2:.2f}' + '$', fontsize=16, color='black')
    
    # Heuristic
    df_human = load_new_human_solving_data() 
    df_heur = pd.read_csv(data_path + 'heuristic.csv')
    df = pd.concat([df_human, df_heur], ignore_index=True)
    df = df[(df["level"]>=new_level_bounds[0]) & (df["level"]<=new_level_bounds[1])]
    df["steps"] = np.minimum(df["steps"], 151)
    df = df.groupby(['agent', 'env', 'seed'], as_index=False)['steps'].mean()
    df = df.groupby(['agent', 'env'], as_index=False)['steps'].mean()
    df['env'] = df['env'].replace('Contingency\n(2 characters)', 'Contingency2')
    df['env'] = df['env'].replace('Contingency\n(6 characters)', 'Contingency6')
    df['env'] = df['env'].replace('Contingency\n(8 characters)', 'Contingency8')
    df['env'] = df['env'].replace('Switching Mappings\ngoal uncertainty', 'Switching Mappings +\ngoal uncertainty')
    df['env'] = df['env'].replace('Contingency\ngoal uncertainty', 'Contingency +\ngoal uncertainty')
    df['env'] = df['env'].replace('Switching Embodiments (7)', 'Switching Embodiments')
    df['env'] = df['env'].replace('Switching Embodiments (10)', 'Switching Embodiments (every 10)')
    
    df_agent1 = df[df['agent'] == 'Humans']
    df_agent2 = df[df['agent'] == 'Proximity heuristic']
    
    df_comparison3 = pd.merge(df_agent1, df_agent2, on='env', suffixes=('_'+agent1, '_'+agent2))
    scatter = sns.regplot(x='steps_'+agent2, y='steps_'+agent1, label="env", data=df_comparison3, ax=axs[2], color=sns.color_palette('viridis', 15)[14])
    scatter.plot(X_plot, Y_plot, color='black', linestyle='dashed', alpha=0.5)
    for line in range(0, df_comparison1.shape[0]):
        e = df_comparison3['env'].iloc[line]
        d = {'Switching Mappings':1.5, 'Noisy Contingency':1.7, 'Contingency +\ngoal uncertainty':1.6, 'Contingency8':0.5, 'Switching Mappings +\ngoal uncertainty':1}
        vsh = d.get(e, 0)
        hsh = 0.3
        scatter.text(df_comparison3['steps_'+agent2].iloc[line]+hsh, df_comparison3['steps_'+agent1].iloc[line]+vsh, df_comparison3['env'].iloc[line], verticalalignment='top', horizontalalignment='left',  fontsize=10, color='black', fontweight='medium')
    #scatter.set_ylabel("Humans", fontsize=14, fontweight='semibold')
    #scatter.set_title('Solving time by game type', fontsize=16, fontweight='bold')
    r2 = scipy.stats.pearsonr(df_agent1["steps"], df_agent2["steps"])[0]**2
    #scatter.text(68, 3, r'$\mathbf{R^2}$=' + f'{r2:.3f}',  size='large', color='black', fontweight='bold')
    scatter.text(60, 3, r'$\mathit{R^2}=' + f'{r2:.3f}' + '$', fontsize=16, color='black')
    
    axs[0].set_ylim(0,58)
    axs[0].set_xlim(0,90)#90
    axs[1].set_ylim(0,58)
    axs[1].set_xlim(0,90)#90
    axs[2].set_ylim(0,58)
    axs[2].set_xlim(0,90)
    
    for ax in axs:
        ax.tick_params(axis='x', labelsize=16)
    axs[0].tick_params(axis='y', labelsize=16)
    
    axs[0].set_ylabel("Solving time (Humans)", fontsize=22)
    axs[1].set_ylabel("")#Solving time (Humans)", fontsize=14, fontweight='semibold')
    axs[2].set_ylabel("")#Solving time (Humans)", fontsize=14, fontweight='semibold')
    axs[0].set_xlabel("Solving time\n(Resource-limited mePOMDP)", fontsize=22)
    axs[1].set_xlabel("Solving time\n(mePOMDP)", fontsize=22)
    axs[2].set_xlabel("Solving time\n(Proximity heuristic)", fontsize=22)
    
    # axs[1].tick_params(labelleft=True)
    # axs[2].tick_params(labelleft=True)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    plt.savefig(plot_path + 'scatter.png', dpi=1000)
    #plt.show()


def plot_solving_new(include_heuristic=False):
    df_human = load_new_human_solving_data() 
    df_model = pd.read_csv(data_path + 'model.csv')
    df_heur = pd.read_csv(data_path + 'heuristic.csv')
    df = pd.concat([df_human, df_model, df_heur], ignore_index=True)
    df = df[(df["level"]>=new_level_bounds[0]) & (df["level"]<=new_level_bounds[1])]
    df["steps"] = np.minimum(df["steps"], 151)
    
    df = df.groupby(['agent', 'env', 'seed'], as_index=False)['steps'].mean()

    df['agent'] = df['agent'].replace('Resource-limited meta-ePOMDP\n'+str(new_arg_tup), 'Resource-limited meta-ePOMDP')
    df['agent'] = df['agent'].replace('Resource-limited meta-ePOMDP', 'Resource-limited mePOMDP')
    df['agent'] = df['agent'].replace('Meta-ePOMDP', 'mePOMDP')
    df['env'] = df['env'].replace('Switching Embodiments (7)', 'Switching Embodiments')
    df['env'] = df['env'].replace('Switching Embodiments (10)', 'Switching Embodiments\n(every 10)')
    df['env'] = df['env'].replace('Contingency\n(2 characters)', 'Contingency2')
    df['env'] = df['env'].replace('Contingency\n(6 characters)', 'Contingency6')
    df['env'] = df['env'].replace('Contingency\n(8 characters)', 'Contingency8')
    df['env'] = df['env'].replace('Contingency\ngoal uncertainty', 'Contingency +\ngoal uncertainty')
    df['env'] = df['env'].replace('Switching Mappings\ngoal uncertainty', 'Switching Mappings +\ngoal uncertainty')
    
    colors =  sns.color_palette('viridis', 3)
    agent_order = ['Humans', 'Resource-limited mePOMDP', 'mePOMDP']
    env_order = ['Logic', 'Contingency', 'Switching Mappings', 'Switching Embodiments', 'Switching Embodiments\n(every 10)', 'Contingency +\ngoal uncertainty', 'Switching Mappings +\ngoal uncertainty', 'Noisy Contingency', 'Contingency2', 'Contingency6', 'Contingency8']    

    if include_heuristic:
        colors += [sns.color_palette('viridis', 15)[14]]
        agent_order += ['Proximity heuristic']
    
    plt.figure(figsize=(16, 7))
    sns.barplot(data=df, x='env', y='steps', hue='agent', hue_order=agent_order, palette=colors, edgecolor='black', order = env_order, dodge=2.0, alpha=0.9, errorbar=('ci',95))
    plt.ylabel("Solving time", fontweight="bold", fontsize=16)
    plt.xticks(fontsize=10, rotation=20, ha="right")
    plt.xlabel("Game type", fontweight="bold", fontsize=16)
    plt.legend(title="", fontsize=12, loc='upper left')
    plt.savefig(plot_path + "solving_new.png", dpi=100, bbox_inches="tight")


def run_t(l1, l2, paired=False):
    equal_var = (np.std(l1)/np.std(l2)) >= 0.5 and (np.std(l1)/np.std(l2)) <= 2
    if paired:
        res = str(scipy.stats.ttest_rel(l1, l2))
    else:
        res = str(scipy.stats.ttest_ind(l1, l2, equal_var=equal_var))
    
    #cohens d
    n1, n2 = len(l1), len(l2) #size of samples
    s1, s2 = np.var(l1, ddof=1), np.var(l2, ddof=1) #variance of samples
    s = (((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))**0.5 #pooled sd
    u1, u2 = np.mean(l1), np.mean(l2) #means of samples
    d = (u1 - u2) / s #effect size
    return res + ", Cohen's d: " + str(d)


def stats_solving_og():
    df_human = load_og_human_solving_data()
    df_human = df_human[(df_human["level"]>=og_level_bounds[0]) & (df_human["level"]<=og_level_bounds[1])]
    envs = pd.unique(df_human["env"])
    print(envs)
    df_model = pd.read_csv(data_path + 'model.csv')
    df_heur = pd.read_csv(data_path + 'heuristic.csv')
    df = pd.concat([df_human, df_model, df_heur], ignore_index=True)
    df = df.groupby(['agent', 'env', 'seed'], as_index=False)['steps'].mean()
    df = df[df["env"].isin(envs)]
    df['agent'] = df['agent'].replace('Resource-limited meta-ePOMDP\n'+str(og_arg_tup), 'Resource-limited meta-ePOMDP')
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
        print("human, meta: " + str(run_t(meta_steps, human_steps)))
        print("human, rr: "+ str(run_t(rr_steps, human_steps)))
        print("human, heuristic: "+ str(run_t(heur_steps, human_steps)))
    
    """
    BayesFactor = importr('BayesFactor')
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
    """
    mean_human_steps = np.array(df_human.groupby(['env'], as_index=False)["steps"].mean()["steps"])
    mean_meta_steps = np.array(df_meta.groupby(['env'], as_index=False)["steps"].mean()["steps"])
    mean_rr_steps = np.array(df_rr.groupby(['env'], as_index=False)["steps"].mean()["steps"])
    mean_heur_steps = np.array(df_heur.groupby(['env'], as_index=False)["steps"].mean()["steps"])
    print("\ncorrelations:")
    print("human, meta: "+ str(scipy.stats.pearsonr(mean_human_steps, mean_meta_steps)))
    print("human, rr: "+ str(scipy.stats.pearsonr(mean_human_steps, mean_rr_steps)))
    print("human, heuristic: "+ str(scipy.stats.pearsonr(mean_human_steps, mean_heur_steps)))



def stats_solving_new():
    df_human = load_new_human_solving_data()
    df_model = pd.read_csv(data_path + 'model.csv')
    df_heur = pd.read_csv(data_path + 'heuristic.csv')

    df = pd.concat([df_human, df_model, df_heur], ignore_index=True)
    df = df[(df["level"]>=new_level_bounds[0]) & (df["level"]<=new_level_bounds[1])]
    df["steps"] = np.minimum(df["steps"], 151)
    df = df.groupby(['agent', 'env', 'seed'], as_index=False)['steps'].mean()
    
    df['agent'] = df['agent'].replace('Resource-limited meta-ePOMDP\n'+str(new_arg_tup), 'Resource-limited meta-ePOMDP')
    
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
        print(len(human_steps), len(meta_steps), len(rr_steps), len(heur_steps))
        print("human, meta: " + str(run_t(meta_steps, human_steps)))
        print("human, rr: "+ str(run_t(rr_steps, human_steps)))
        print("human, heuristic: "+ str(run_t(heur_steps, human_steps)))
        print("\n")
    """
    BayesFactor = importr('BayesFactor')
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
    """
    
    mean_human_steps = np.array(df_human.groupby(['env'], as_index=False)["steps"].mean()["steps"])
    mean_meta_steps = np.array(df_meta.groupby(['env'], as_index=False)["steps"].mean()["steps"])
    mean_rr_steps = np.array(df_rr.groupby(['env'], as_index=False)["steps"].mean()["steps"])
    mean_heur_steps = np.array(df_heur.groupby(['env'], as_index=False)["steps"].mean()["steps"])
    
    meta_abs_err = abs(mean_meta_steps-mean_human_steps)
    rr_abs_err = abs(mean_rr_steps-mean_human_steps)
    heur_abs_err = abs(mean_heur_steps-mean_human_steps)
    print("\nmean absolute errors")
    print("meta: ", np.mean(meta_abs_err))
    print("rr: ", np.mean(rr_abs_err))
    print("heur: ", np.mean(heur_abs_err))
    
    print("\npaired t-tests over absolute errors")
    print("meta, rr: " + str(run_t(meta_abs_err, rr_abs_err,  paired=True)))
    print("heuristic, meta: " + str(run_t(heur_abs_err, meta_abs_err, paired=True)))
    print("heuristic, rr: " + str(run_t(heur_abs_err, rr_abs_err, paired=True)))
    
    print("\npaired t-tests bt models and humans")
    print("human, meta: " + str(run_t(mean_human_steps, mean_meta_steps, paired=True)))
    print("human, rr: " + str(run_t(mean_human_steps, mean_rr_steps, paired=True)))
    print("human, heuristic: " + str(run_t(mean_human_steps, mean_heur_steps,paired=True)))
    
    print("\ncorrelations:")
    print("human, meta: "+ str(scipy.stats.pearsonr(mean_human_steps, mean_meta_steps)))
    print("human, rr: "+ str(scipy.stats.pearsonr(mean_human_steps, mean_rr_steps)))
    print("human, heuristic: "+ str(scipy.stats.pearsonr(mean_human_steps, mean_heur_steps)))

    #compute variance accounted for when slope = 1 and intercept = 0
    #1-B/A, where A = total variance in the 11 solving times (means of each game)
    #and B (for a given model) is the variance in solving times
    #when you subtract out the model's predicted solving time.
    
    print("\nr^2 with and without slope/intercept fit:")
    def squared_error(ys_orig,ys_line):
        return sum((ys_line - ys_orig) * (ys_line - ys_orig))
    def cod(ys_orig,ys_pred):
        y_mean_line = [ys_orig.mean() for y in ys_orig]
        squared_error_regr = squared_error(ys_orig, ys_pred)
        squared_error_y_mean = squared_error(ys_orig, y_mean_line)
        return 1 - (squared_error_regr/squared_error_y_mean)
    y = mean_human_steps
    x = mean_rr_steps
    print("humans and rr, no fit:", cod(y,x))
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    print("humans and rr, fit:", cod(y,model.fittedvalues))
    x = mean_meta_steps
    print("humans and meta, no fit:", cod(y,x))
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    print("humans and meta, fit:", cod(y,model.fittedvalues))
    x = mean_heur_steps
    print("humans and heuristic, no fit:", cod(y,x))
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    print("humans and heuristic, fit:", cod(y,model.fittedvalues))

    #note difference due to attention constraint in 2 vs 4 chars:
    print("\nDifference between game types: contingency")
    human_steps_2 = df_human[df_human["env"]=='Contingency\n(2 characters)']["steps"]
    human_steps_4 = df_human[df_human["env"]=='Contingency']["steps"]
    human_steps_6 = df_human[df_human["env"]=='Contingency\n(6 characters)']["steps"]
    human_steps_8 = df_human[df_human["env"]=='Contingency\n(8 characters)']["steps"]
    meta_steps_2 = df_meta[df_meta["env"]=='Contingency\n(2 characters)']["steps"]
    meta_steps_4 = df_meta[df_meta["env"]=='Contingency']["steps"]
    meta_steps_6 = df_meta[df_meta["env"]=='Contingency\n(6 characters)']["steps"]
    meta_steps_8 = df_meta[df_meta["env"]=='Contingency\n(8 characters)']["steps"]
    rr_steps_2 = df_rr[df_rr["env"]=='Contingency\n(2 characters)']["steps"]
    rr_steps_4 = df_rr[df_rr["env"]=='Contingency']["steps"]
    rr_steps_6 = df_rr[df_rr["env"]=='Contingency\n(6 characters)']["steps"]
    rr_steps_8 = df_rr[df_rr["env"]=='Contingency\n(8 characters)']["steps"]
    print("2,4:")
    print("human:", run_t(human_steps_2, human_steps_4))
    print("rr:", run_t(rr_steps_2, rr_steps_4))
    print("meta:", run_t(meta_steps_2, meta_steps_4))
    print("4,6:")
    print("human:", run_t(human_steps_4, human_steps_6))
    print("rr:", run_t(rr_steps_4, rr_steps_6))
    print("meta:", run_t(meta_steps_4, meta_steps_6))
    print("6,8:")
    print("human:", run_t(human_steps_6, human_steps_8))
    print("rr:", run_t(rr_steps_6, rr_steps_8))
    print("meta:", run_t(meta_steps_6, meta_steps_8))
    
    #effect of resource constraints in switching mapping game:
    #switching mappings vs contingency + goal uncertainty
    #switching mappings + goal uncertainty vs noisy contingency
    print("\nSwitching mappings vs contingency + gu:")
    human_steps_sm = df_human[df_human["env"]=='Switching Mappings']["steps"]
    human_steps_cu = df_human[df_human["env"]=='Contingency\ngoal uncertainty']["steps"]
    rr_steps_sm = df_rr[df_rr["env"]=='Switching Mappings']["steps"]
    rr_steps_cu = df_rr[df_rr["env"]=='Contingency\ngoal uncertainty']["steps"]
    meta_steps_sm = df_meta[df_meta["env"]=='Switching Mappings']["steps"]
    meta_steps_cu = df_meta[df_meta["env"]=='Contingency\ngoal uncertainty']["steps"]
    print("human:", run_t(human_steps_sm, human_steps_cu))
    print("rr:", run_t(rr_steps_sm, rr_steps_cu))
    print("meta:", run_t(meta_steps_sm, meta_steps_cu))
    
    print("\nSwitching mappings + gu vs noisy contingency:")
    human_steps_sm = df_human[df_human["env"]=='Switching Mappings\ngoal uncertainty']["steps"]
    human_steps_nc = df_human[df_human["env"]=='Noisy Contingency']["steps"]
    human_steps_se = df_human[df_human["env"]=='Switching Embodiments (10)']["steps"]
    meta_steps_sm = df_meta[df_meta["env"]=='Switching Mappings\ngoal uncertainty']["steps"]
    meta_steps_nc = df_meta[df_meta["env"]=='Noisy Contingency']["steps"]
    meta_steps_se = df_meta[df_meta["env"]=='Switching Embodiments (10)']["steps"]
    rr_steps_sm = df_rr[df_rr["env"]=='Switching Mappings\ngoal uncertainty']["steps"]
    rr_steps_nc = df_rr[df_rr["env"]=='Noisy Contingency']["steps"]
    rr_steps_se = df_rr[df_rr["env"]=='Switching Embodiments (10)']["steps"]
    print("human:", run_t(human_steps_sm, human_steps_nc))
    print("rr:", run_t(rr_steps_sm, rr_steps_nc))
    print("meta:", run_t(meta_steps_sm, meta_steps_nc))
    
    print("\nSwitching mappings + gu vs switch emb (10):")
    print("human:", run_t(human_steps_sm, human_steps_se))
    print("rr:", run_t(rr_steps_sm, rr_steps_se))
    print("meta:", run_t(meta_steps_sm, meta_steps_se))


# num steps before finding correct avatar
def plot_centering():
    df_model = pd.read_csv(data_path + 'model.csv')
    df_model = df_model.groupby(['agent', 'env', 'seed'], as_index=False)[['steps', 'sf_steps']].mean()
    df_human_centering = load_og_human_centering_data()
    df_human_centering = df_human_centering[(df_human_centering["level"]>=og_level_bounds[0]) & (df_human_centering["level"]<=og_level_bounds[1])]
    df_human_centering = df_human_centering.groupby(['agent', 'env', 'seed'], as_index=False)[['steps', 'sf_steps']].mean()
    df = pd.concat([df_human_centering, df_model], ignore_index=True)
    
    df['agent'] = df['agent'].replace('Resource-limited meta-ePOMDP\n'+str(og_arg_tup), 'Resource-limited meta-ePOMDP')
    
    df = df[df["agent"] != "Proximity heuristic"]
    df['agent'] = df['agent'].replace('Resource-limited meta-ePOMDP', 'Resource-limited mePOMDP')
    df['agent'] = df['agent'].replace('Meta-ePOMDP', 'mePOMDP')
    agent_order = ['Humans', 'Resource-limited mePOMDP', 'mePOMDP']
    env_order = ['Logic', 'Contingency', 'Switching Mappings']
    df['agent_before'] = df['agent'] + ", before selection"
    df['agent_after'] = df['agent'] + ", after selection"
    ax = sns.barplot(data=df, x='env', y='steps', hue='agent_after', hue_order=[a + ', after selection' for a in agent_order], order=env_order, palette='viridis', edgecolor='black', alpha=0.5, errorbar=('ci', 95))
    sns.barplot(data=df, x='env', y='sf_steps', hue='agent_before', hue_order=[a + ', before selection' for a in agent_order], order=env_order, palette='viridis', edgecolor='black',  alpha=0.9, errorbar=('ci', 95))
    plt.xticks(fontsize=12)
    plt.xlabel('Game type', fontsize=15, fontweight="bold")
    plt.ylabel('Average no. steps', fontsize=15, fontweight="bold")
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3,0,4,1,5,2]
    plt.legend(handles=[handles[i] for i in order], labels=[labels[i] for i in order], title="", loc='upper left')
    plt.gcf().set_size_inches(12, 6)
    plt.savefig(plot_path + "centering.png", dpi=1000)
    plt.show()
    
    
def stats_centering():
    # load in human data
    df_human_centering = load_og_human_centering_data()
    df_human_centering = df_human_centering[(df_human_centering["level"]>=og_level_bounds[0]) & (df_human_centering["level"]<=og_level_bounds[1])]
    df_human_centering = df_human_centering.groupby(['agent', 'env', 'seed'], as_index=False)[['steps', 'sf_steps']].mean()

    # compare solving times to og solving times
    df_human_og = load_og_human_solving_data()
    df_human_og = df_human_og[(df_human_og["level"]>=og_level_bounds[0]) & (df_human_og["level"]<=og_level_bounds[1])]
    df_human_og = df_human_og.groupby(['agent', 'env', 'seed'], as_index=False)[['steps']].mean()

    print("original vs centering human solving times:")
    for env in ["Logic", "Contingency", "Switching Mappings"]:
        print("game type: " + env)
        steps_new = df_human_centering[(df_human_centering["env"] == env) & (df_human_centering["agent"] == "Humans")]["steps"]
        steps_prev = df_human_og[(df_human_og["env"] == env) & (df_human_og["agent"] == "Humans")]["steps"]
        print("og mean: " + str(np.mean(steps_prev)))
        print("new mean: " + str(np.mean(steps_new)))
        print(run_t(steps_new, steps_prev))
    
    # load in model data
    df_model = pd.read_csv(data_path + 'model.csv')
    df_model = df_model.groupby(['agent', 'env', 'seed'], as_index=False)[['steps', 'sf_steps']].mean()
    df = pd.concat([df_human_centering, df_model], ignore_index=True)
    df = pd.concat([df_human_centering, df_model], ignore_index=True)
    df['agent'] = df['agent'].replace('Resource-limited meta-ePOMDP\n'+str(og_arg_tup), 'Resource-limited meta-ePOMDP')
    
    # compare model and human centering
    print("\nhuman vs model centering data:")
    for env in ["Logic", "Contingency", "Switching Mappings"]:
        print("game type: " + env)
        sf_steps_human = df[(df["env"] == env) & (df["agent"] == "Humans")]["sf_steps"]
        sf_steps_rr = df[(df["env"] == env) & (df["agent"] == 'Resource-limited meta-ePOMDP')]["sf_steps"]
        sf_steps_meta = df[(df["env"] == env) & (df["agent"] == 'Meta-ePOMDP')]["sf_steps"]
        sf_steps_human = [x for x in sf_steps_human if not np.isnan(x)]
        sf_steps_rr = [x for x in sf_steps_rr if not np.isnan(x)]
        sf_steps_meta = [x for x in sf_steps_meta if not np.isnan(x)]
        print("human mean: " + str(np.mean(sf_steps_human)))
        print("rr mean: " + str(np.mean(sf_steps_rr)))
        print("meta mean: ", np.mean(sf_steps_meta))
        print("human vs rr: ", run_t(sf_steps_human, sf_steps_rr))
        print("human vs meta: ", run_t(sf_steps_human, sf_steps_meta))
        
def compare_new_and_og_humans():
    df_old = load_og_human_solving_data()
    df_old = df_old[(df_old["level"]>=og_level_bounds[0]) & (df_old["level"]<=og_level_bounds[1])]
    df_new = load_new_human_solving_data()
    df_new = df_new[(df_new["level"]>=new_level_bounds[0]) & (df_new["level"]<=new_level_bounds[1])]
    #df_new["steps"] = np.minimum(df_new["steps"], 151)
    #df_old["steps"] = np.minimum(df_old["steps"], 151)
    df_old = df_old.groupby(['agent', 'env', 'seed'], as_index=False)['steps'].mean()
    df_new = df_new.groupby(['agent', 'env', 'seed'], as_index=False)['steps'].mean()
    print("t tests:")
    for env in pd.unique(df_old["env"]):
        print("game type: " + env)
        old_steps = df_old[df_old["env"]==env]["steps"]
        new_steps = df_new[df_new["env"]==env]["steps"]
        print("old vs new: " + str(run_t(old_steps, new_steps)))
        print("new: ", np.mean(new_steps), np.std(new_steps))
        print("old: ", np.mean(old_steps), np.std(old_steps))
    df_old = df_old.groupby(['env'], as_index=False)['steps'].mean()
    df_new = df_new.groupby(['env'], as_index=False)['steps'].mean()
    df_new = df_new[df_new["env"].isin(['Logic', 'Contingency', 'Switching Mappings', 'Switching Embodiments (7)'])]
    print(scipy.stats.pearsonr(df_new["steps"], df_old["steps"]))
    
def plot_solving_og():
    df_human_old = load_og_human_solving_data()
    df_human_old = df_human_old[(df_human_old["level"]>=og_level_bounds[0]) & (df_human_old["level"]<=og_level_bounds[1])]
    df_model = pd.read_csv(data_path + 'model.csv')
    df_heuristic = pd.read_csv(data_path + 'heuristic.csv')
    df = pd.concat([df_human_old, df_model, df_heuristic], ignore_index=True)
    df = df.groupby(['agent', 'env', 'seed'], as_index=False)['steps'].mean()
    m1 = np.mean(df[(df['agent']=='Proximity heuristic') & (df['env']=='Contingency')]['steps'])
    m2 = np.mean(df[(df['agent']=='Proximity heuristic') & (df['env']=='Switching Mappings')]['steps'])
    df.loc[(df['agent']=='Proximity heuristic') & (df['env']=='Contingency'), 'steps'] = 86#np.mean(df[(df['agent']=='Proximity heuristic') & (df['env']=='Contingency')]['steps'])
    df.loc[(df['agent']=='Proximity heuristic') & (df['env']=='Switching Mappings'), 'steps'] = 86#hednp.mean(df[(df['agent']=='Proximity heuristic') & (df['env']=='Switching Mappings')]['steps'])
    df['agent'] = df['agent'].replace('Resource-limited meta-ePOMDP\n'+str(og_arg_tup), 'Resource-limited meta-ePOMDP')
    df['agent'] = df['agent'].replace('Resource-limited meta-ePOMDP', 'Resource-limited mePOMDP')
    df['agent'] = df['agent'].replace('Meta-ePOMDP', 'mePOMDP')
    df['env'] = df['env'].replace('Switching Embodiments (7)', 'Switching Embodiments')
    colors =  sns.color_palette('viridis', 3)
    colors += [sns.color_palette('viridis', 15)[14]]
    agent_order = ['Humans', 'Resource-limited mePOMDP', 'mePOMDP', 'Proximity heuristic']
    env_order = ['Logic', 'Contingency', 'Switching Mappings', 'Switching Embodiments']
    
    #df.loc[df['agent'] != 'Humans', 'steps'] = 0
    
    plt.figure(figsize=(16, 6))
    ax = sns.barplot(data=df, x='env', y='steps',hue='agent', hue_order=agent_order, palette=colors, edgecolor='black', order = env_order, dodge=2.0, alpha=0.9, errorbar=('ci',95))
    ax.axhline(y=86, xmin=0.425, xmax=0.477, color=sns.color_palette('viridis', 15)[14], linestyle='dashed', linewidth=2, alpha=0.9)
    ax.axhline(y=86, xmin=0.678, xmax=0.725, color=colors[-1], linestyle='dashed', linewidth=2, alpha=0.9)
    
    #add an arrow pointing up to the target number
    ax.annotate('',
            xy=(1.3,95),
            xytext=(1.3,86),
            ha='center', va='center',
            arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->'))
    ax.text(1.3, 96, str(int(round(m1,0))), ha='center', va='center', fontsize=10, color='black')
    ax.annotate('',
            xy=(2.3,95),
            xytext=(2.3,86),
            ha='center', va='center',
            arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->'))
    ax.text(2.3, 96, str(int(round(m2,0))), ha='center', va='center', fontsize=10, color='black')

    ax.spines['top'].set_visible(False)
    plt.ylabel("Solving time", fontweight="bold", fontsize=16)
    plt.xticks(fontsize=12)
    plt.ylim(0,99)
    plt.xlabel("Game type", fontweight="bold", fontsize=16)
    plt.legend(title="", fontsize=12)
    plt.gcf().set_size_inches(12, 6)
    plt.savefig(plot_path + "solving_og.png", dpi=1000)
    plt.show()
    
def load_new_human_solving_data():
    with open(human_data_path + 'study2/data.json') as f:
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
    return df

def load_og_human_solving_data():
    path = human_data_path + 'study1/data/solving/'
    data = []
    folder_to_env_name = {'logic_game/':"Logic", 'contingency_game/':"Contingency", 'contingency_game_shuffled_1/':'Switching Mappings', 'change_agent_game/':'Switching Embodiments (7)'}
    for folder, env_name in folder_to_env_name.items():
        i=0
        for fname in os.listdir(path + folder + 'human/'):
            if fname.startswith('.'):
                continue
            with open(path + folder + 'human/' + fname) as f:
                subj = json.load(f)
            for (level, level_steps) in enumerate(subj['data']['steps']):
                data.append({'env':env_name, 'agent':'Humans', 'steps':level_steps, 'seed':i, 'level':level})
            i=i+1
    df = pd.DataFrame.from_dict(data)
    return df
    
def load_og_human_centering_data():
    path = human_data_path + 'study1/data/centering/'
    env_names = {'contingency_game':"Contingency", 'contingency_game_shuffled_1':'Switching Mappings', "logic_game":"Logic"}
    df = pd.DataFrame()
    for env in ["contingency_game", "contingency_game_shuffled_1", "logic_game"]:
        env_data = pd.read_csv(path + 'self_orienting_' + env + '.csv')
        if env=="logic_game":
            env_data = env_data.rename(columns={'human_self_finding_steps':'self_finding_steps','human_total_steps':'steps'})
            env_data = env_data[env_data["level"]<100]
        env_data = env_data[['self_finding_steps', 'steps', 'participant', 'level']]
        env_data = env_data.rename(columns={"self_finding_steps": "sf_steps", 'participant':'seed'})
        env_data['env'] = env_names[env]
        env_data['agent'] = 'Humans'
        df = pd.concat([df, env_data])
    return df

if __name__ == "__main__": 

    plot_centering()
    plot_solving_og()
    plot_solving_new(include_heuristic=True)
    scatter_solving_new()
    
    print("/\\"*50)
    compare_new_and_og_humans()
    print("/\\"*50)
    stats_solving_new()
    print("/\\"*50)
    stats_solving_og()
    print("/\\"*50)
    stats_centering()
    print("/\\"*50)
    
    

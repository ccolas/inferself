import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={"lines.linewidth":0.7})

#want to know prob of true agent, with change points marked
#plot multiple lines and the mean
#TODO: add indication of change points
def track_agent_prob_all():
    df = pd.read_csv('output/out.csv')
    #df = df[df['env']=='contingency-shuffle-noisy-v0']

    df['true_self_proba'] = df.apply(lambda x: eval(x['all_self_probas'])[x['true_self']], axis=1)
    g = sns.FacetGrid(df, col="agent_type", height=0.5, row="env", ylim=(-.01,1.01), margin_titles=True, hue="run")
    g.map(sns.pointplot, "tpt", "true_self_proba", markers=".")
    g.set(xlim=(0, 100))
    g.set_axis_labels("tpt", "prob of\ntrue self", fontsize=10)
    g.set_titles(col_template="{col_name}",row_template="{row_name}", fontweight='bold')
    g.tight_layout()
    g.figure.subplots_adjust(wspace=.05, hspace=.1)
    plt.show()


def track_agent_prob_by_env():
    df = pd.read_csv('output/out.csv')
    df['true_self_proba'] = df.apply(lambda x: eval(x['all_self_probas'])[x['true_self']], axis=1)

    for env in df["env"].unique():
        df_temp = df[df['env']==env]
        changepoints = get_changepoints(df_temp)
        g = sns.FacetGrid(df_temp, height=0.5, row="agent_type", ylim=(-.01,1.01), margin_titles=True, hue="run")
        g.map(sns.pointplot, "tpt", "true_self_proba", markers=".")
        g.set(xlim=(0, 100))

        #add lines at change points
        for i, ax in enumerate(g.axes.flat):
            for cp in changepoints[i]:
                ax.axvline(x=cp, color='b', linestyle=':')
                
        #g.set_axis_labels("tpt", "prob of\ntrue self", fontsize=10)
        #g.set_titles(col_template="{col_name}",row_template="{row_name}", fontweight='bold')
        g.tight_layout()
        g.figure.subplots_adjust(wspace=.05, hspace=.1)
        plt.show()

def get_changepoints(df):
    changepoints = []
    for agent in df["agent_type"].unique():
        df_temp = df[df['agent_type']==agent]
        df_temp = df_temp[df_temp['agent_change']]
        changepoints.append(df_temp["tpt"].unique())
    return changepoints

track_agent_prob_by_env()
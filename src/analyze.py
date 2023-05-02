import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={"lines.linewidth":0.7})

#want to know prob of true agent, with change points marked
#plot multiple lines and the mean
#TODO: add indication of change points
def track_agent_prob():
    df = pd.read_csv('output/out.csv')
    df['true_self_proba'] = df.apply(lambda x: eval(x['all_self_probas'])[x['true_self']], axis=1)
    g = sns.FacetGrid(df, col="agent_type", row="env",  height=1, ylim=(-.01,1.01), margin_titles=True)#hue="run",
    g.map(sns.pointplot, "tpt", "true_self_proba", markers=".")
    g.set_axis_labels("tpt", "prob of\ntrue self", fontsize=10)
    g.set_titles(col_template="{col_name}",row_template="{row_name}", fontweight='bold')
    g.tight_layout()
    g.figure.subplots_adjust(wspace=.05, hspace=.1)
    plt.show()

track_agent_prob()
import os

import matplotlib.pyplot as plt
import pickle
import numpy as np
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
          '#7f7f7f', '#bcbd22', '#17becf']

save_dir = "/mnt/e85692fd-9cbc-4a8d-b5c5-9252bd9a34fd/Research/Scratch/inferself/data/experiments/"
plot_dir = "/mnt/e85692fd-9cbc-4a8d-b5c5-9252bd9a34fd/Research/Scratch/inferself/data/plots/plot_all/"

# compile all data
all_data = dict()
for expe_name in os.listdir(save_dir):
    if 'all_data' not in expe_name:
        data_path = save_dir + expe_name
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        all_data.update(data)

with open(save_dir + 'all_data.pkl', 'wb') as f:
    pickle.dump(all_data, f)

def plot_one(data, path):
    fig, ax = plt.subplots(figsize=(15, 7))

    # plot success
    if 'False' in path:
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

for expe_name in all_data.keys():
    for env in all_data[expe_name].keys():
        for agent in all_data[expe_name][env].keys():
            plot_path = plot_dir + expe_name + '/' + env + '/' + agent + '/'
            print(plot_path)
            os.makedirs(plot_path, exist_ok=True)
            for seed in all_data[expe_name][env][agent].keys():
                data = all_data[expe_name][env][agent][seed]
                filename = f'{env}__{agent}__{seed}.png'
                plot_one(data, path=plot_path + filename)

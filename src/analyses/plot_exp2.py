import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

save_dir = "../../data/experiments/"
plot_dir = "../../data/plots/"

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

max_steps = 60
n_seeds = 50
def plot_histograms(data, n_detected, alg_names, save_path, ylabel=None, alpha=0.3, n_bins=10):
    if n_detected:
        fig, axs = plt.subplots(2, figsize=(10, 7))
    else:
        fig, axs = plt.subplots(figsize=(10, 7))
        axs = [axs]
    min = np.min([np.min(d) for d in data])
    max = np.max([np.max(d) for d in data])
    bins = np.arange(min, max, (max - min) / n_bins)
    for i in range(len(data)):
        out = axs[0].hist(data[i], bins=bins, color=COLORS[i], alpha=alpha, label=alg_names[i], density=True)
        axs[0].axvline(np.mean(data[i]), ymin=0, ymax=out[1].max(), color=COLORS[i])
        if n_detected and 'success' not in ylabel:
            axs[1].bar([i * 1.2], [n_detected[i]], width=1, color=COLORS[i])
            axs[1].legend(alg_names)
            axs[1].set_ylabel(f'fraction {ylabel} detected')
            axs[1].set_xticks([])
    if ylabel:
        axs[0].set_title(f'distribution of # steps before {ylabel}')
        axs[0].set_xlabel('steps')
    axs[0].legend()
    plt.savefig(save_path)
    plt.close('all')

def plot_best_theory(data, names, save_path):
    for d, name in zip(data, names):
        changes = np.argwhere(np.array(d['changes'])).flatten()
        fig, axs = plt.subplots(figsize=(10, 7))
        for c in changes:
            plt.axvline(c, ymin=0, ymax=1, color='k')
        plt.plot(np.array(d['true_theory_probas']), alpha=0.4)
        plt.plot(np.array(d['true_theory_probas']).mean(axis=0), linewidth=3, label='mean')
        plt.legend()
        plt.xlabel('steps')
        plt.ylabel('prob best theory')
        plt.savefig(save_path + f'{name}_proba_best.pkl')
        plt.close('all')
    changes = np.argwhere(np.array(data[0]['changes'])).flatten()
    fig, axs = plt.subplots(figsize=(10, 7))
    for c in changes:
        plt.axvline(c, ymin=0, ymax=1, color='k')
    for i, d, name in zip(range(len(data)), data, names):
        plt.plot(np.array(d['true_theory_probas']), alpha=0.4, color=COLORS[i])
        plt.plot(np.array(d['true_theory_probas']).mean(axis=0), linewidth=3, label=name,  color=COLORS[i])
    plt.legend()
    plt.xlabel('steps')
    plt.ylabel('prob best theory')
    plt.savefig(save_path + f'all_proba_best.pkl')
    plt.close('all')

def get_data_from_agent_and_env(data):
    results = dict(success=[], infer05=[], infer07=[], infer09=[],
                   frac_infer05=[], frac_infer07=[], frac_infer09=[],
                   detected_success=0, detected_infer05=0, detected_infer07=0, detected_infer09=0,
                   true_theory_probas=[])
    n_detected = []
    count = 0
    for i in data.keys():
        n_detected.append(0)
        changes = [0] + list(np.argwhere(np.array(data[str(i)]['agent_change'])).flatten()) + [max_steps]
        results['changes'] = np.array(data[str(i)]['agent_change'])
        for start, end in zip(changes[:-1], changes[1:]):
            count  += 1
            to_track = np.array(data[str(i)]['success'])[start: end]
            indexes = np.argwhere(to_track).flatten()
            if indexes.size > 0:
                results['success'].append(indexes[0] + 1)
                results['detected_success'] += 1
            to_track = (np.array(data[str(i)]['true_theory_probas']) > 0.5)[start: end]
            indexes = np.argwhere(to_track).flatten()
            if indexes.size > 0:
                results['infer05'].append(indexes[0] + 1)
                results['frac_infer05'].append(to_track.sum() / len(to_track))
                results['detected_infer05'] += 1
            to_track = (np.array(data[str(i)]['true_theory_probas']) > 0.7)[start: end]
            indexes = np.argwhere(to_track).flatten()
            if indexes.size > 0:
                results['infer07'].append(indexes[0] + 1)
                results['frac_infer07'].append(to_track.sum() / len(to_track))
                results['detected_infer07'] += 1
            to_track = (np.array(data[str(i)]['true_theory_probas']) > 0.9)[start: end]
            indexes = np.argwhere(to_track).flatten()
            if indexes.size > 0:
                results['infer09'].append(indexes[0] + 1)
                results['frac_infer09'].append(to_track.sum() / len(to_track))
                results['detected_infer09'] += 1
        results['true_theory_probas'].append(np.array(data[str(i)]['true_theory_probas']))
    for key in ['detected_infer05', 'detected_infer07', 'detected_infer09']:
        results[key] /= count
    return results

# no mapping inference, logic and contingency + noise
envs = ['logic-noisy-v0', 'contingency-noisy-v0']
agent = 'no_infer_mapping'
expe_name = 'no_infer_mapping'
exploit_only = True
env_data = []
for env in envs:
    env_ = env + '_' + str(exploit_only)
    env_agent_data = all_data[expe_name][env_][agent]
    data = get_data_from_agent_and_env(env_agent_data)
    env_data.append(data)

save_path = plot_dir + 'no_change_no_infer_mapping' + '/'
os.makedirs(save_path, exist_ok=True)
for to_plot in ['infer05', 'infer07', 'infer09', 'frac_infer05', 'frac_infer07', 'frac_infer09']:
    data = [d[to_plot] for d in env_data]
    if 'frac' in to_plot:
        n_detected = None
    else:
        n_detected = [d['detected_' + to_plot] for d in env_data]
    plot_histograms(data, n_detected, envs, save_path + to_plot + '.png', ylabel=to_plot, alpha=0.4, n_bins=10)
    plot_best_theory(env_data, envs, save_path)

# no mapping inference, logic and contingency + noise
agent = 'base'
envs = ['logic-shuffle-noisy-v0', 'contingency-shuffle-noisy-v0']
expe_name = 'base'
exploit_only = True
env_data = []
for env in envs:
    env_ = env + '_' + str(exploit_only)
    env_agent_data = all_data[expe_name][env_][agent]
    data = get_data_from_agent_and_env(env_agent_data)
    env_data.append(data)

save_path = plot_dir + 'no_change_infer_mapping' + '/'
os.makedirs(save_path, exist_ok=True)
for to_plot in ['infer05', 'infer07', 'infer09', 'frac_infer05', 'frac_infer07', 'frac_infer09']:
    data = [d[to_plot] for d in env_data]
    if 'frac' in to_plot:
        n_detected = None
    else:
        n_detected = [d['detected_' + to_plot] for d in env_data]
    plot_histograms(data, n_detected, envs, save_path + to_plot + '.png', ylabel=to_plot, alpha=0.4, n_bins=10)


# with agent change
agents = ['base', 'explicit_resetter', 'current_focused_forgetter', 'hierarchical']
envs = ['changeAgent-noisy-v0', 'changeAgent-shuffle-noisy-10-v0']
expe_name = 'with_agent_change'
exploit_only = True
for env in envs:
    agent_data = []
    env_ = env + '_' + str(exploit_only)
    for agent in agents:
        env_agent_data = all_data[expe_name][env_][agent]
        data = get_data_from_agent_and_env(env_agent_data)
        agent_data.append(data)
    save_path = plot_dir + expe_name + '/' + env + '/'
    os.makedirs(save_path, exist_ok=True)
    for to_plot in ['infer05', 'infer07', 'infer09', 'frac_infer05', 'frac_infer07', 'frac_infer09']:
        data = [d[to_plot] for d in agent_data]
        if 'frac' in to_plot:
            n_detected = None
        else:
            n_detected = [d['detected_' + to_plot] for d in agent_data]
        plot_histograms(data, n_detected, agents, save_path + to_plot + '.png', ylabel=to_plot, alpha=0.3, n_bins=10)
stop = 1


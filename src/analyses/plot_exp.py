import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib
font = {'size'   : 15}
matplotlib.rc('font', **font)

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
    if max - min < n_bins:
        delta = 1
        bins = np.arange(min, max + 2)
        xs = np.arange(min, max + 1)
    else:
        delta = (max - min) // n_bins + 1
        bins = np.arange(min, max + 2 * delta, delta)
        xs = np.arange(min, max + delta , delta)
    for i in range(len(data)):
        ys, _ = np.histogram(data[i], bins=bins - delta / 2)
        ys = ys.astype(float)
        ys /= ys.sum()
        ysmax = np.max(ys)
        axs[0].bar(xs, ys, color=COLORS[i], alpha=alpha, label=alg_names[i], width=delta)
        axs[0].axvline(np.mean(data[i]), ymin=0, ymax=ysmax, color=COLORS[i])
        axs[0].set_xticks(bins)
        if n_detected:
            axs[1].bar([i * 1.2], [n_detected[i]], width=1, color=COLORS[i])
            axs[1].legend(alg_names)
            axs[1].set_ylabel(f'fraction {ylabel} detected')
            axs[1].set_xticks([])
    if ylabel:
        if 'frac' in ylabel:
            axs[0].set_title(f'distribution of % steps {ylabel} detected')
            axs[0].set_xlabel('% steps')
        else:
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
            plt.axvline(c, ymin=0, ymax=1, linestyle='--', color='b')
        a = np.zeros((len(d['true_theory_probas']), max_steps))
        a.fill(np.nan)
        for i, dd in enumerate(d['true_theory_probas']):
            a[i][:len(dd)] = dd
        plt.plot(np.arange(1, a.shape[1] + 1), a.T, alpha=0.25, color='k')
        plt.plot(np.arange(1, a.shape[1] + 1), np.nanmean(a, axis=0), linewidth=5, color='r', label='mean')
        plt.legend()
        plt.xlabel('steps')
        plt.ylabel('prob best theory')
        plt.savefig(save_path + f'{name}_proba_best.png')
        plt.close('all')
    changes = np.argwhere(np.array(data[0]['changes'])).flatten()
    fig, axs = plt.subplots(figsize=(10, 7))
    for c in changes:
        plt.axvline(c, ymin=0, ymax=1, linestyle='--', color='b')
    # for i, d, name in zip(range(len(data)), data, names):
    #     plt.plot(np.arange(1, a.shape[1] + 1), np.array(d['true_theory_probas']).T, alpha=0.25, color=COLORS[i])
    for i, d, name in zip(range(len(data)), data, names):
        a = np.zeros((len(d['true_theory_probas']), max_steps))
        a.fill(np.nan)
        for j, dd in enumerate(d['true_theory_probas']):
            a[j][:len(dd)] = dd
        plt.plot(np.arange(1, max_steps + 1), np.nanmean(a, axis=0), linewidth=3, label=name,  color=COLORS[i])
    plt.legend()
    plt.xlabel('steps')
    plt.ylabel('prob best theory')
    plt.savefig(save_path + f'all_proba_best.png')
    plt.close('all')

def get_data_from_agent_and_env(data):
    results = dict(success=[], infer05=[], infer07=[], infer09=[],
                   frac_infer05=[], frac_infer07=[], frac_infer09=[],
                   detected_success=0, detected_infer05=0, detected_infer07=0, detected_infer09=0,
                   true_theory_probas=[])
    n_detected = []
    count = 0
    run_count = 0
    for i in data.keys():
        n_detected.append(0)
        changes = [0] + list(np.argwhere(np.array(data[str(i)]['agent_change'])).flatten()) + [max_steps]
        results['changes'] = np.array(data[str(i)]['agent_change'])
        run_count += 1
        to_track = np.array(data[str(i)]['success'])
        indexes = np.argwhere(to_track).flatten()
        if indexes.size > 0:
            results['success'].append(indexes[0] + 1)
            results['detected_success'] += 1
        for start, end in zip(changes[:-1], changes[1:]):
            count += 1
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
    results['detected_success'] /= run_count
    return results

# # no mapping inference, logic and contingency + noise
envs = ['logic-v0', 'contingency-v0', 'contingency-shuffle-v0',
        'changeAgent-v0', 'changeAgent-shuffle-v0',
        'changeAgent-7-v0', 'changeAgent-10-v0', 'changeAgent-15-v0',
        'changeAgent-markovian-7-v0', 'changeAgent-markovian-10-v0', 'changeAgent-markovian-15-v0']
agents = ['base', 'hierarchical']
expe_name = 'all_expe'
explore_only = False
for env in envs:
    agent_data = []
    env_ = env + '_' + str(explore_only)
    if 'shuffle' in env:
        stop = 1
    for agent in agents:
        env_agent_data = all_data[expe_name][env_][agent]
        data = get_data_from_agent_and_env(env_agent_data)
        agent_data.append(data)
    save_path = plot_dir + expe_name + '/' + env + '/'
    os.makedirs(save_path, exist_ok=True)
    to_plotss = ['success']
    for to_plot in to_plotss:
        data = [d[to_plot] for d in agent_data]
        if 'frac' in to_plot:
            n_detected = None
        else:
            n_detected = [d['detected_' + to_plot] for d in agent_data]
        plot_histograms(data, n_detected, agents, save_path + to_plot + '.png', ylabel=to_plot, alpha=0.3, n_bins=10)
    plot_best_theory(agent_data, agents, save_path)


import matplotlib.pyplot as plt
import pickle
import numpy as np

save_dir = "/mnt/e85692fd-9cbc-4a8d-b5c5-9252bd9a34fd/Research/Scratch/inferself/data/experiments/"
plot_dir = "/mnt/e85692fd-9cbc-4a8d-b5c5-9252bd9a34fd/Research/Scratch/inferself/data/plots/"
expe_name = 'without_agent_change'
data_path = save_dir + expe_name + '.pkl'

with open(data_path, 'rb') as f:
    all_data = pickle.load(f)

# How slower is it when we need to infer action mapping? if we have a prior?
agents = ['base', 'no_infer_mapping', 'biased_action_mapping']
envs = ['logic-v0', 'logic-noisy-v0', 'contingency-v0', 'contingency-noisy-v0']
plot_name = 'inference_action_mapping'
for env in envs:
    all_n_steps_before_success = []
    all_n_steps_before_inference = []
    for agent in agents:
        n_steps_before_success = []
        n_steps_before_inference = []
        for i in all_data[env][agent].keys():
            agent_found = np.array(all_data[env][agent][str(i)]['agent_found'])
            indexes = np.argwhere(agent_found).flatten()
            if indexes.size > 0:
                n_steps_before_inference.append(indexes[0])
            else:
                n_steps_before_inference.append(np.nan)
            success = np.array(all_data[env][agent][str(i)]['success'])
            indexes = np.argwhere(success).flatten()
            if indexes.size > 0:
                n_steps_before_success.append(indexes[0])
            else:
                n_steps_before_success.append(np.nan)
        all_n_steps_before_success.append(n_steps_before_success)
        all_n_steps_before_inference.append(n_steps_before_inference)
    fig, ax = plt.subplots(figsize=(10, 7))
    data = np.array(all_n_steps_before_inference).T
    plt.boxplot(data, labels=agents)
    plt.ylabel('steps before self identification')
    plt.gca().set_ylim(bottom=0)
    plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env}_inference.png')
    fig, ax = plt.subplots(figsize=(10, 7))
    data = np.array(all_n_steps_before_success).T
    plt.boxplot(data, labels=agents)
    plt.ylabel('steps before game solved')
    plt.gca().set_ylim(bottom=0)
    plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env}_success.png')
    plt.close('all')

# How slower is it when there is noise vs no noise (noise inference in both cases)?
agent = 'base'
envs = [('logic-v0', 'logic-noisy-v0'), ('contingency-v0', 'contingency-noisy-v0')]
plot_name = 'noisy_vs_non_noisy'
for env_pair in envs:
    all_n_steps_before_success = []
    all_n_steps_before_inference = []
    for env in env_pair:
        n_steps_before_success = []
        n_steps_before_inference = []
        for i in all_data[env][agent].keys():
            agent_found = np.array(all_data[env][agent][str(i)]['agent_found'])
            indexes = np.argwhere(agent_found).flatten()
            if indexes.size > 0:
                n_steps_before_inference.append(indexes[0])
            else:
                n_steps_before_inference.append(np.nan)
            success = np.array(all_data[env][agent][str(i)]['success'])
            indexes = np.argwhere(success).flatten()
            if indexes.size > 0:
                n_steps_before_success.append(indexes[0])
            else:
                n_steps_before_success.append(np.nan)
        all_n_steps_before_success.append(n_steps_before_success)
        all_n_steps_before_inference.append(n_steps_before_inference)
    fig, ax = plt.subplots(figsize=(10, 7))
    data = np.array(all_n_steps_before_inference).T
    plt.boxplot(data, labels=env_pair)
    plt.ylabel('steps before self identification')
    plt.gca().set_ylim(bottom=0)
    plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env_pair[0].split("-")[0]}_inference.png')
    fig, ax = plt.subplots(figsize=(10, 7))
    data = np.array(all_n_steps_before_success).T
    plt.boxplot(data, labels=env_pair)
    plt.ylabel('steps before game solved')
    plt.gca().set_ylim(bottom=0)
    plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env_pair[0].split("-")[0]}_success.png')
    plt.close('all')

# how good is the estimation of the noise for the true agent theory?
envs = ['logic-shuffle-noisy-v0', 'contingency-shuffle-noisy-v0']
agent = 'base'
plot_name = 'noise_tracking'
for env in envs:
    all_noises = []
    for i in  all_data[env][agent].keys():
        noise = np.array(all_data[env][agent][str(i)]['true_theory_noise_mean'])
        all_noises.append(noise)
    max_len = np.max([len(data) for data in all_noises])
    data = np.zeros((len(all_noises), max_len))
    data.fill(np.nan)
    for i_noise, noises in enumerate(all_noises):
        data[i_noise, :len(noises)] = noises
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.plot(np.nanmean(data, axis=0))
    plt.gca().set_ylim(bottom=0)
    plt.ylabel('noise estimate true theory')
    plt.fill_between(np.arange(max_len), np.nanmean(data, axis=0) - np.nanstd(data, axis=0), np.nanmean(data, axis=0) + np.nanstd(data, axis=0), alpha=0.2)
    plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env}.png')
    plt.close('all')

# how fast is it (solving + inferring) if we just do random actions for the exploration step?
agents = ['base', 'random_explo']
envs = ['logic-shuffle-noisy-v0', 'contingency-shuffle-noisy-v0']
plot_name = 'optimal_vs_random_explo'
for env in envs:
    all_n_steps_before_success = []
    all_n_steps_before_inference = []
    for agent in agents:
        n_steps_before_success = []
        n_steps_before_inference = []
        for i in all_data[env][agent].keys():
            agent_found = np.array(all_data[env][agent][str(i)]['agent_found'])
            indexes = np.argwhere(agent_found).flatten()
            if indexes.size > 0:
                n_steps_before_inference.append(indexes[0])
            else:
                n_steps_before_inference.append(np.nan)
            success = np.array(all_data[env][agent][str(i)]['success'])
            indexes = np.argwhere(success).flatten()
            if indexes.size > 0:
                n_steps_before_success.append(indexes[0])
            else:
                n_steps_before_success.append(np.nan)
        all_n_steps_before_success.append(n_steps_before_success)
        all_n_steps_before_inference.append(n_steps_before_inference)
    fig, ax = plt.subplots(figsize=(10, 7))
    data = all_n_steps_before_inference
    for i_d in range(len(data)):
        data[i_d] = [dd for dd in data[i_d] if not np.isnan(dd)]
    plt.boxplot(data, labels=agents)
    plt.ylabel('steps before self identification')
    plt.gca().set_ylim(bottom=0)
    plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env}_inference.png')
    fig, ax = plt.subplots(figsize=(10, 7))
    data = np.array(all_n_steps_before_success).T
    plt.boxplot(data, labels=agents)
    plt.ylabel('steps before game solved')
    plt.gca().set_ylim(bottom=0)
    plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env}_success.png')
    plt.close('all')

##################################################################3

expe_name = 'with_agent_change'
data_path = save_dir + expe_name + '.pkl'

with open(data_path, 'rb') as f:
    all_data = pickle.load(f)

# how fast is it (solving + inferring) if we just do random actions for the exploration step?
agents = ['base', 'random_explo']
envs = ['changeAgent-shuffle-noisy-v0']
plot_name = 'optimal_vs_random_explo'
for env in envs:
    all_n_steps_before_success = []
    all_n_steps_before_inference = []
    for agent in agents:
        n_steps_before_success = []
        n_steps_before_inference = []
        for i in all_data[env][agent].keys():
            agent_found = np.array(all_data[env][agent][str(i)]['agent_found'])
            indexes = np.argwhere(agent_found).flatten()
            if indexes.size > 0:
                n_steps_before_inference.append(indexes[0])
            else:
                n_steps_before_inference.append(np.nan)
            success = np.array(all_data[env][agent][str(i)]['success'])
            indexes = np.argwhere(success).flatten()
            if indexes.size > 0:
                n_steps_before_success.append(indexes[0])
            else:
                n_steps_before_success.append(np.nan)
        all_n_steps_before_success.append(n_steps_before_success)
        all_n_steps_before_inference.append(n_steps_before_inference)
    fig, ax = plt.subplots(figsize=(10, 7))
    data = all_n_steps_before_inference
    for i_d in range(len(data)):
        data[i_d] = [dd for dd in data[i_d] if not np.isnan(dd)]
    plt.boxplot(data, labels=agents)
    plt.ylabel('steps before self identification')
    plt.gca().set_ylim(bottom=0)
    plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env}_inference.png')
    fig, ax = plt.subplots(figsize=(10, 7))
    data = np.array(all_n_steps_before_success).T
    plt.boxplot(data, labels=agents)
    plt.ylabel('steps before game solved')
    plt.gca().set_ylim(bottom=0)
    plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env}_success.png')
    plt.close('all')


# best heirarchical model?
agents = ['base', 'explicit_resetter', 'current_focused', 'current_focused_forgetter', 'hierarchical']
envs = ['changeAgent-v0', 'changeAgent-noisy-v0', 'changeAgent-shuffle-noisy-v0']
plot_name = 'best_hierarchical'
for env in envs:
    all_n_steps_before_success = []
    all_n_steps_before_inference = []
    for agent in agents:
        n_steps_before_success = []
        n_steps_before_inference = []
        for i in all_data[env][agent].keys():
            agent_found = np.array(all_data[env][agent][str(i)]['agent_found'])
            indexes = np.argwhere(agent_found).flatten()
            if indexes.size > 0:
                n_steps_before_inference.append(indexes[0])
            else:
                n_steps_before_inference.append(np.nan)
            success = np.array(all_data[env][agent][str(i)]['success'])
            indexes = np.argwhere(success).flatten()
            if indexes.size > 0:
                n_steps_before_success.append(indexes[0])
            else:
                n_steps_before_success.append(np.nan)
        all_n_steps_before_success.append(n_steps_before_success)
        all_n_steps_before_inference.append(n_steps_before_inference)
    fig, ax = plt.subplots(figsize=(10, 7))
    data = all_n_steps_before_inference
    for i_d in range(len(data)):
        data[i_d] = [dd for dd in data[i_d] if not np.isnan(dd)]
    plt.boxplot(data, labels=agents)
    plt.ylabel('steps before self identification')
    plt.gca().set_ylim(bottom=0)
    plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env}_inference.png')
    fig, ax = plt.subplots(figsize=(10, 7))
    data = np.array(all_n_steps_before_success).T
    plt.boxplot(data, labels=agents)
    plt.ylabel('steps before game solved')
    plt.gca().set_ylim(bottom=0)
    plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env}_success.png')
    plt.close('all')


##################################################################3

expe_name = 'one_switch'
data_path = save_dir + expe_name + '.pkl'

with open(data_path, 'rb') as f:
    all_data = pickle.load(f)

# Post switch recovery: test speed recovery after a switch a t=30
agents = ['base', 'explicit_resetter', 'current_focused', 'current_focused_forgetter', 'hierarchical']
env = 'changeAgent-shuffle-noisy-oneswitch-v0'
plot_name = 'post_switch_recovery'
all_n_steps_before_success = []
all_n_steps_before_inference = []
for agent in agents:
    n_steps_before_success = []
    n_steps_before_inference = []
    for i in all_data[env][agent].keys():
        agent_found = np.array(all_data[env][agent][str(i)]['agent_found'])
        indexes = np.array([i for i in np.argwhere(agent_found).flatten() if i > 29])
        if indexes.size > 0:
            n_steps_before_inference.append(indexes[0])
        else:
            n_steps_before_inference.append(np.nan)
        success = np.array(all_data[env][agent][str(i)]['success'])
        indexes = np.array([i for i in np.argwhere(agent_found).flatten() if i > 29])
        if indexes.size > 0:
            n_steps_before_success.append(indexes[0])
        else:
            n_steps_before_success.append(np.nan)
    all_n_steps_before_success.append(n_steps_before_success)
    all_n_steps_before_inference.append(n_steps_before_inference)
fig, ax = plt.subplots(figsize=(10, 7))
data = all_n_steps_before_inference
for i_d in range(len(data)):
    data[i_d] = [dd for dd in data[i_d] if not np.isnan(dd)]
plt.boxplot(data, labels=agents)
plt.ylabel('steps before self identification')
plt.gca().set_ylim(bottom=0)
plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env}_inference.png')
fig, ax = plt.subplots(figsize=(10, 7))
data = np.array(all_n_steps_before_success).T
plt.boxplot(data, labels=agents)
plt.ylabel('steps before game solved')
plt.gca().set_ylim(bottom=0)
plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env}_success.png')
plt.close('all')

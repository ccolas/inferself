import matplotlib.pyplot as plt
import pickle
import numpy as np

save_dir = "../data/experiments/"
plot_dir = "../data/plots/"
expe_name = 'without_agent_change'
data_path = save_dir + expe_name + '.pkl'

with open(data_path, 'rb') as f:
    all_data = pickle.load(f)

# # How slower is it when we need to infer action mapping? if we have a prior?
# agents = ['base', 'no_infer_mapping', 'biased_action_mapping']
# envs = ['logic-v0', 'logic-noisy-v0', 'contingency-v0', 'contingency-noisy-v0']
# plot_name = 'inference_action_mapping'
# for env in envs:
#     for explore_only in [True, False]:
#         env_ = env + '_' + str(explore_only)
#         all_n_steps_before = []
#         for agent in agents:
#             n_steps_before = []
#             for i in all_data[env_][agent].keys():
#                 if 'True' in env_:
#                     to_track = np.array(all_data[env_][agent][str(i)]['true_theory_probas']) > 0.7
#                 else:
#                     to_track = np.array(all_data[env_][agent][str(i)]['success'])
#                 indexes = np.argwhere(to_track).flatten()
#                 if indexes.size > 0:
#                     n_steps_before.append(indexes[0])
#                 else:
#                     n_steps_before.append(np.nan)
#             all_n_steps_before.append(n_steps_before)
#         fig, ax = plt.subplots(figsize=(10, 7))
#         data = all_n_steps_before
#         for i_d in range(len(data)):
#             data[i_d] = [dd for dd in data[i_d] if not np.isnan(dd)]
#         plt.boxplot(data, labels=agents)
#         plt.gca().set_ylim(bottom=0)
#         if 'True' in env_:
#             plt.ylabel('steps before self identification')
#             plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env}_inference.png')
#         else:
#             plt.ylabel('steps before game solved')
#             plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env}_solved.png')
#     plt.close('all')
#
# # How slower is it when there is noise vs no noise (noise inference in both cases)?
# agent = 'base'
# envs = [('logic-v0', 'logic-noisy-v0'), ('contingency-v0', 'contingency-noisy-v0')]
# plot_name = 'noisy_vs_non_noisy'
# for env_pair in envs:
#     for explore_only in [True, False]:
#         all_n_steps_before = []
#         for env in env_pair:
#             env_ = env + '_' + str(explore_only)
#             n_steps_before = []
#             for i in all_data[env_][agent].keys():
#                 if 'True' in env_:
#                     to_track = np.array(all_data[env_][agent][str(i)]['true_theory_probas']) > 0.7
#                 else:
#                     to_track = np.array(all_data[env_][agent][str(i)]['success'])
#                 indexes = np.argwhere(to_track).flatten()
#                 if indexes.size > 0:
#                     n_steps_before.append(indexes[0])
#                 else:
#                     n_steps_before.append(np.nan)
#             all_n_steps_before.append(n_steps_before)
#
#         fig, ax = plt.subplots(figsize=(10, 7))
#         data = np.array(all_n_steps_before).T
#         plt.boxplot(data, labels=env_pair)
#         plt.gca().set_ylim(bottom=0)
#         if 'True' in env_:
#             plt.ylabel('steps before self identification')
#             plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env_pair[0].split("-")[0]}_inference.png')
#         else:
#             plt.ylabel('steps before game solved')
#             plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env_pair[0].split("-")[0]}_solved.png')
#     plt.close('all')
#
#
#
#
# # how good is the estimation of the noise for the true agent theory?
# # envs = ['logic-shuffle-noisy-v0', 'contingency-shuffle-noisy-v0']
# # agent = 'base'
# # plot_name = 'noise_tracking'
# # for env in envs:
# #     all_noises = []
# #     for i in  all_data[env][agent].keys():
# #         noise = np.array(all_data[env][agent][str(i)]['true_theory_noise_mean'])
# #         all_noises.append(noise)
# #     max_len = np.max([len(data) for data in all_noises])
# #     data = np.zeros((len(all_noises), max_len))
# #     data.fill(np.nan)
# #     for i_noise, noises in enumerate(all_noises):
# #         data[i_noise, :len(noises)] = noises
# #     fig, ax = plt.subplots(figsize=(10, 7))
# #     plt.plot(np.nanmean(data, axis=0))
# #     plt.gca().set_ylim(bottom=0)
# #     plt.ylabel('noise estimate true theory')
# #     plt.fill_between(np.arange(max_len), np.nanmean(data, axis=0) - np.nanstd(data, axis=0), np.nanmean(data, axis=0) + np.nanstd(data, axis=0), alpha=0.2)
# #     plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env}.png')
# #     plt.close('all')
#
# # how fast is it (solving + inferring) if we just do random actions for the exploration step?
# agents = ['base', 'random_explo']
# envs = ['logic-shuffle-noisy-v0', 'contingency-shuffle-noisy-v0']
# plot_name = 'optimal_vs_random_explo'
# for env in envs:
#     for explore_only in [True, False]:
#         env_ = env + '_' + str(explore_only)
#         all_n_steps_before = []
#         for agent in agents:
#             n_steps_before = []
#             for i in all_data[env_][agent].keys():
#                 if 'True' in env_:
#                     to_track = np.array(all_data[env_][agent][str(i)]['true_theory_probas']) > 0.7
#                 else:
#                     to_track = np.array(all_data[env_][agent][str(i)]['success'])
#                 indexes = np.argwhere(to_track).flatten()
#                 if indexes.size > 0:
#                     n_steps_before.append(indexes[0])
#                 else:
#                     n_steps_before.append(np.nan)
#             all_n_steps_before.append(n_steps_before)
#         fig, ax = plt.subplots(figsize=(10, 7))
#         data = all_n_steps_before
#         for i_d in range(len(data)):
#             data[i_d] = [dd for dd in data[i_d] if not np.isnan(dd)]
#         plt.boxplot(data, labels=agents)
#         plt.gca().set_ylim(bottom=0)
#         if 'True' in env_:
#             plt.ylabel('steps before self identification')
#             plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env}_inference.png')
#         else:
#             plt.ylabel('steps before game solved')
#             plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env}_solved.png')
#     plt.close('all')
#
# ##################################################################3
#
expe_name = 'with_agent_change'
data_path = save_dir + expe_name + '.pkl'

with open(data_path, 'rb') as f:
    all_data = pickle.load(f)
#
# # how fast is it (solving + inferring) if we just do random actions for the exploration step?
# agents = ['base', 'random_explo']
# envs = ['changeAgent-shuffle-noisy-v0']
# plot_name = 'optimal_vs_random_explo'
# for env in envs:
#     for explore_only in [True, False]:
#         env_ = env + '_' + str(explore_only)
#         all_n_steps_before = []
#         for agent in agents:
#             n_steps_before = []
#             for i in all_data[env_][agent].keys():
#                 if 'True' in env_:
#                     to_track = np.array(all_data[env_][agent][str(i)]['true_theory_probas']) > 0.7
#                 else:
#                     to_track = np.array(all_data[env_][agent][str(i)]['success'])
#                 indexes = np.argwhere(to_track).flatten()
#                 if indexes.size > 0:
#                     n_steps_before.append(indexes[0])
#                 else:
#                     n_steps_before.append(np.nan)
#             all_n_steps_before.append(n_steps_before)
#         fig, ax = plt.subplots(figsize=(10, 7))
#         data = all_n_steps_before
#         for i_d in range(len(data)):
#             data[i_d] = [dd for dd in data[i_d] if not np.isnan(dd)]
#         plt.boxplot(data, labels=agents)
#         plt.gca().set_ylim(bottom=0)
#         if 'True' in env_:
#             plt.ylabel('steps before self identification')
#             plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env}_inference.png')
#         else:
#             plt.ylabel('steps before game solved')
#             plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env}_solved.png')
#     plt.close('all')

# best heirarchical model?
agents = ['base', 'explicit_resetter', 'current_focused_forgetter', 'hierarchical']
envs = ['changeAgent-v0', 'changeAgent-noisy-v0', 'changeAgent-shuffle-noisy-v0']
plot_name = 'best_hierarchical'
for env in envs:
    for explore_only in [True, False]:
        env_ = env + '_' + str(explore_only)
        all_n_steps_before = []
        for agent in agents:
            n_steps_before = []
            for i in all_data[env_][agent].keys():
                if 'True' in env_:
                    to_track = np.array(all_data[env_][agent][str(i)]['true_theory_probas']) > 0.7
                else:
                    to_track = np.array(all_data[env_][agent][str(i)]['success'])
                indexes = np.argwhere(to_track).flatten()
                if indexes.size > 0:
                    n_steps_before.append(indexes[0])
                else:
                    n_steps_before.append(np.nan)
            all_n_steps_before.append(n_steps_before)
        fig, ax = plt.subplots(figsize=(10, 7))
        data = all_n_steps_before
        for i_d in range(len(data)):
            data[i_d] = [dd for dd in data[i_d] if not np.isnan(dd)]
        plt.boxplot(data, labels=agents)
        plt.gca().set_ylim(bottom=0)
        if 'True' in env_:
            plt.ylabel('steps before self identification')
            plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env}_inference.png')
        else:
            plt.ylabel('steps before game solved')
            plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env}_solved.png')
    plt.close('all')

##################################################################3

expe_name = 'one_switch'
data_path = save_dir + expe_name + '.pkl'

with open(data_path, 'rb') as f:
    all_data = pickle.load(f)

# Post switch recovery: test speed recovery after a switch a t=30
agents = ['base', 'explicit_resetter', 'current_focused_forgetter', 'hierarchical']
envs = ['changeAgent-shuffle-noisy-oneswitch-v0_True', 'changeAgent-shuffle-noisy-oneswitch-v0_False']
plot_name = 'post_switch_recovery'
for env in envs:
    all_n_steps_before = []
    for agent in agents:
        n_steps_before = []
        for i in all_data[env][agent].keys():
            if 'True' in env:
                to_track = np.array(all_data[env][agent][str(i)]['true_theory_probas']) > 0.7
            else:
                to_track = np.array(all_data[env][agent][str(i)]['success'])
            indexes = np.array([i for i in np.argwhere(to_track).flatten() if i > 29])
            if indexes.size > 0:
                n_steps_before.append(indexes[0] - 30)
            else:
                n_steps_before.append(np.nan)
        all_n_steps_before.append(n_steps_before)
    fig, ax = plt.subplots(figsize=(10, 7))
    data = all_n_steps_before
    for i_d in range(len(data)):
        data[i_d] = [dd for dd in data[i_d] if not np.isnan(dd)]
    plt.boxplot(data, labels=agents)
    plt.gca().set_ylim(bottom=0)
    if 'True' in env:
        plt.ylabel('steps before self identification')
        plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env}_inference.png')
    else:
        plt.ylabel('steps before game solved')
        plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{env}_solved.png')
    plt.close('all')

##################################################################3

expe_name = 'switch_frequency'
data_path = save_dir + expe_name + '.pkl'

with open(data_path, 'rb') as f:
    all_data = pickle.load(f)
envs = ['changeAgent-shuffle-noisy-7-v0', 'changeAgent-shuffle-noisy-10-v0', 'changeAgent-shuffle-noisy-15-v0',
        'changeAgent-shuffle-noisy-20-v0']
agents = ['base', 'explicit_resetter', 'current_focused_forgetter', 'hierarchical']
plot_name = 'difficulty_f_freq'
for agent in agents:
    for explore_only in [True, False]:
        all_n_steps_before = []
        for env in envs:
            env_ = env + '_' + str(explore_only)
            n_steps_before = []
            for i in all_data[env_][agent].keys():
                if 'True' in env_:
                    to_track = np.array(all_data[env_][agent][str(i)]['true_theory_probas']) > 0.7
                else:
                    to_track = np.array(all_data[env_][agent][str(i)]['success'])
                indexes = np.array([i for i in np.argwhere(to_track).flatten()])
                if indexes.size > 0:
                    n_steps_before.append(indexes[0])
                else:
                    n_steps_before.append(np.nan)
            all_n_steps_before.append(n_steps_before)
        fig, ax = plt.subplots(figsize=(15, 7))
        data = all_n_steps_before
        for i_d in range(len(data)):
            data[i_d] = [dd for dd in data[i_d] if not np.isnan(dd)]
        plt.boxplot(data, labels=envs)
        plt.gca().set_ylim(bottom=0)
        if 'True' in env_:
            plt.ylabel('steps before self identification')
            plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{agent}_inference.png')
        else:
            plt.ylabel('steps before game solved')
            plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{agent}_solved.png')
            frac_solved = [len(d)/10 for d in data]
            fig, ax = plt.subplots(figsize=(15, 7))
            plt.bar(np.arange(len(frac_solved)), frac_solved)
            plt.xticks(np.arange(len(frac_solved)), envs)
            plt.ylabel('fraction solved')
            plt.savefig(f'{plot_dir}{expe_name}_{plot_name}_{agent}_frac_solved.png')
    plt.close('all')
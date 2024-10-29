import numpy as np
from Population_gen import execute_population_analysis, Segment
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import copy


def evolutionary_game_execution(segment_params, intra_params, inter_params, a_params, b_params, iterations, EPA, HPA,
                                title, group_names, delta=0.01):
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 = a_params
    b1, b2, b3, b4 = b_params

    global history
    history = []

    def initialize_decision(G):
        for node in G.nodes:
            attrs = G.nodes[node]
            AE, AB, SN, PBC = attrs['AE'], attrs['AB'], attrs['SN'], attrs['PBC']
            PBC_scaled = (PBC + 1) / 2
            numerator = a2 * AE + a3 * AB + a4 * SN + a5 * PBC_scaled * (a7 * AB + a8 * AE)
            denominator = a2 + a3 + a4 + a5 * (a7 + a8)
            decision = numerator / denominator
            attrs['decision'] = decision
            attrs['behavior'] = 1 if decision > 0 else -1

    def update_AE_AB(G):
        for node in G.nodes:
            attrs = G.nodes[node]
            AE_t_1, AB_t_1, SN, decision_t_1 = attrs['AE'], attrs['AB'], attrs['SN'], attrs['decision']
            neighbors = list(G.neighbors(node))
            if neighbors:
                NAD = np.mean([G.nodes[n]['behavior'] * G.edges[node, n]['weight'] for n in neighbors])
            else:
                NAD = 0  # If no neighbors, set NAD to 0

            # Calculate AE_new with conditional coefficients
            b1_eff = b1 if EPA != 0 else 0
            b3_eff = b3 if HPA != 0 else 0
            a9_eff = a9 if NAD != 0 else 0

            # Ensure the remaining coefficient sums up to 1
            remaining_AE = 1 - b1_eff*EPA - b3_eff*HPA - a9_eff * SN
            AE_new = b1_eff * EPA + b3_eff * HPA + a9_eff * NAD * SN + remaining_AE * AE_t_1

            # Calculate AB_new with conditional coefficients
            b2_eff = b2 if EPA != 0 else 0
            b4_eff = b4 if HPA != 0 else 0
            a10_eff = a10 if NAD != 0 else 0

            # Ensure the remaining coefficient sums up to 1
            remaining_AB = 1 - b2_eff*EPA - b4_eff*HPA - a6 - a10_eff * SN
            AB_new = b2_eff * EPA + b4_eff * HPA + a10_eff * NAD * SN + a6 * AE_new + remaining_AB * AB_t_1

            # Clip AE and AB to [-1, 1]
            AE_new = np.clip(AE_new, -1, 1)
            AB_new = np.clip(AB_new, -1, 1)

            attrs['AE'], attrs['AB'] = AE_new, AB_new

    def update_decision(G):
        for node in G.nodes:
            attrs = G.nodes[node]
            decision_t_1 = attrs['decision']
            AE, AB, SN, PBC = attrs['AE'], attrs['AB'], attrs['SN'], attrs['PBC']
            PBC_scaled = (PBC + 1) / 2
            numerator = a1 * decision_t_1 + a2 * AE + a3 * AB + a4 * SN + a5 * PBC_scaled*(a7 * AB + a8 * AE)
            denominator = a1 + a2 + a3 + a4 + a5 * (a7 + a8)
            decision = numerator / denominator
            attrs['decision'] = decision
            attrs['behavior'] = 1 if decision > 0 else -1

    def evolutionary_game(G, iterations, delta):
        initialize_decision(G)  # Initialize the first decision
        save_history(G, 0)  # Save the initial state

        for iteration in range(iterations):
            update_AE_AB(G)
            update_decision(G)
            save_history(G, iteration + 1)

            if iteration >= 5:
                current_decisions = np.array([history[iteration + 1][node] for node in G.nodes])
                prev_decisions = np.array([history[iteration][node] for node in G.nodes])
                prev_prev_decisions = np.array([history[iteration - 1][node] for node in G.nodes])

                mean_sq_diff_prev = np.mean((current_decisions - prev_decisions) ** 2)
                mean_sq_diff_prev_prev = np.mean((current_decisions - prev_prev_decisions) ** 2)

                if mean_sq_diff_prev < delta or mean_sq_diff_prev_prev < delta:
                    print(f"Converged after {iteration + 1} iterations")
                    break

        return G

    def save_history(G, iteration):
        history.append({node: G.nodes[node]['behavior'] for node in G.nodes})

    # Function to plot histograms
    def plot_histograms(initial_state, final_state, title):
        total_population_size = sum(segment_sizes.values())
        max_segment_size = max(segment_sizes.values())

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Initial state
        behaviors_initial = [initial_state[node] for node in initial_state]
        segment_behaviors_initial = {segment: [initial_state[f"{segment}_{i}"] for i in range(segment_sizes[segment])]
                                     for segment in range(len(segment_sizes))}

        # Total population histogram (initial)
        counts_initial = [behaviors_initial.count(-1), behaviors_initial.count(1)]
        axes[0, 0].bar([-1, 1], counts_initial, width=0.8)
        axes[0, 0].set_xticks([-1, 1])
        axes[0, 0].set_xticklabels(['SUPT', 'Reusable'])
        axes[0, 0].set_title('Total Population Behaviors (Initial)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_ylim([0, total_population_size])

        # Segmented population histogram (initial)
        bar_width = 0.15
        x = np.arange(2)  # 2 bins: SUPT and Reusable
        for i, segment in enumerate(segment_behaviors_initial):
            counts = [segment_behaviors_initial[segment].count(-1), segment_behaviors_initial[segment].count(1)]
            axes[0, 1].bar(x + i * bar_width, counts, width=bar_width, label=group_names[segment])

        axes[0, 1].set_xticks(x + bar_width * (len(segment_sizes) - 1) / 2)
        axes[0, 1].set_xticklabels(['SUPT', 'Reusable'])
        axes[0, 1].set_title('Segmented Population Behaviors (Initial)')
        axes[0, 1].legend(fontsize='small')
        axes[0, 1].set_ylim([0, max_segment_size])

        # Final state
        behaviors_final = [final_state[node] for node in final_state]
        segment_behaviors_final = {segment: [final_state[f"{segment}_{i}"] for i in range(segment_sizes[segment])] for
                                   segment in range(len(segment_sizes))}

        # Total population histogram (final)
        counts_final = [behaviors_final.count(-1), behaviors_final.count(1)]
        axes[1, 0].bar([-1, 1], counts_final, width=0.8)
        axes[1, 0].set_xticks([-1, 1])
        axes[1, 0].set_xticklabels(['SUPT', 'Reusable'])
        axes[1, 0].set_title('Total Population Behaviors (Final)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_ylim([0, total_population_size])

        # Segmented population histogram (final)
        for i, segment in enumerate(segment_behaviors_final):
            counts = [segment_behaviors_final[segment].count(-1), segment_behaviors_final[segment].count(1)]
            axes[1, 1].bar(x + i * bar_width, counts, width=bar_width, label=group_names[segment])

        axes[1, 1].set_xticks(x + bar_width * (len(segment_sizes) - 1) / 2)
        axes[1, 1].set_xticklabels(['SUPT', 'Reusable'])
        axes[1, 1].set_title('Segmented Population Behaviors (Final)')
        axes[1, 1].legend(fontsize='small')
        axes[1, 1].set_ylim([0, max_segment_size])

        plt.tight_layout()
        plt.savefig(f"./{title.replace(':', '').replace(' ', '_')}/behaviors_histogram.jpeg", format='jpeg')
        plt.show()

    def animate_histograms(history, title):
        total_population_size = sum(segment_sizes.values())
        max_segment_size = max(segment_sizes.values())

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Initial state
        initial_state = history[0]
        behaviors_initial = [initial_state[node] for node in initial_state]
        segment_behaviors_initial = {segment: [initial_state[f"{segment}_{i}"] for i in range(segment_sizes[segment])]
                                     for segment in range(len(segment_sizes))}

        # Total population histogram (initial)
        counts_initial = [behaviors_initial.count(-1), behaviors_initial.count(1)]
        axes[0, 0].bar([-1, 1], counts_initial, width=0.8)
        axes[0, 0].set_xticks([-1, 1])
        axes[0, 0].set_xticklabels(['SUPT', 'Reusable'])
        axes[0, 0].set_title('Total Population Behaviors (Initial)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_ylim([0, total_population_size])

        # Segmented population histogram (initial)
        bar_width = 0.15
        x = np.arange(2)  # 2 bins: SUPT and Reusable
        for i, segment in enumerate(segment_behaviors_initial):
            counts = [segment_behaviors_initial[segment].count(-1), segment_behaviors_initial[segment].count(1)]
            axes[0, 1].bar(x + i * bar_width, counts, width=bar_width, label=group_names[segment])

        axes[0, 1].set_xticks(x + bar_width * (len(segment_sizes) - 1) / 2)
        axes[0, 1].set_xticklabels(['SUPT', 'Reusable'])
        axes[0, 1].set_title('Segmented Population Behaviors (Initial)')
        axes[0, 1].legend(fontsize='small')
        axes[0, 1].set_ylim([0, max_segment_size])

        def update_hist(num):
            final_state = history[num]
            behaviors_final = [final_state[node] for node in final_state]
            segment_behaviors_final = {segment: [final_state[f"{segment}_{i}"] for i in range(segment_sizes[segment])]
                                       for segment in range(len(segment_sizes))}

            # Update total population histogram (final)
            counts_final = [behaviors_final.count(-1), behaviors_final.count(1)]
            axes[1, 0].cla()
            axes[1, 0].bar([-1, 1], counts_final, width=0.8)
            axes[1, 0].set_xticks([-1, 1])
            axes[1, 0].set_xticklabels(['SUPT', 'Reusable'])
            axes[1, 0].set_title(f'Total Population Behaviors (Final) - Iteration {num}')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_ylim([0, total_population_size])

            # Update segmented population histogram (final)
            axes[1, 1].cla()
            for i, segment in enumerate(segment_behaviors_final):
                counts = [segment_behaviors_final[segment].count(-1), segment_behaviors_final[segment].count(1)]
                axes[1, 1].bar(x + i * bar_width, counts, width=bar_width, label=group_names[segment])

            axes[1, 1].set_xticks(x + bar_width * (len(segment_sizes) - 1) / 2)
            axes[1, 1].set_xticklabels(['SUPT', 'Reusable'])
            axes[1, 1].set_title(f'Segmented Population Behaviors (Final) - Iteration {num}')
            axes[1, 1].legend(fontsize='small')
            axes[1, 1].set_ylim([0, max_segment_size])

        ani = animation.FuncAnimation(fig, update_hist, frames=len(history), repeat=False)
        plt.tight_layout()

        # Save animation
        directory = f"./{title.replace(':', '').replace(' ', '_')}/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Try to use ffmpeg if available, otherwise fall back to pillow
        try:
            ani.save(os.path.join(directory, "behavior_histograms.mp4"), writer='ffmpeg')
        except ValueError as e:
            print(f"ffmpeg writer not available, using pillow instead: {e}")
            ani.save(os.path.join(directory, "behavior_histograms.mov"), writer='pillow')

        plt.show()

    G_initial = execute_population_analysis(segment_params, intra_params, inter_params, title)
    segment_sizes = {idx: params[0] for idx, params in enumerate(segment_params)}

    # Initialize decisions to set behavior attribute
    initialize_decision(G_initial)

    # Make a copy of the initial state
    G_initial_copy = copy.deepcopy(G_initial)

    G_final = evolutionary_game(G_initial, iterations, delta)

    # Save the final behaviors for further analysis
    final_behaviors = {node: G_final.nodes[node]['behavior'] for node in G_final.nodes}

    # Plot initial and final state histograms using history
    plot_histograms(history[0], history[-1], title)

    # Create the animation
    animate_histograms(history, title)

    return history

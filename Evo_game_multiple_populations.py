import numpy as np
from scipy.stats import chi2_contingency, norm
import matplotlib.pyplot as plt
import os
from Evolutionary_game import evolutionary_game_execution

# Define the population configurations
populations = [
    {
        'segment_params': [
            (500, 1, 0.0, 1, 0.0, 0.5, 0.1, 0.5, 0.2),
            (500, -1, 0.0, -1, 0.0, 0.5, 0.1, 0.5, 0.2),
            (4000, 0.0, 0.3, 0.0, 0.4, 0.5, 0.1, 0.5, 0.2),
            (1000, 0.8, 0.2, 0.7, 0.3, 0.8, 0.1, 0.5, 0.2),
            (1000, -0.8, 0.2, -0.7, 0.3, 0.8, 0.1, 0.5, 0.2)
        ],
        'intra_params': [
            (50, 5, 0.3, 0.01),
            (50, 5, 0.3, 0.01),
            (400, 40, 0.3, 0.01),
            (100, 10, 0.8, 0.1),
            (100, 10, 0.8, 0.1)
        ],
        'inter_params': [
            [(100, 10, 0.5, 0.2), (200, 10, 0.5, 0.2), (30, 10, 0.2, 0.2), (30, 10, 0.2, 0.2)],
            [(200, 10, 0.5, 0.2), (30, 10, 0.2, 0.2), (30, 10, 0.2, 0.2)],
            [(30, 10, 0.2, 0.2), (30, 10, 0.2, 0.2)],
            [(30, 10, 0.2, 0.2)]
        ],
        'title': 'Symmetric Pop.',
        'group_names': {0: 'Extreme - reusables', 1: 'Extreme - SUPT', 2: 'General', 3: 'Mild - reusables', 4: 'Mild - SUPT'}
    },
    {
        'segment_params': [
            (400, 1, 0.0, 1, 0.0, 0.5, 0.1, 0.5, 0.2),
            (600, -1, 0.0, -1, 0.0, 0.5, 0.1, 0.5, 0.2),
            (3000, -0.3, 0.2, -0.2, 0.2, 0.5, 0.1, 0.5, 0.2),
            (1000, 0.8, 0.2, 0.7, 0.3, 0.8, 0.1, 0.5, 0.2),
            (2000, -0.8, 0.2, -0.7, 0.3, 0.8, 0.1, 0.5, 0.2)
        ],
        'intra_params': [
            (50, 5, 0.3, 0.01),
            (50, 5, 0.3, 0.01),
            (400, 40, 0.3, 0.01),
            (100, 10, 0.8, 0.1),
            (100, 10, 0.8, 0.1)
        ],
        'inter_params': [
            [(100, 10, 0.5, 0.2), (200, 10, 0.5, 0.2), (30, 10, 0.2, 0.2), (30, 10, 0.2, 0.2)],
            [(200, 10, 0.5, 0.2), (30, 10, 0.2, 0.2), (30, 10, 0.2, 0.2)],
            [(30, 10, 0.2, 0.2), (30, 10, 0.2, 0.2)],
            [(30, 10, 0.2, 0.2)]
        ],
        'title': 'Mixed Pop.',
        'group_names': {0: 'Extreme - reusables', 1: 'Extreme - SUPT', 2: 'General', 3: 'Mild - reusables', 4: 'Mild - SUPT'}
    },
    {
        'segment_params': [
            (400, 1, 0.0, 1, 0.0, 0.5, 0.1, 0.5, 0.2),
            (600, -1, 0.0, -1, 0.0, 0.5, 0.1, 0.5, 0.2),
            (3000, -0.3, 0.2, -0.2, 0.2, 0.5, 0.1, 0.5, 0.2),
            (1000, 0.8, 0.2, 0.7, 0.3, 0.8, 0.1, 0.5, 0.2),
            (2000, -0.8, 0.2, -0.7, 0.3, 0.8, 0.1, 0.5, 0.2)
        ],
        'intra_params': [
            (100, 20, 0.6, 0.1),
            (150, 20, 0.6, 0.1),
            (700, 40, 0.6, 0.1),
            (500, 50, 0.8, 0.1),
            (500, 50, 0.8, 0.1)
        ],
        'inter_params': [
            [(50, 10, 0.3, 0.2), (50, 10, 0.3, 0.2), (20, 10, 0.2, 0.2), (20, 10, 0.2, 0.2)],
            [(50, 10, 0.3, 0.2), (20, 10, 0.2, 0.2), (20, 10, 0.2, 0.2)],
            [(20, 10, 0.2, 0.2), (20, 10, 0.2, 0.2)],
            [(20, 10, 0.2, 0.2)]
        ],
        'title': 'Fragmented Pop.',
        'group_names': {0: 'Extreme - reusables', 1: 'Extreme - SUPT', 2: 'General', 3: 'Mild - reusables', 4: 'Mild - SUPT'}
    },
    {
        'segment_params': [
            (400, 1, 0.0, 1, 0.0, 0.5, 0.1, 0.5, 0.2),
            (600, -1, 0.0, -1, 0.0, 0.5, 0.1, 0.5, 0.2),
            (3000, -0.3, 0.2, -0.2, 0.2, 0.5, 0.1, 0.5, 0.2),
            (1000, 0.8, 0.2, 0.7, 0.3, 0.8, 0.1, 0.5, 0.2),
            (2000, -0.8, 0.2, -0.7, 0.3, 0.8, 0.1, 0.5, 0.2)
        ],
        'intra_params': [
            (40, 10, 0.2, 0.05),
            (40, 10, 0.2, 0.05),
            (100, 40, 0.2, 0.05),
            (200, 10, 0.2, 0.05),
            (200, 10, 0.2, 0.05)
        ],
        'inter_params': [
            [(200, 20, 0.6, 0.2), (400, 30, 0.6, 0.2), (100, 20, 0.5, 0.2), (100, 20, 0.5, 0.2)],
            [(400, 30, 0.6, 0.2), (100, 20, 0.5, 0.2), (100, 20, 0.5, 0.2)],
            [(100, 20, 0.5, 0.2), (100, 20, 0.5, 0.2)],
            [(100, 20, 0.5, 0.2)]
        ],
        'title': 'Connected Pop.',
        'group_names': {0: 'Extreme - reusables', 1: 'Extreme - SUPT', 2: 'General', 3: 'Mild - reusables', 4: 'Mild - SUPT'}
    }
]

# Parameters for the simulation
a_params = [0.5, 0.087, 0.306, 0.0, 0.193, 0.422, 0.472, 0.071, 0.09, 0.229]
b_params = [0.234, 0.045, 0.346, 0.019]
iterations = 100
delta = 5e-6

# Define the interventions (EPA, HPA)
interventions = [
    (0.0, 0.0),
    (0.06, 0.0),
    (0.0, 0.06),
]

# Run the evolutionary game for each population and intervention
histories = []
for idx, population in enumerate(populations):
    for intervention in interventions:
        EPA, HPA = intervention
        title = f"{population['title']} - {intervention}"
        history = evolutionary_game_execution(population['segment_params'], population['intra_params'],
                                              population['inter_params'], a_params, b_params, iterations, EPA, HPA,
                                              title, population['group_names'], delta=delta)
        histories.append((history, title))


def z_test_proportions(count1, nobs1, count2, nobs2):
    # Calculate proportions
    p1 = count1 / nobs1
    p2 = count2 / nobs2
    p_combined = (count1 + count2) / (nobs1 + nobs2)

    # Handle edge cases where p_combined * (1 - p_combined) could be zero
    if p_combined == 0 or p_combined == 1:
        if p1 == p2:
            return 1.0  # p-value of 1.0 indicates no difference when both proportions are the same
        else:
            return 0.0  # p-value of 0.0 indicates a significant difference when proportions are different

    # Calculate Z statistic
    z_stat = (p1 - p2) / np.sqrt(p_combined * (1 - p_combined) * (1 / nobs1 + 1 / nobs2))
    p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))

    return p_value

# Function to filter segment-specific histories
def get_segment_history(histories, segment_index):
    segment_histories = []

    for history, title in histories:
        segment_history = [
            {k: v for k, v in state.items() if k.startswith(f"{segment_index}_")}
            for state in history
        ]
        segment_histories.append((segment_history, title))

    return segment_histories

# Function to visualize significance levels
def visualize_significance(p_values, labels, title, output_file):
    fig, ax = plt.subplots(figsize=(12, 8))  # Increased figure size for better readability
    cax = ax.matshow(p_values, cmap=plt.colormaps['RdYlBu_r'])
    fig.colorbar(cax)

    for (i, j), val in np.ndenumerate(p_values):
        ax.text(j, i, f'{val:.3e}', ha='center', va='center', fontsize=6)  # Smaller font size for text

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='left', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    plt.savefig(output_file, format='jpeg')
    plt.show()

# Function to compare populations and visualize significance
def compare_population_histories(histories, output_dir):
    num_histories = len(histories)
    p_values = np.zeros((num_histories, num_histories))

    for i in range(num_histories):
        for j in range(i, num_histories):
            history_1, title_1 = histories[i]
            history_2, title_2 = histories[j]

            if i == j:
                # Compare initial and final state for the same population (diagonal)
                initial_counts = [list(history_1[0].values()).count(-1), list(history_1[0].values()).count(1)]
                final_counts = [list(history_1[-1].values()).count(-1), list(history_1[-1].values()).count(1)]
                p_value = z_test_proportions(final_counts[0], sum(final_counts), initial_counts[0], sum(initial_counts))
            else:
                # Comparing final states of different populations
                final_counts_1 = [list(history_1[-1].values()).count(-1), list(history_1[-1].values()).count(1)]
                final_counts_2 = [list(history_2[-1].values()).count(-1), list(history_2[-1].values()).count(1)]
                p_value = z_test_proportions(final_counts_1[0], sum(final_counts_1), final_counts_2[0], sum(final_counts_2))

            p_values[i, j] = p_values[j, i] = p_value

    # Visualize the significance matrix
    labels = [f"{title.split(' - ')[0]} ({title.split(' - ')[1]})" for _, title in histories]
    visualize_significance(p_values, labels, "Population Comparison Significance", os.path.join(output_dir, "population_significance.jpeg"))

    return p_values

# Function to compare segments between populations and visualize significance
def compare_segment_histories(histories, output_dir):
    segment_results = {}
    num_segments = len(populations[0]['segment_params'])

    for seg in range(num_segments):
        segment_histories = get_segment_history(histories, seg)
        p_values = np.zeros((len(segment_histories), len(segment_histories)))

        for i in range(len(segment_histories)):
            for j in range(i, len(segment_histories)):
                history_1, title_1 = segment_histories[i]
                history_2, title_2 = segment_histories[j]

                if i == j:
                    # Compare initial and final state for the same segment (diagonal)
                    initial_counts = [list(history_1[0].values()).count(-1), list(history_1[0].values()).count(1)]
                    final_counts = [list(history_1[-1].values()).count(-1), list(history_1[-1].values()).count(1)]
                    p_value = z_test_proportions(final_counts[0], sum(final_counts), initial_counts[0], sum(initial_counts))
                else:
                    # Comparing final states of the same segment across different populations
                    final_counts_1 = [list(history_1[-1].values()).count(-1), list(history_1[-1].values()).count(1)]
                    final_counts_2 = [list(history_2[-1].values()).count(-1), list(history_2[-1].values()).count(1)]
                    p_value = z_test_proportions(final_counts_1[0], sum(final_counts_1), final_counts_2[0], sum(final_counts_2))

                p_values[i, j] = p_values[j, i] = p_value

        # Visualize the segment comparison matrix
        labels = [f"{title.split(' - ')[0]} ({title.split(' - ')[1]})" for _, title in segment_histories]
        segment_name = populations[0]['group_names'][seg]
        visualize_significance(p_values, labels, f"Segment Comparison: {segment_name}", os.path.join(output_dir, f"segment_{seg}_significance.jpeg"))

        segment_results[f"Segment {segment_name}"] = p_values

    return segment_results

# Function to plot histograms of the entire population
def plot_population_histories(histories, output_dir):
    num_populations = len(populations)
    num_interventions = len(interventions) + 1  # +1 for the initial state

    fig, axes = plt.subplots(num_populations, num_interventions, figsize=(num_interventions * 5, num_populations * 5), squeeze=False)

    for i, (history, title) in enumerate(histories):
        population_idx = i // len(interventions)
        intervention_idx = (i % len(interventions)) + 1  # +1 because 0 is for the initial state

        initial_state = history[0]
        final_state = history[-1]

        # Plot initial state
        if intervention_idx == 1:
            behaviors_initial = [initial_state[node] for node in initial_state]
            counts_initial = [behaviors_initial.count(-1), behaviors_initial.count(1)]
            bars = axes[population_idx, 0].bar([-1, 1], counts_initial, width=0.8)
            axes[population_idx, 0].set_xticks([-1, 1])
            axes[population_idx, 0].set_xticklabels(['SUPT', 'Reusable'])
            axes[population_idx, 0].set_title(f'{populations[population_idx]["title"]} (Initial)', fontsize=10)
            axes[population_idx, 0].set_ylabel('Count')
            axes[population_idx, 0].set_ylim([0, sum([params[0] for params in populations[population_idx]['segment_params']])])

            # Add count numbers on top of each bar
            for bar in bars:
                height = bar.get_height()
                axes[population_idx, 0].annotate(f'{height}',
                                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                                 xytext=(0, 3),  # 3 points vertical offset
                                                 textcoords="offset points",
                                                 ha='center', va='bottom')

        # Plot final state
        behaviors_final = [final_state[node] for node in final_state]
        counts_final = [behaviors_final.count(-1), behaviors_final.count(1)]
        bars = axes[population_idx, intervention_idx].bar([-1, 1], counts_final, width=0.8)
        axes[population_idx, intervention_idx].set_xticks([-1, 1])
        axes[population_idx, intervention_idx].set_xticklabels(['SUPT', 'Reusable'])
        axes[population_idx, intervention_idx].set_title(f'{populations[population_idx]["title"]} - {interventions[intervention_idx - 1]} (Iter {len(history)})', fontsize=10)
        axes[population_idx, intervention_idx].set_ylim([0, sum([params[0] for params in populations[population_idx]['segment_params']])])

        # Add count numbers on top of each bar
        for bar in bars:
            height = bar.get_height()
            axes[population_idx, intervention_idx].annotate(f'{height}',
                                                            xy=(bar.get_x() + bar.get_width() / 2, height),
                                                            xytext=(0, 3),  # 3 points vertical offset
                                                            textcoords="offset points",
                                                            ha='center', va='bottom')

    plt.tight_layout()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, "population_histories.jpeg"), format='jpeg')
    plt.show()

# Function to plot histograms of the different segments
def plot_segment_histories(histories, output_dir):
    num_populations = len(populations)
    num_interventions = len(interventions) + 1  # +1 for the initial state

    fig, axes = plt.subplots(num_populations, num_interventions, figsize=(num_interventions * 5, num_populations * 5), squeeze=False)

    for i, (history, title) in enumerate(histories):
        population_idx = i // len(interventions)
        intervention_idx = (i % len(interventions)) + 1  # +1 because 0 is for the initial state

        initial_state = history[0]
        final_state = history[-1]
        segment_sizes = {idx: params[0] for idx, params in enumerate(populations[population_idx]['segment_params'])}

        # Plot initial state
        if intervention_idx == 1:
            segment_behaviors_initial = {segment: [initial_state[f"{segment}_{i}"] for i in range(segment_sizes[segment])]
                                         for segment in range(len(segment_sizes))}
            bar_width = 0.15
            x = np.arange(2)  # 2 bins: SUPT and Reusable
            for j, segment in enumerate(segment_behaviors_initial):
                counts = [segment_behaviors_initial[segment].count(-1), segment_behaviors_initial[segment].count(1)]
                bars = axes[population_idx, 0].bar(x + j * bar_width, counts, width=bar_width, label=populations[population_idx]['group_names'][segment])
                # Add count numbers on top of each bar
                for k, count in enumerate(counts):
                    axes[population_idx, 0].text(x[k] + j * bar_width, count + 20, str(count), ha='center')
            axes[population_idx, 0].set_xticks(x + bar_width * (len(segment_sizes) - 1) / 2)
            axes[population_idx, 0].set_xticklabels(['SUPT', 'Reusable'])
            axes[population_idx, 0].set_title(f'{populations[population_idx]["title"]} (Initial)', fontsize=10)

            # Add legend only to the first subplot
            if population_idx == 0:
                axes[population_idx, 0].legend(loc='upper right', fontsize='small')
            axes[population_idx, 0].set_ylim([0, max(segment_sizes.values())])

        # Plot final state
        segment_behaviors_final = {segment: [final_state[f"{segment}_{i}"] for i in range(segment_sizes[segment])]
                                   for segment in range(len(segment_sizes))}
        for j, segment in enumerate(segment_behaviors_final):
            counts = [segment_behaviors_final[segment].count(-1), segment_behaviors_final[segment].count(1)]
            bars = axes[population_idx, intervention_idx].bar(x + j * bar_width, counts, width=bar_width, label=populations[population_idx]['group_names'][segment])
            # Add count numbers on top of each bar
            for k, count in enumerate(counts):
                axes[population_idx, intervention_idx].text(x[k] + j * bar_width, count + 20, str(count), ha='center')
        axes[population_idx, intervention_idx].set_xticks(x + bar_width * (len(segment_sizes) - 1) / 2)
        axes[population_idx, intervention_idx].set_xticklabels(['SUPT', 'Reusable'])
        axes[population_idx, intervention_idx].set_title(f'{populations[population_idx]["title"]} - {interventions[intervention_idx - 1]} (Iter {len(history)})', fontsize=10)
        # No legend for the final state subplots
        axes[population_idx, intervention_idx].set_ylim([0, max(segment_sizes.values())])

    plt.tight_layout()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, "segment_histories.jpeg"), format='jpeg')
    plt.show()

# Define output directory for the results
output_dir = './summary'

# Plot the population and segment histories
plot_population_histories(histories, output_dir)
plot_segment_histories(histories, output_dir)

# Compare the population histories
population_comparison_results = compare_population_histories(histories, output_dir)

# Compare the segment histories
segment_comparison_results = compare_segment_histories(histories, output_dir)

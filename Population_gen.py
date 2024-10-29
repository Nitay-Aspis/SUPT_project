import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import os

# Set seed for consistency
random.seed(42)
np.random.seed(42)

# Colors for each group
group_colors = {0: 'blue', 1: 'red', 2: 'yellow', 3: 'purple', 4: 'green'}

# Mapping group identifiers to full names
group_names = {0: 'Extreme - reusables', 1: 'Extreme - SUPT', 2: 'General', 3: 'Mild - reusables', 4: 'Mild - SUPT'}

node_size = 5  # Adjusted node size
edge_width_factor = 0.5  # Adjusted edge width factor


class Segment:
    def __init__(self, size, mean_ae, std_ae, mean_ab, std_ab, mean_sn, std_sn, mean_pbc, std_pbc):
        self.size = size
        self.mean_ae = mean_ae
        self.std_ae = std_ae
        self.mean_ab = mean_ab
        self.std_ab = std_ab
        self.mean_sn = mean_sn
        self.std_sn = std_sn
        self.mean_pbc = mean_pbc
        self.std_pbc = std_pbc

        # Debugging: Check for invalid standard deviations
        assert self.std_ae >= 0, f"Invalid std_ae: {self.std_ae}"
        assert self.std_ab >= 0, f"Invalid std_ab: {self.std_ab}"
        assert self.std_sn >= 0, f"Invalid std_sn: {self.std_sn}"
        assert self.std_pbc >= 0, f"Invalid std_pbc: {self.std_pbc}"

    def generate_consumers(self):
        consumers = {
            'AE': np.clip(np.random.normal(self.mean_ae, self.std_ae, self.size), -1, 1),
            'AB': np.clip(np.random.normal(self.mean_ab, self.std_ab, self.size), -1, 1),
            'SN': np.clip(np.random.normal(self.mean_sn, self.std_sn, self.size), -1, 1),
            'PBC': np.clip(np.random.normal(self.mean_pbc, self.std_pbc, self.size), -1, 1)
        }
        return pd.DataFrame(consumers)


def generate_connections(segments, intra_params, inter_params):
    G = nx.Graph()

    # Create nodes for each consumer in each segment
    for idx, segment in enumerate(segments):
        consumers = segment.generate_consumers()
        consumers['segment'] = idx
        consumers['decision'] = np.clip(np.random.normal(0, 0.1, len(consumers)), -1, 1)  # Initial decision
        for i in range(len(consumers)):
            G.add_node(f"{idx}_{i}", **consumers.iloc[i].to_dict())

    # Add intra-segment connections based on intra_params
    for idx, (mean_num_connections, std_num_connections, mean_strength, std_strength) in enumerate(intra_params):
        segment_nodes = [n for n in G.nodes if G.nodes[n]['segment'] == idx]
        num_connections = int(np.clip(np.random.normal(mean_num_connections, std_num_connections), 0,
                                      len(segment_nodes) * (len(segment_nodes) - 1) // 2))
        if len(segment_nodes) > 1:
            for _ in range(num_connections):
                i, j = np.random.choice(len(segment_nodes), 2, replace=False)
                strength = np.clip(np.random.normal(mean_strength, std_strength), 0, 1)
                G.add_edge(segment_nodes[i], segment_nodes[j], weight=strength)

    # Add inter-segment connections based on inter_params
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            mean_num_connections, std_num_connections, mean_strength, std_strength = inter_params[i][j - i - 1]
            segment_i_nodes = [n for n in G.nodes if G.nodes[n]['segment'] == i]
            segment_j_nodes = [n for n in G.nodes if G.nodes[n]['segment'] == j]
            num_connections = int(np.clip(np.random.normal(mean_num_connections, std_num_connections), 0,
                                          len(segment_i_nodes) * len(segment_j_nodes)))
            for _ in range(num_connections):
                node_i = np.random.choice(segment_i_nodes)
                node_j = np.random.choice(segment_j_nodes)
                strength = np.clip(np.random.normal(mean_strength, std_strength), 0, 1)
                G.add_edge(node_i, node_j, weight=strength)

    return G


def custom_layout(G, segments):
    group_centers = {
        0: np.array([0.25, 0.75]),
        1: np.array([0.75, 0.75]),
        2: np.array([0.5, 0.5]),
        3: np.array([0.25, 0.25]),
        4: np.array([0.75, 0.25]),
    }
    pos = {}
    for node in G.nodes:
        segment = G.nodes[node]['segment']
        center = group_centers[segment]
        pos[node] = center + np.random.normal(0, 0.05, 2)
    return pos


def plot_population(G, segments, title):
    pos = custom_layout(G, segments)  # Custom layout positions

    # Draw nodes
    segments_set = set(nx.get_node_attributes(G, 'segment').values())
    for segment in segments_set:
        nodes = [n for n in G.nodes if G.nodes[n]['segment'] == segment]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=group_colors[segment], node_size=node_size,
                               label=group_names[segment])

    # Draw edges
    weights = nx.get_edge_attributes(G, 'weight').values()
    edge_widths = [w * edge_width_factor for w in weights]
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=edge_widths, alpha=0.5)

    plt.legend(scatterpoints=1, loc='best', fontsize='small')
    plt.axis('equal')
    plt.axis('off')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save the plot
    directory = f"./{title.replace(':', '').replace(' ', '_')}/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, "population_network.jpeg"), format='jpeg')
    plt.show()


def plot_debugging_info(segments, title):
    directory = f"./{title.replace(':', '').replace(' ', '_')}/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig, axes = plt.subplots(len(segments), 1, figsize=(12, 3 * len(segments)))

    for idx, segment in enumerate(segments):
        consumers = segment.generate_consumers().head(10)
        consumers.plot(kind='bar', ax=axes[idx], title=f'Segment {idx} - {group_names[idx]}',
                       color=['blue', 'red', 'yellow', 'purple'])

    plt.tight_layout()
    plt.savefig(os.path.join(directory, "debugging_info.jpeg"), format='jpeg')
    plt.show()


def plot_histograms(segments, title):
    directory = f"./{title.replace(':', '').replace(' ', '_')}/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    attributes = ['AE', 'AB', 'SN', 'PBC']
    for i, attribute in enumerate(attributes):
        ax = axes[i // 2, i % 2]
        for idx, segment in enumerate(segments):
            consumers = segment.generate_consumers()
            consumers[attribute].plot(kind='hist', alpha=0.5, ax=ax, label=f'Segment {idx} - {group_names[idx]}',
                                      color=group_colors[idx])
        ax.set_title(attribute)
        ax.legend(fontsize='small')

    plt.tight_layout()
    plt.savefig(os.path.join(directory, "histograms.jpeg"), format='jpeg')
    plt.show()


def execute_population_analysis(segment_params, intra_params, inter_params, title):
    segments = [Segment(*params) for params in segment_params]
    G = generate_connections(segments, intra_params, inter_params)
    plot_population(G, segments, title)
    plot_debugging_info(segments, title)
    plot_histograms(segments, title)
    return G

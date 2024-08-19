import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
import networkx as nx
import matplotlib.patches as patches

rng = np.random.default_rng(12345)


def moment_estimate_k(offspring_dist):
    mean = np.mean(offspring_dist)
    variance = np.var(offspring_dist, ddof=1)
    k_estimate = mean ** 2 / (variance - mean)
    return round(k_estimate, 2)


def get_mean_and_std(data):
    mean_value = np.mean(data)
    std_dev = np.std(data, ddof=1)  # Sample standard deviation
    return {"mean": mean_value, "std_dev": std_dev, }


def generate_transmission_tree(size_sse, R0, k, n_generations):
    # Calculate the parameter for the negative binomial distribution
    p = k / (k + R0)
    sse_infectees = list(np.arange(1, size_sse + 1))
    offsprings = {0: set(sse_infectees)}
    current_infectors = sse_infectees
    susceptible = np.arange(size_sse + 1, 100_000)
    generations = [current_infectors + [0]]
    transmissions = [(0, ie) for ie in current_infectors]  # Initial number of people infected by superspreader 0

    for _ in range(n_generations):
        new_infections = rng.negative_binomial(n=k, p=p, size=len(current_infectors))
        total_new_infections = new_infections.sum()

        # check to prevent selecting more infectees than available in the susceptible population.
        if total_new_infections > len(susceptible):
            total_new_infections = len(susceptible)

        chosen_infectees = rng.choice(susceptible, size=total_new_infections, replace=False)
        susceptible = np.setdiff1d(susceptible, chosen_infectees, assume_unique=True)

        new_infectees = []
        new_transmissions = []

        idx = 0
        for infector, n in zip(current_infectors, new_infections):
            if idx + n > total_new_infections:
                n = total_new_infections - idx
            infectees = chosen_infectees[idx:idx + n]
            idx += n
            new_infectees.extend(infectees)
            new_transmissions.extend([(infector, ie) for ie in infectees])
            offsprings[infector] = set(infectees)

        transmissions.extend(new_transmissions)
        current_infectors = new_infectees
        generations.append(new_infectees)

    G = nx.DiGraph()
    G.add_edges_from(transmissions)
    return G, generations, offsprings


def plot_trans_tree_model():
    def update_plots(size_sse, r0, k, n_gene):
        tree, generations, offsprings = generate_transmission_tree(size_sse=size_sse, R0=r0,
                                                                   k=k, n_generations=n_gene)

        sns.set_style("white")
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params)
        sns.set_context("paper", font_scale=2)

        cmap = plt.get_cmap("hsv")
        # Create a mapping of nodes to colours, with the root node (0) set to red
        node_colors_map = {
            node: ("red" if node == 0 else cmap(i / len(generations)))
            for i, generation in enumerate(generations)
            for node in generation
        }

        fig = plt.figure(figsize=(12, 8))
        gs = grid.GridSpec(1, 2, width_ratios=[1, 3])

        # Offspring Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        offsprings_dist = [len(infectees) for infector, infectees in offsprings.items()]
        o_mean_std = get_mean_and_std(offsprings_dist)
        k = moment_estimate_k(offsprings_dist)

        ax1.hist(offsprings_dist, bins=range(0, max(offsprings_dist) + 1), density=True,
                 color="gray", edgecolor="black")
        ax1.set_title(f"Offspring Distribution\nPosteriors ($R_0$:{o_mean_std['mean']:.2f}, $k$:{k})")
        ax1.set_xlabel("Offsprings")
        ax1.set_ylabel("Probability")

        # Transmission Tree
        ax2 = fig.add_subplot(gs[:, 1])
        prob_node_colors = [node_colors_map.get(node, "black") for node in tree.nodes()]

        # Determine layers based on distance from root using single-source the shortest path
        layers = []
        distance_from_root = nx.single_source_shortest_path_length(tree, 0)
        max_distance = max(distance_from_root.values())

        for i in range(max_distance + 1):
            layer = [node for node, dist in distance_from_root.items() if dist == i]
            if layer:
                layers.append(layer)

        pos = nx.shell_layout(tree, nlist=layers)

        nx.draw(tree, pos, with_labels=False, arrowsize=10, node_color=prob_node_colors,
                node_size=[120 if node == 0 else 20 for node in tree.nodes()], ax=ax2)
        ax2.set_title(f"Transmission Tree\nPriors ($R_0$:{r0}, $k$:{k})")
        ax2.axis("equal")

        # Add a legend for the generation colors
        legend_patches = []
        for i, generation in enumerate(generations):
            if i == 0:
                color = "red"  # Red for the root node
                label = f"Gen {i} (Index)"
            else:
                color = cmap(i / len(generations))
                label = f"Gene {i}"
            legend_patches.append(patches.Patch(color=color, label=label))

        ax2.legend(handles=legend_patches, title="Generations", loc="upper center",
                   bbox_to_anchor=(0.5, -0.01), ncol=3, borderaxespad=0., frameon=False)

        plt.tight_layout()
        st.pyplot(plt)

    size_sse_values = st.sidebar.slider("Size of SSE", 10.0, 50.0, 30.0, 10.0)
    r0_values = st.sidebar.slider("Basic Reproductive Number (R0)", 0.1, 3.0, 1.5, 0.1)
    k_values = st.sidebar.slider("Overdispersion Parameter (K)", 0.1, 5.0, 0.5, 0.1)
    n_generations = st.sidebar.slider("Number of Infection Generations", 1, 10, 5, 1)

    update_plots(size_sse_values, r0_values, k_values, n_generations)


if __name__ == "__main__":
    st.title("Simulating Transmission Tree 🌴🌴🌴🌴🌴")
    st.write(
        "Use the sliders in the sidebar to adjust the parameters and observe the changes in the epidemic dynamics.")

    plot_trans_tree_model()

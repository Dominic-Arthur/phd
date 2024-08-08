import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
import networkx as nx
from Simulation import generate_transmission_tree, rng
from SimAnalysis import get_mean_and_std, moment_estimate_k
import matplotlib.patches as mpatches


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

        node_colors_map[0] = "red"

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
                label = f"Index (Gen {i})"
            else:
                color = cmap(i / len(generations))
                label = f"Gene {i}"
            legend_patches.append(mpatches.Patch(color=color, label=label))

        ax2.legend(handles=legend_patches, title="Generations", loc="upper center",
                   bbox_to_anchor=(0.5, -0.01), ncol=3, borderaxespad=0., frameon=False)

        plt.tight_layout()
        st.pyplot(plt)

    size_sse_values = st.sidebar.slider("Size of SSE", 10.0, 50.0, 30.0, 10.0)
    r0_values = st.sidebar.slider("Basic Reproductive Number (R0)", 0.1, 3.0, 1.5, 0.1)
    k_values = st.sidebar.slider("Overdispersion Parameter (K)", 0.1, 5.0, 0.5, 0.1)
    n_generations = st.sidebar.slider("Number of Infection Generations", 1, 10, 5, 1)

    update_plots(size_sse_values, r0_values, k_values, n_generations)


def plot_epi_time_scales_model(latent_period=4.0, incubation_period=5.0, betas=None, num_samples=1000):
    if betas is None:
        betas = {"noise_exp_inf": 5.0, "noise_inf_sym": 5.0, "scale_inf_trans": 2.0, "scale_sym_test": 1.0}

    def update_plots(lp, ip, noise_exp_inf, noise_inf_sym, scale_inf_trans, scale_sym_test,
                     alpha):
        inf_sym_mean = ip - lp
        if inf_sym_mean < 0:
            st.error("Latent Period should be less than the Incubation Period")
        else:
            noise_exp_inf += alpha / betas["noise_exp_inf"]
            noise_inf_sym += alpha / betas["noise_inf_sym"]
            scale_inf_trans += alpha / betas["scale_inf_trans"]
            scale_sym_test += alpha / betas["scale_sym_test"]

            exp_inf = rng.lognormal(mean=np.log(lp), sigma=np.log(noise_exp_inf), size=num_samples)
            inf_sym = exp_inf + rng.lognormal(mean=np.log(inf_sym_mean), sigma=np.log(noise_inf_sym), size=num_samples)
            inf_trans = exp_inf + rng.gamma(shape=4, scale=scale_inf_trans, size=num_samples)
            sym_test = rng.gamma(shape=3, scale=scale_sym_test, size=num_samples)
            inf_trest = inf_sym + sym_test

            trans_within_range = (inf_trans >= inf_sym) & (inf_trans <= inf_sym + 5)
            prop_within_range = np.mean(trans_within_range) * 100
            prop_trans_before_testing = np.mean((inf_trest >= inf_trans)) * 100
            prop_pre_sym_cases = np.mean((inf_trans <= inf_sym)) * 100

            st.write(f"**Average generation time**: {inf_trans.mean():.1f} days")
            st.write(f"**{prop_within_range:.2f}%** transmissions occurred within 5 days after onset of symptoms")
            st.write(f"**{prop_trans_before_testing:.2f}%** transmissions occurred before testing")
            st.write(f"**{prop_pre_sym_cases:.2f}%** pre-symptomatic transmissions")

            sns.set_style('white')
            custom_params = {"axes.spines.right": False, "axes.spines.top": False}
            sns.set_theme(style="ticks", rc=custom_params)
            sns.set_context("paper", font_scale=2)

            plt.figure(figsize=(10, 6))
            sns.kdeplot(exp_inf, color='orange', fill=True, label='Infectiousness')
            sns.kdeplot(inf_trans, color='green', fill=True, label='Transmissions')
            sns.kdeplot(inf_sym, color='blue', fill=True, label='Symptom Onset')
            sns.kdeplot(inf_trest, color='red', fill=True, label='Testing')
            plt.xlabel('Time From Exposure (Days)')
            plt.ylabel('Density')
            plt.title("Epidemic Time Scale Dynamics")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
            st.pyplot(plt)

    # Streamlit sliders
    latent_periods = st.sidebar.slider("Latent Period", 1.0, 10.0, latent_period, 0.1)
    incubation_periods = st.sidebar.slider("Incubation Period", 1.1, 15.0, incubation_period, 0.1)
    noise_exp_inf_values = st.sidebar.slider("Noise Exp_Inf", 1.01, 2.0, 1.01, 0.01)
    noise_inf_sym_values = st.sidebar.slider("Noise Inf_Sym", 1.01, 2.0, 1.01, 0.01)
    scale_inf_trans_values = st.sidebar.slider("Scale Inf_Trans", 0.0, 2.0, 0.0, 0.01)
    scale_sym_test_values = st.sidebar.slider("Scale Sym_Test", 0.0, 2.0, 0.0, 0.01)
    alpha_values = st.sidebar.slider("Alpha", 0.0, 2.0, 0.5, 0.01)

    update_plots(latent_periods, incubation_periods, noise_exp_inf_values,
                 noise_inf_sym_values, scale_inf_trans_values, scale_sym_test_values, alpha_values)


if __name__ == "__main__":
    st.title("Modelling Epidemic Time Scale Dynamics of Symptomatic Infections")
    st.write(
        "Use the sliders in the sidebar to adjust the parameters and observe the changes in the epidemic dynamics.")

    st.header("Transmission Tree 🌴🌴🌴🌴🌴")
    plot_trans_tree_model()

    st.divider()

    st.header("Epidemiological Data 📊📊📊📊📊")
    plot_epi_time_scales_model()

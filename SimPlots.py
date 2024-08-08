from SimAnalysis import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
import seaborn as sns
from sklearn.linear_model import LinearRegression
from itertools import combinations
from ipywidgets import widgets, interactive
from IPython.display import display, HTML


def print_formatted(text):
    html_text = f"<span style='font-size:20px; font-weight:bold;'>{text}</span>"
    display(HTML(html_text))


def plot_epi_time_scales_model(latent_period=4, incubation_period=5, betas=None, num_samples=1000):
    # Initialize betas with default values if not provided
    if betas is None:
        betas = {"noise_exp_inf": 5, "noise_inf_sym": 5, "scale_inf_trans": 2, "scale_sym_test": 1}

    # Function to update the plots based on parameter changes
    def update_plots(lp, ip, noise_exp_inf, noise_inf_sym, scale_inf_trans, scale_sym_test, alpha):
        inf_sym_mean = ip - lp
        if inf_sym_mean < 0:
            print_formatted("Latent Period should be less than the Incubation Period")
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

            print()
            print_formatted(f">> Average generation time: {inf_trans.mean():.1f} days")
            print_formatted(f">> {prop_within_range:.2f}% transmissions occurred within 5 days after onset of symptoms")
            print_formatted(f">> {prop_trans_before_testing:.2f}% transmissions occurred before testing")
            print_formatted(f">> {prop_pre_sym_cases:.2f}% pre-symptomatic transmissions")
            print()

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
            plt.title("Epidemic Time Scale Dynamics of Symptomatic Infections")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
            plt.show()

    # Create interactive widgets
    slider_layout = widgets.Layout(width='500px', height='40px',
                                   margin='10px 0px 10px 0px',
                                   padding='0px 5px 0px 5px')

    latent_period_slider = widgets.FloatSlider(value=latent_period, min=1, max=10, step=0.1,
                                               description='Latent Period', style={'description_width': 'initial'},
                                               layout=slider_layout)
    incubation_period_slider = widgets.FloatSlider(value=incubation_period, min=1.1, max=15, step=0.1,
                                                   description='Incubation Period',
                                                   style={'description_width': 'initial'},
                                                   layout=slider_layout)
    noise_exp_inf_slider = widgets.FloatSlider(value=1.01, min=1.01, max=2.0, step=0.01,
                                               description='Noise Exp_Inf', style={'description_width': 'initial'},
                                               layout=slider_layout)
    noise_inf_sym_slider = widgets.FloatSlider(value=1.01, min=1.01, max=2.0, step=0.01,
                                               description='Noise Inf_Sym', style={'description_width': 'initial'},
                                               layout=slider_layout)

    scale_inf_trans_slider = widgets.FloatSlider(value=0.0, min=0.0, max=2.0, step=0.01,
                                                 description='Scale Inf_Trans', style={'description_width': 'initial'},
                                                 layout=slider_layout)
    scale_sym_test_slider = widgets.FloatSlider(value=0.0, min=0.0, max=2.0, step=0.01,
                                                description='Scale Sym_Test', style={'description_width': 'initial'},
                                                layout=slider_layout)

    alpha_slider = widgets.FloatSlider(value=0.5, min=0.0, max=2.0, step=0.01,
                                       description='Alpha', style={'description_width': 'initial'},
                                       layout=slider_layout)

    # Display interactive plot
    interactive_plot = interactive(update_plots, lp=latent_period_slider, ip=incubation_period_slider,
                                   noise_exp_inf=noise_exp_inf_slider, noise_inf_sym=noise_inf_sym_slider,
                                   scale_inf_trans=scale_inf_trans_slider, scale_sym_test=scale_sym_test_slider,
                                   alpha=alpha_slider)

    return interactive_plot


def plot_gamma_distribution(shape, scale=1, scale_range=(0.1, 3.0), range_oit=(0, 5),
                            num_points=100, num_runs=100):
    scale_values = np.linspace(scale_range[0], scale_range[1], num_points)
    proportions_within_ranges = np.zeros((num_runs, num_runs))
    variance_to_mean_ratios = np.zeros((num_runs, num_runs))

    # Compute the proportion of values within the specified range
    samples = rng.gamma(shape=shape, scale=scale, size=1000)
    within_range = (samples >= range_oit[0]) & (samples <= range_oit[1])
    proportion_within_range = np.mean(within_range) * 100

    print(f"Mean: {np.mean(samples):.2f}")
    print(f"Std: {np.std(samples):.2f}")
    print(f"{proportion_within_range:.2f}% of transmissions occurred within {range_oit[1]} days after onset"
          f" of infectiousness")

    for run in range(num_runs):
        variance_to_mean_ratio = []
        proportions_within_range = []
        for s in scale_values:
            gamma_data = rng.gamma(shape, s, size=1000)
            within_range = (gamma_data >= range_oit[0]) & (gamma_data <= range_oit[1])
            proportions_within_range.append(np.mean(within_range) * 100)
            calculated_mean = np.mean(gamma_data)
            calculated_variance = np.var(gamma_data)
            v_m_ratio = calculated_variance / calculated_mean
            variance_to_mean_ratio.append(v_m_ratio)
        variance_to_mean_ratios[run] = variance_to_mean_ratio
        proportions_within_ranges[run] = proportions_within_range

    # Calculate the mean and standard deviation across runs
    mean_proportions_within_ranges = np.mean(proportions_within_ranges, axis=0)
    std_proportions_within_ranges = np.std(variance_to_mean_ratios, axis=0)

    mean_variance_to_mean_ratios = np.mean(variance_to_mean_ratios, axis=0)
    std_variance_to_mean_ratios = np.std(variance_to_mean_ratios, axis=0)

    # Create the plot
    sns.set_style('white')
    # custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    # sns.set_theme(style="ticks", rc=custom_params)
    sns.set_context("paper", font_scale=2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f'Onset of Infectiousness to Transmission')

    axes[0].hist(samples, density=True, color='blue', edgecolor='black', bins=20)
    axes[0].set_xlabel(f"Days, (k={shape}, θ={scale})")
    axes[0].set_ylabel("Density")

    # Plot variance-to-mean ratio
    axes[1].errorbar(scale_values, mean_variance_to_mean_ratios, yerr=std_variance_to_mean_ratios, fmt='o',
                     linestyle='-', color='black', ecolor='gray', capsize=3, label='Mean (Std)')
    axes[1].set_xlabel('Noise (θ)')
    axes[1].set_ylabel('Variance to Mean Ratio')
    axes[1].legend(loc='lower center')
    axes[1].tick_params(axis='y')

    # Create a twin y-axis to plot the mean and proportion within range
    ax2 = axes[1].twinx()
    ax2.errorbar(scale_values, mean_proportions_within_ranges, yerr=std_proportions_within_ranges, fmt='o',
                 linestyle='-', color='r', ecolor='gray', capsize=3, label='Mean (Std)')
    ax2.set_ylabel(f'% of Transmissions Within \n{range_oit[1]} Days Postonset', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='upper center')
    fig.tight_layout()
    plt.show()


def plot_lognormal_distribution(mean, sigma=1, sigma_range=(1, 2), num_points=100, num_runs=100,
                                title="Latent Period Distribution"):
    # Define a range of noise values
    noise_values = np.linspace(sigma_range[0], sigma_range[1], num_points)
    variance_to_mean_ratios = np.zeros((num_runs, num_runs))

    sigma_symbol = "\u03C3"
    mu_symbol = "\u03BC"

    for run in range(num_runs):
        variance_to_mean_ratio = []
        for noise in noise_values:
            incubation_period = rng.lognormal(np.log(mean), np.log(noise), size=1000)
            calculated_mean = np.mean(incubation_period)
            calculated_variance = np.var(incubation_period)
            v_m_ratio = calculated_variance / calculated_mean
            variance_to_mean_ratio.append(v_m_ratio)
        variance_to_mean_ratios[run] = variance_to_mean_ratio

    # Calculate the mean and standard deviation across runs
    variance_to_mean_ratio_means = np.mean(variance_to_mean_ratios, axis=0)
    variance_to_mean_ratio_stds = np.std(variance_to_mean_ratios, axis=0)

    samples = rng.lognormal(np.log(mean), np.log(sigma), size=1000)

    # Plot the results
    sns.set_style('white')
    # custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    # sns.set_theme(style="ticks", rc=custom_params)
    sns.set_context("paper", font_scale=2)

    # Plot the lognormal distribution

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'{title}')

    axes[0].hist(samples, color='blue', density=True, edgecolor='black', bins=20)
    # axes[0].set_title(f'Noise={sigma})')
    axes[0].set_xlabel(f'Days, ({mu_symbol}={mean}, {sigma_symbol}={sigma})')
    axes[0].set_ylabel('Density')

    axes[1].errorbar(noise_values, variance_to_mean_ratio_means, yerr=variance_to_mean_ratio_stds,
                     fmt='o', linestyle='-', color='black', ecolor='gray', capsize=3, label='Mean (Std)')
    # axes[1].set_title(f'Effect of Noise')
    axes[1].set_xlabel(f'Noise ({sigma_symbol})')
    axes[1].set_ylabel('Variance to Mean Ratio')
    axes[1].legend(loc='best')

    fig.tight_layout()
    plt.show()


def plot_poisson_subs_model(clock_rate=(1e-3 / 365) * 29903, num_runs=100):
    """
        Plots the mean number of substitutions as a Poisson process over a range of evolution periods,
        with standard deviation error bars, and fits a straight line to the mean values.

        Parameters:
        - clock_rate (float): The rate parameter for the Poisson process.
        - num_runs (int): The number of runs to average over.
    """
    periods = np.arange(0, 101)

    # Store the results of multiple runs
    all_draws = np.zeros((num_runs, 101))
    for run in range(num_runs):
        all_draws[run] = [rng.poisson(lam=clock_rate * period) for period in periods]

    # Calculate the mean and standard deviation across runs
    mean_draws = np.mean(all_draws, axis=0)
    std_draws = np.std(all_draws, axis=0)

    # Fit a linear regression model to the mean values
    model = LinearRegression()
    model.fit(periods.reshape(-1, 1), mean_draws)
    fitted_line = model.predict(periods.reshape(-1, 1))
    gradient = model.coef_[0]

    # Plot the results
    sns.set_style('white')
    # custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    # sns.set_theme(style="ticks", rc=custom_params)
    sns.set_context("paper", font_scale=2)

    plt.figure(figsize=(10, 6))
    plt.errorbar(periods, mean_draws, yerr=std_draws, fmt='o', linestyle='-', color='black', ecolor='r', capsize=3,
                 label='Mean (Std)')
    plt.plot(periods, fitted_line, linestyle='--', color='g', label=f'Fitted Gradient = {gradient:.2f} '
                                                                    f'(Clock Rate = {clock_rate:.2f})')
    plt.title('Observable Number of Substitutions as Poisson Process')
    plt.xlabel("Evolution Period, Days")
    plt.ylabel("Substitutions")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_probabilities_heatmap(sim):
    sns.set_style('white')
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    sns.set_context("paper", font_scale=2)

    result = sim.result

    filtered_data = result.pairwise_data[(result.pairwise_data["TemporalDistance"] <= 15) &
                                         (result.pairwise_data["HammingDistance"] <= 10)].copy()

    filtered_data["Probability"] = filtered_data["Probability"].round(decimals=2)

    heatmap_data = filtered_data.pivot_table(index='HammingDistance', columns='TemporalDistance', values='Probability')

    fontsize = 25
    plt.figure(figsize=(20, 6))
    heatmap = sns.heatmap(heatmap_data, cmap='inferno', annot=True)
    heatmap.figure.axes[-1].set_ylabel('Likelihood', size=fontsize)
    plt.gca().invert_yaxis()  # Reverse the order of y-axis labels
    plt.xlabel('Sampling Interval (Days)', fontsize=fontsize)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('SNPs', fontsize=fontsize)
    plt.show()


def visualize_trans_tree(sim):
    sns.set_style("white")
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    sns.set_context("paper", font_scale=2)

    cmap = plt.get_cmap("hsv")
    # Create a mapping of nodes to colours, with the root node (0) set to red
    node_colors_map = {
        node: ("red" if node == 0 else cmap(i / len(sim.generations)))
        for i, generation in enumerate(sim.generations)
        for node in generation
    }

    node_colors_map[0] = "red"

    fig = plt.figure(figsize=(12, 8))
    gs = grid.GridSpec(1, 2, width_ratios=[1, 3])

    # Offspring Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    offsprings = sim.offsprings
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
    prob_node_colors = [node_colors_map.get(node, "black") for node in sim.tree.nodes()]
    pos = nx.nx_agraph.graphviz_layout(sim.tree, prog="twopi")
    nx.draw(sim.tree, pos, with_labels=False, arrowsize=10, node_color=prob_node_colors,
            node_size=[120 if node == 0 else 20 for node in sim.tree.nodes()], ax=ax2)
    ax2.set_title(f"Transmission Tree\nPriors ($R_0$:{sim.R0}, $k$:{sim.k})")
    ax2.axis("equal")

    plt.tight_layout()
    plt.show()


def visualize_networks(sim, p=0.5, snp=1):
    sns.set_style('white')
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    sns.set_context("paper", font_scale=2)

    prob_net = sim.probability_network(p)
    snp_net = sim.snp_network(snp)

    cmap = plt.get_cmap('hsv')
    node_colors_map = {
        node: ("red" if node == 0 else cmap(i / len(sim.generations)))
        for i, generation in enumerate(sim.generations)
        for node in generation
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 16))

    prob_net_color = [node_colors_map[node] for node in prob_net.nodes()]
    prob_pos = nx.nx_agraph.graphviz_layout(prob_net, prog="twopi")
    nx.draw(prob_net, prob_pos, with_labels=False, ax=axes[0], arrowsize=10, node_color=prob_net_color,
            node_size=[120 if node == 0 else 20 for node in prob_net.nodes()])
    axes[0].set_title(f"Probability Network (threshold = {p * 100:.2f}%)")
    axes[0].axis("equal")

    snp_node_colors = [node_colors_map.get(node, 'black') for node in snp_net.nodes()]
    snp_pos = nx.nx_agraph.graphviz_layout(snp_net, prog="twopi")
    nx.draw(snp_net, snp_pos, with_labels=False, arrowsize=10, node_color=snp_node_colors,
            node_size=[120 if node == 0 else 20 for node in snp_net.nodes()], ax=axes[1])
    axes[1].set_title(f"SNP Network (threshold = {snp} SNP)")
    axes[1].axis("equal")

    plt.tight_layout()
    plt.show()


def plot_epi_data(sim):
    sns.set_style("white")
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    sns.set_context("paper", font_scale=2)

    result = sim.result
    share_parent_df = result.pairwise_data[result.pairwise_data.ShareParent == 1]
    trans_pair_data = result.pairwise_data[result.pairwise_data.IsTransPair == 1]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 10))

    # Plot sampling_interval
    s_mean_std = get_mean_and_std(trans_pair_data["TemporalDistance"])
    axes[0, 0].hist(trans_pair_data["TemporalDistance"], bins=20, color="purple", edgecolor="black")
    axes[0, 0].set_title(f"Sampling Interval\nMean: {s_mean_std['mean']:.2f} "
                         f"(Std: {s_mean_std['std_dev']:.2f})")
    axes[0, 0].set_xlabel("Days")
    axes[0, 0].set_ylabel("Transmission Pairs")

    sh_mean_std = get_mean_and_std(share_parent_df["TemporalDistance"])
    axes[1, 0].hist(share_parent_df["TemporalDistance"], bins=20, color="blue", edgecolor="black")
    axes[1, 0].set_title(f"Sampling Interval\nMean: {sh_mean_std['mean']:.2f} "
                         f"(Std: {sh_mean_std['std_dev']:.2f})")
    axes[1, 0].set_xlabel("Days")
    axes[1, 0].set_ylabel("Child Pairs")

    # Plot genetic_distances
    g_mean_std = get_mean_and_std(trans_pair_data["HammingDistance"])
    axes[0, 1].hist(trans_pair_data["HammingDistance"], bins=20, color="purple", edgecolor="black")
    axes[0, 1].set_title(f"Genetic Distances\nMean: {g_mean_std['mean']:.2f} "
                         f"(Std: {g_mean_std['std_dev']:.2f})")
    axes[0, 1].set_xlabel("SNPs")
    axes[0, 1].set_ylabel("Transmission Pairs")

    gs_mean_std = get_mean_and_std(share_parent_df["HammingDistance"])
    axes[1, 1].hist(share_parent_df["HammingDistance"], bins=20, color="blue", edgecolor="black")
    axes[1, 1].set_title(f"Genetic Distances\nMean: {gs_mean_std['mean']:.2f} "
                         f"(Std: {gs_mean_std['std_dev']:.2f})")
    axes[1, 1].set_xlabel("SNPs")
    axes[1, 1].set_ylabel("Child Pairs")

    identical_seqs = result.pairwise_data[result.pairwise_data.HammingDistance == 0]
    i_mean_std = get_mean_and_std(identical_seqs["TemporalDistance"])
    axes[1, 2].hist(identical_seqs["TemporalDistance"], bins=20, color="green", edgecolor="black")
    axes[1, 2].set_title(f"Life-span\nMean: {i_mean_std['mean']:.2f}"
                         f"(Std: {i_mean_std['std_dev']:.2f})")
    axes[1, 2].set_xlabel("Days")
    axes[1, 2].set_ylabel("Identical Pairs")

    # Plot generation_times
    generation_times = [np.abs(sim.tree.nodes[s]["exp_date"] - sim.tree.nodes[t]["exp_date"])
                        for s, t in sim.tree.edges]
    t_mean_std = get_mean_and_std(generation_times)

    axes[0, 2].hist(generation_times, bins=20, color="red", edgecolor="black")
    axes[0, 2].set_title(f"Generation Times\nMean: {t_mean_std['mean']:.2f} "
                         f"(Std: {t_mean_std['std_dev']:.2f})")
    axes[0, 2].set_xlabel("Days")
    axes[0, 2].set_ylabel("Transmission Pairs")

    plt.tight_layout()
    plt.show()


def plot_clustering_eva(eva_df):
    # Create a DataFrame for seaborn with metrics in a long format
    eva_df_long = eva_df.melt(id_vars="truth_size", var_name="Metric", value_name="Score")

    sns.set_style('white')
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    sns.set_context("paper", font_scale=2)
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=eva_df_long, x="truth_size", y="Score", hue="Metric", marker='o')

    # Add labels and title
    plt.xlabel("Truth Size")
    plt.ylabel("Scores")
    plt.legend(title="Metric")
    plt.legend(title="Metric", bbox_to_anchor=(0.5, -0.15), loc='upper center', borderaxespad=0., ncol=3)
    plt.show()


def plot_clustering_analysis(results, para_name):
    # Initialise lists to store metrics
    keys = []
    sigclus_size_means = []

    evalssecluster_qi_means = []
    evalssecluster_qi_stds = []
    evalssecluster_com_means = []
    evalssecluster_com_stds = []
    evalssecluster_pur_means = []
    evalssecluster_pur_stds = []

    evalclusters_sseoptparam_means_qi = []
    evalclusters_sseoptparam_means_com = []
    evalclusters_sseoptparam_means_pur = []
    evalclusters_sseoptparam_std_means_qi = []
    evalclusters_sseoptparam_std_means_com = []
    evalclusters_sseoptparam_std_means_pur = []

    for k in results.keys():
        if k != "SSEOptParams":
            keys.append(float(k))

            # Calculate metrics
            sim_result = results[k]

            # Mean SigClusSize
            sigclus_size_means.append(np.mean(sim_result["SigClusSize"]))

            # EvalSSECluster
            evalssecluster = sim_result["EvalSSECluster"].describe()
            evalssecluster_qi_means.append(evalssecluster["QI"].loc["mean"])
            evalssecluster_qi_stds.append(evalssecluster["QI"].loc["std"])
            evalssecluster_com_means.append(evalssecluster["Completeness"].loc["mean"])
            evalssecluster_com_stds.append(evalssecluster["Completeness"].loc["std"])
            evalssecluster_pur_means.append(evalssecluster["Purity"].loc["mean"])
            evalssecluster_pur_stds.append(evalssecluster["Purity"].loc["std"])

            # EvalClustersSSEOptParam
            evalclusters_sseoptparam = sim_result["EvalClustersSSEOptParam"]
            evalclusters_sseoptparam_means = evalclusters_sseoptparam[
                evalclusters_sseoptparam.index.get_level_values('Metric') == "mean"].mean()
            evalclusters_sseoptparam_std_means = evalclusters_sseoptparam[
                evalclusters_sseoptparam.index.get_level_values('Metric') == "std"].mean()

            evalclusters_sseoptparam_means_qi.append(evalclusters_sseoptparam_means["QI"])
            evalclusters_sseoptparam_means_com.append(evalclusters_sseoptparam_means["Completeness"])
            evalclusters_sseoptparam_means_pur.append(evalclusters_sseoptparam_means["Purity"])

            evalclusters_sseoptparam_std_means_qi.append(evalclusters_sseoptparam_std_means["QI"])
            evalclusters_sseoptparam_std_means_com.append(evalclusters_sseoptparam_std_means["Completeness"])
            evalclusters_sseoptparam_std_means_pur.append(evalclusters_sseoptparam_std_means["Purity"])

    print(f"Mean significant cluster size threshold {np.mean(sigclus_size_means)}")

    sns.set_style('white')
    sns.set_context("paper", font_scale=2)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Evaluating Only SSE Cluster
    axes[0].plot(keys, evalssecluster_qi_means, marker="o", linestyle="-", color="b", label="Mean QI")
    axes[0].fill_between(keys, np.array(evalssecluster_qi_means) - np.array(evalssecluster_qi_stds),
                         np.array(evalssecluster_qi_means) + np.array(evalssecluster_qi_stds), alpha=0.3, color="b",
                         label="±1 Std QI")

    axes[0].plot(keys, evalssecluster_com_means, marker="s", linestyle="--", color="r", label="Mean Completeness")
    # axes[0].fill_between(keys, np.array(evalssecluster_com_means) - np.array(evalssecluster_com_stds),
    #                      np.array(evalssecluster_com_means) + np.array(evalssecluster_com_stds), alpha=0.3,
    #                      color="r", label="±1 Std Completeness")

    axes[0].plot(keys, evalssecluster_pur_means, marker="^", linestyle=":", color="g", label="Mean Purity")
    # axes[0].fill_between(keys, np.array(evalssecluster_pur_means) - np.array(evalssecluster_pur_stds),
    #                      np.array(evalssecluster_pur_means) + np.array(evalssecluster_pur_stds),
    #                      alpha=0.3, color="g", label="±1 Std Purity")

    axes[0].set_title("Evaluating Only SSE Cluster")
    axes[0].set_xlabel(f"{para_name}")
    axes[0].set_ylabel("Values")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Evaluating Significant True Clusters
    axes[1].plot(keys, evalclusters_sseoptparam_means_qi, marker="o", linestyle="--", color="b", label="Mean QI")
    axes[1].fill_between(keys,
                         np.array(evalclusters_sseoptparam_means_qi) - np.array(evalclusters_sseoptparam_std_means_qi),
                         np.array(evalclusters_sseoptparam_means_qi) + np.array(evalclusters_sseoptparam_std_means_qi),
                         alpha=0.3, color="b", label="±1 Std QI")

    axes[1].plot(keys, evalclusters_sseoptparam_means_com, marker="s", linestyle="--", color="r",
                 label="Mean Completeness")
    # axes[1].fill_between(keys,
    #                 np.array(evalclusters_sseoptparam_means_com) - np.array(evalclusters_sseoptparam_std_means_com),
    #                 np.array(evalclusters_sseoptparam_means_com) + np.array(evalclusters_sseoptparam_std_means_com),
    #                 alpha=0.3, color="r",
    #                 label="±1 Std Completeness")

    axes[1].plot(keys, evalclusters_sseoptparam_means_pur, marker="^", linestyle="--", color="g",
                 label="Mean Purity")
    # axes[1].fill_between(keys,
    #                np.array(evalclusters_sseoptparam_means_pur) - np.array(evalclusters_sseoptparam_std_means_pur),
    #                np.array(evalclusters_sseoptparam_means_pur) + np.array(evalclusters_sseoptparam_std_means_pur),
    #                 alpha=0.3, color="g", label="±1 Std Purity")

    axes[1].set_title("Evaluating Significant True Clusters")
    axes[1].set_xlabel(f"{para_name}")
    axes[1].set_ylabel("Values")
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


def plot_clustering_analysis_vrm(result, para_name):
    # Initialise lists to store metrics
    keys = []

    evalssecluster_qi_ratios = []
    evalssecluster_com_ratios = []
    evalssecluster_pur_ratios = []

    evalclusters_sseoptparam_ratios_qi = []
    evalclusters_sseoptparam_ratios_com = []
    evalclusters_sseoptparam_ratios_pur = []

    for k in result.keys():
        if k != "SSEOptParams":
            keys.append(float(k))

            # Calculate metrics
            sim_result = result[k]

            # EvalSSECluster
            evalssecluster = sim_result["EvalSSECluster"].describe()
            evalssecluster_qi_ratios.append(
                evalssecluster["QI"].loc["std"] ** 2 / evalssecluster["QI"].loc["mean"])
            evalssecluster_com_ratios.append(
                evalssecluster["Completeness"].loc["std"] ** 2 / evalssecluster["Completeness"].loc["mean"])
            evalssecluster_pur_ratios.append(
                evalssecluster["Purity"].loc["std"] ** 2 / evalssecluster["Purity"].loc["mean"])

            # EvalClustersSSEOptParam
            evalclusters_sseoptparam = sim_result["EvalClustersSSEOptParam"]
            evalclusters_sseoptparam_means = evalclusters_sseoptparam[
                evalclusters_sseoptparam.index.get_level_values('Metric') == "mean"].mean()
            evalclusters_sseoptparam_stds = evalclusters_sseoptparam[
                evalclusters_sseoptparam.index.get_level_values('Metric') == "std"].mean()

            evalclusters_sseoptparam_ratios_qi.append(
                evalclusters_sseoptparam_stds["QI"] ** 2 / evalclusters_sseoptparam_means["QI"])
            evalclusters_sseoptparam_ratios_com.append(
                evalclusters_sseoptparam_stds["Completeness"] ** 2 / evalclusters_sseoptparam_means["Completeness"])
            evalclusters_sseoptparam_ratios_pur.append(
                evalclusters_sseoptparam_stds["Purity"] ** 2 / evalclusters_sseoptparam_means["Purity"])

    sns.set_style('white')
    sns.set_context("paper", font_scale=2)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Evaluating Only SSE Cluster
    axes[0].plot(keys, evalssecluster_qi_ratios, marker="o", linestyle="-", color="b", label="QI VMR")
    axes[0].plot(keys, evalssecluster_com_ratios, marker="s", linestyle="--", color="r", label="Completeness VMR")
    axes[0].plot(keys, evalssecluster_pur_ratios, marker="^", linestyle=":", color="g", label="Purity VMR")

    axes[0].set_title("Evaluating Only SSE Cluster")
    axes[0].set_xlabel(f"{para_name}")
    axes[0].set_ylabel("VMR")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Evaluating Significant True Clusters
    axes[1].plot(keys, evalclusters_sseoptparam_ratios_qi, marker="o", linestyle="--", color="b", label="QI VMR")
    axes[1].plot(keys, evalclusters_sseoptparam_ratios_com, marker="s", linestyle="--", color="r",
                 label="Completeness VMR")
    axes[1].plot(keys, evalclusters_sseoptparam_ratios_pur, marker="^", linestyle="--", color="g", label="Purity VMR")

    axes[1].set_title("Evaluating Significant True Clusters")
    axes[1].set_xlabel(f"{para_name}")
    axes[1].set_ylabel("VMR")
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


def plot_optimising_one_clustering_parameter(sim, truth_cluster, mode='fixed_prob', fixed_value=0.7):
    # Define default ranges for probability and resolution
    default_probs = np.linspace(1.0, 0.4, 10)
    default_resolutions = np.linspace(1.0, 0.1, 10)

    # Initialize variables
    best_quality = -1
    best_prob = None
    best_res = None

    if mode == 'fixed_prob':
        probs = np.array([fixed_value])
        resolutions = default_resolutions
    elif mode == 'fixed_res':
        probs = default_probs
        resolutions = np.array([fixed_value])
    else:
        raise ValueError("Mode must be either 'fixed_prob' or 'fixed_res'")

    # Create matrices to store metrics for different combinations of prob and res
    quality = []
    purity = []
    completeness = []

    for i, prob in enumerate(probs):
        for j, res in enumerate(resolutions):
            clusters = leiden_community_detection(sim, prob=prob, res=res)
            evaluation = evaluate_clustering(clusters, truth_cluster)
            qua = evaluation['QI']
            quality.append(qua)
            purity.append(evaluation['Purity'])
            completeness.append(evaluation['Completeness'])

            if qua > best_quality:
                best_quality = qua
                best_prob = prob
                best_res = res

    # Print the results for the highest quality index
    print(f"Highest QI: {best_quality:.2f} at Probability: {best_prob:.2f} and Resolution: {best_res:.2f}")

    # Plot the results
    sns.set_style('white')
    sns.set_context("paper", font_scale=2)

    plt.figure(figsize=(8, 5))
    plt.plot(probs if mode == "fixed_res" else resolutions, quality, label="QI", color="blue")
    plt.plot(probs if mode == "fixed_res" else resolutions, purity, label="Purity", color="red")
    plt.plot(probs if mode == "fixed_res" else resolutions, completeness, label="Completeness", color="green")
    plt.xlabel("Probability" if mode == "fixed_res" else "Resolution")
    plt.ylabel("Score")
    plt.axhline(y=best_quality, linestyle=":", color="blue")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.show()


def plot_clusters(clusters, sim):
    clusters_network = nx.Graph([edge for c in clusters for edge in combinations(c, 2)])
    clusters_network.add_nodes_from(sim.tree.nodes())

    cmap = plt.get_cmap('hsv')
    node_colors_map = {
        node: ("red" if node == 0 else cmap(i / len(sim.generations)))
        for i, generation in enumerate(sim.generations)
        for node in generation
    }

    colors = [node_colors_map[node] for node in clusters_network.nodes()]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    prob_pos = nx.nx_agraph.graphviz_layout(clusters_network, prog="twopi")
    nx.draw(clusters_network, prob_pos, with_labels=False, ax=ax, arrowsize=10, node_color=colors,
            node_size=[120 if node == 0 else 20 for node in clusters_network.nodes()])
    ax.set_title("Clusters")
    ax.axis("equal")
    plt.show()


def plot_optimising_clustering_parameters(sim):
    truth_cluster = set(range(sim.size_sse + 1))
    max_prob = max(sim.result.pairwise_data["Probability"])
    min_prob = min(sim.result.pairwise_data["Probability"])
    # Ranges should be from highest to lowest so that we get the highest quality at high probability and resolutions

    probs = np.linspace(max_prob, min_prob, 10)
    resolutions = np.linspace(0.1, 1.0, 10)

    # Create matrices to store metrics for different combinations of prob and res
    quality_matrix = np.zeros((len(probs), len(resolutions)))
    purity_matrix = np.zeros((len(probs), len(resolutions)))
    completeness_matrix = np.zeros((len(probs), len(resolutions)))

    for i, prob in enumerate(probs):
        for j, res in enumerate(resolutions):
            clusters = leiden_community_detection(sim, prob=prob, res=res)
            evaluation = evaluate_clustering(clusters, truth_cluster)
            quality = evaluation['QI']
            quality_matrix[i, j] = quality
            purity_matrix[i, j] = evaluation['Purity']
            completeness_matrix[i, j] = evaluation['Completeness']

    # Plot the results
    sns.set_style('white')
    sns.set_context("paper", font_scale=2)

    plt.figure(figsize=(15, 15))

    # Plot Quality Index heatmap
    plt.subplot(3, 1, 1)
    heatmap1 = sns.heatmap(quality_matrix, annot=True, fmt=".2f", cmap='viridis', xticklabels=np.round(resolutions, 2),
                           yticklabels=np.round(probs, 2))
    plt.xlabel('Resolution')
    plt.ylabel('Probability')
    plt.title('Quality Index')
    heatmap1.figure.axes[-1].set_ylabel('Quality Index')
    # plt.gca().invert_yaxis()

    # Plot Purity heatmap
    plt.subplot(3, 1, 2)
    heatmap2 = sns.heatmap(purity_matrix, annot=True, fmt=".2f", cmap='Blues', xticklabels=np.round(resolutions, 2),
                           yticklabels=np.round(probs, 2))
    plt.xlabel('Resolution')
    plt.ylabel('Probability')
    plt.title('Purity')
    heatmap2.figure.axes[-1].set_ylabel('Purity')
    # plt.gca().invert_yaxis()

    # Plot Completeness heatmap
    plt.subplot(3, 1, 3)
    heatmap3 = sns.heatmap(completeness_matrix, annot=True, fmt=".2f", cmap='Reds',
                           xticklabels=np.round(resolutions, 2),
                           yticklabels=np.round(probs, 2))
    plt.xlabel('Resolution')
    plt.ylabel('Probability')
    plt.title('Completeness')
    heatmap3.figure.axes[-1].set_ylabel('Completeness')
    # plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()


def evaluate_clustering(predictions, truth_cluster):
    # filter for clusters that at least on of its members belong to the truth set
    predicted = [clus for clus in predictions if len(clus.intersection(truth_cluster)) > 0]
    purity = cluster_purity(predicted, truth_cluster)
    completeness = cluster_completeness(predicted, truth_cluster)

    return {
        'Purity': purity,
        'Completeness': completeness,
        'QI': f1_score(purity, completeness),
    }


def main():
    pass


if __name__ == "__main__":
    main()

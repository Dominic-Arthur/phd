from Simulation import *
from scipy.stats import poisson
from tqdm import tqdm
import igraph as ig


def moment_estimate_k(offspring_dist):
    mean = np.mean(offspring_dist)
    variance = np.var(offspring_dist, ddof=1)
    k_estimate = mean ** 2 / (variance - mean)
    return round(k_estimate, 2)


def get_mean_and_std(data):
    mean_value = np.mean(data)
    std_dev = np.std(data, ddof=1)  # Sample standard deviation
    return {"mean": mean_value, "std_dev": std_dev, }


def get_truth_clusters(sim):
    offsprings = sim.offsprings
    truth_clusters = []
    for infector, infectees in offsprings.items():
        cluster = infectees
        cluster.add(infector)
        truth_clusters.append(cluster)
    return truth_clusters


def cluster_purity(predictions, truth_set):
    purities = []
    for s in predictions:
        if len(s) == 0:
            purities.append(0)
        else:
            purities.append(len(s.intersection(truth_set)) / len(s))
    return sum(purities) / len(purities) if purities else 0


def cluster_completeness(predictions, truth_set):
    scores = [len(s.intersection(truth_set)) / len(truth_set) for s in predictions]
    return max(scores) if scores else 0


def f1_score(purity, completeness):
    if purity + completeness == 0:
        return 0
    return 2 * (purity * completeness) / (purity + completeness)


def evaluate_clustering(predictions, truth_set):
    # filter for clusters that at least on of its members belong to the truth set
    predicted = [clus for clus in predictions if len(clus.intersection(truth_set)) > 0]
    purity = cluster_purity(predicted, truth_set)
    completeness = cluster_completeness(predicted, truth_set)

    return {
        'Purity': purity,
        'Completeness': completeness,
        'QI': f1_score(purity, completeness),
    }


def evaluate_clustering_all(predictions, truth_sets):
    metrics = []
    for t in truth_sets:
        metrics.append(evaluate_clustering(predictions, t))
    return pd.DataFrame(metrics)


def evaluate_stability_optimisation_method(sim, truth_cluster, runs=10):
    results = sc.parallelize(get_optimum_clustering_parameters, [(sim, truth_cluster)] * runs)
    return pd.DataFrame(results)


def evaluate_stability_of_clustering(sim, prob=0.54, res=0.5, runs=10):
    truth_cluster = set(range(sim.size_sse + 1))
    clusters = sc.parallelize(leiden_community_detection, [(sim, prob, res)] * runs)
    results = [evaluate_clustering(predicted, truth_cluster) for predicted in clusters]
    return pd.DataFrame(results)


def get_optimum_clustering_parameters_all_parallelize(sim, truth_sets):
    def get_parameters(t):
        return get_optimum_clustering_parameters(sim, t)

    parameters = sc.parallelize(get_parameters, truth_sets)
    return pd.DataFrame(parameters)


def leiden_community_detection(sim, snp=1, prob=0.65, res=0.5, weight="Probability"):
    """ Weight is either "Probability" or "HammingDistance" """
    if weight == "Probability":
        nx_graph = sim.probability_network(prob)
    else:
        nx_graph = sim.snp_network(snp)

    igraph = ig.Graph.from_networkx(nx_graph)
    partition = igraph.community_leiden(weights=f"{weight}", resolution=res, n_iterations=10)
    clusters = []
    for cluster_id, cluster in enumerate(partition):
        node_names = igraph.vs[cluster]["_nx_name"]
        clusters.append(set(node_names))
    return clusters


def get_optimum_clustering_parameters(sim, truth_cluster):
    max_prob = max(sim.result.pairwise_data["Probability"])
    min_prob = min(sim.result.pairwise_data["Probability"])

    # Ranges should be from highest to lowest so that we get the highest quality at high probability and resolutions
    probs = np.linspace(max_prob, min_prob, 10)
    resolutions = np.linspace(1.0, 0.1, 10)

    best_quality = -1
    best_prob = None
    best_res = None

    for i, prob in enumerate(probs):
        for j, res in enumerate(resolutions):
            clusters = leiden_community_detection(sim, prob=prob, res=res)
            evaluation = evaluate_clustering(clusters, truth_cluster)
            quality = evaluation['QI']

            if quality > best_quality:
                best_quality = quality
                best_prob = prob
                best_res = res

    return {
        "QI": best_quality,
        "Probability": best_prob,
        "Resolution": best_res,
    }


def significant_cluster_size_threshold(r0=1.5, percentile=99):
    poisson_dist = poisson(r0)
    return int(poisson_dist.ppf(percentile / 100.0))


def analyse_simulations_all_truth_clusters(simulations, para_name):
    timer = sc.timer()
    results = sc.objdict()
    total_sims = len(simulations)
    for k, sims in tqdm(simulations.items(), total=total_sims, desc="Processing Simulations"):
        sig_cluster_sizes = []
        sse_opt_params = []  # list of dictionaries
        eval_sse_cluster = []  # list of dictionaries
        eval_clusters_sse_opt_params = []  # list of dataframes
        sig_truth_opt_params = []  # list of dataframes
        eval_sse_cluster_sig_truth_opt_params = []  # list of dictionaries
        eval_clusters_sig_truth_opt_params = []  # list of dataframes

        for sim in sims:
            truth_clusters = get_truth_clusters(sim)  # returns a list of sets
            sig_cluster_size = significant_cluster_size_threshold(r0=sim.R0)  # returns an int
            sig_truth_clusters = [clus for clus in truth_clusters if len(clus) >= sig_cluster_size]
            sig_cluster_sizes.append(sig_cluster_size)

            # Optimise clustering parameters with only SSE true cluster
            sse_cluster = set(range(sim.size_sse))
            sse_param = get_optimum_clustering_parameters(sim, sse_cluster)  # returns a dictionary
            sse_opt_params.append(sse_param)
            predicted_clusters_with_sse_param = leiden_community_detection(sim,
                                                                           prob=sse_param["Probability"],
                                                                           res=sse_param["Resolution"])

            eval_sse_cluster.append(evaluate_clustering(predicted_clusters_with_sse_param,
                                                        sse_cluster))  # returns a dictionary

            # Evaluate significant truth clusters with SSE optimised parameters
            evaluate_predicted_sse_cluster_param = evaluate_clustering_all(predicted_clusters_with_sse_param,
                                                                           sig_truth_clusters)  # dataframe
            eval_clusters_sse_opt_params.append(evaluate_predicted_sse_cluster_param.describe())

            # Optimise clustering parameters with all true clusters
            parameters = get_optimum_clustering_parameters_all_parallelize(sim, sig_truth_clusters)
            sig_truth_opt_params.append(parameters.describe())
            predicted_clusters_mean_params = leiden_community_detection(sim,
                                                                        prob=parameters.mean()["Probability"],
                                                                        res=parameters.mean()["Resolution"])

            evaluate_clusters = evaluate_clustering_all(predicted_clusters_mean_params, sig_truth_clusters)  # dataframe
            eval_sse_cluster_sig_truth_opt_params.append(evaluate_clusters.describe())

            eval_clusters_sig_truth_opt_params.append(evaluate_clustering(predicted_clusters_mean_params,
                                                                          sse_cluster))  # returns a dictionary

        sse_opt_params_df = pd.DataFrame(sse_opt_params)
        sse_opt_params_df.rename_axis('Run', inplace=True)

        eval_sse_cluster_df = pd.DataFrame(eval_sse_cluster)
        eval_sse_cluster_df.rename_axis('Run', inplace=True)

        eval_clusters_sse_opt_params_df = pd.concat(eval_clusters_sse_opt_params,
                                                    keys=range(len(eval_clusters_sse_opt_params)))
        eval_clusters_sse_opt_params_df.index = eval_clusters_sse_opt_params_df.index.set_names(["Run", "Metric"])

        sig_truth_opt_params_df = pd.concat(sig_truth_opt_params,
                                            keys=range(len(sig_truth_opt_params)))
        sig_truth_opt_params_df.index = sig_truth_opt_params_df.index.set_names(["Run", "Metric"])

        eval_sse_cluster_sig_truth_opt_params_df = pd.concat(eval_sse_cluster_sig_truth_opt_params,
                                                             keys=range(len(eval_sse_cluster_sig_truth_opt_params)))
        eval_sse_cluster_sig_truth_opt_params_df.index = eval_sse_cluster_sig_truth_opt_params_df.index.set_names(
            ["Run", "Metric"])

        eval_clusters_sig_truth_opt_params_df = pd.DataFrame(eval_clusters_sig_truth_opt_params)
        eval_clusters_sig_truth_opt_params_df.rename_axis('Run', inplace=True)

        results[k] = sc.odict(
            Parameter=f"{para_name}: {float(k)}",
            SigClusSize=sig_cluster_sizes,
            SSEOptParams=sse_opt_params_df,
            EvalSSECluster=eval_sse_cluster_df,
            EvalClustersSSEOptParam=eval_clusters_sse_opt_params_df,
            SigTruthOptParams=sig_truth_opt_params_df,
            EvalClustersSigTruthOptParams=eval_sse_cluster_sig_truth_opt_params_df,
            EvalSSEClusterSigTruthOptParams=eval_clusters_sig_truth_opt_params_df,
            Simulations=sims,
        )

    sc.save(f"../OutputFiles/Simulations/{para_name}_simulations_analysed.obj", results)

    timer.toc()
    return results


def analyse_simulations(simulations, para_name):
    timer = sc.timer()
    results = sc.objdict()
    sse_cluster = set(range(simulations[0][0].size_sse))
    sse_param = get_optimum_clustering_parameters(simulations[0][0], sse_cluster)  # returns a dictionary
    results['SSEOptParams'] = sse_param

    def analyse_simulation(sim):
        res = {}
        truth_clusters = get_truth_clusters(sim)  # returns a list of sets
        sig_cluster_size = significant_cluster_size_threshold(r0=sim.R0)  # returns an int
        sig_truth_clusters = [clus for clus in truth_clusters if len(clus) >= sig_cluster_size]
        predicted_clusters_with_sse_param = leiden_community_detection(sim,
                                                                       prob=sse_param["Probability"],
                                                                       res=sse_param["Resolution"])

        eval_sse_clus = evaluate_clustering(predicted_clusters_with_sse_param, sse_cluster)  # returns a dictionary

        # Evaluate significant truth clusters with SSE optimised parameters
        evaluate_predicted_sse_cluster_param = evaluate_clustering_all(predicted_clusters_with_sse_param,
                                                                       sig_truth_clusters)  # dataframe

        res['SigClusterSize'] = sig_cluster_size
        res['EvalSSECluster'] = eval_sse_clus
        res['EvalClustersSSEOptParams'] = evaluate_predicted_sse_cluster_param.describe()

        return res

    for k, sims in simulations.items():
        result = sc.parallelize(analyse_simulation, sims, progress=True)

        sig_cluster_sizes = [res['SigClusterSize'] for res in result]
        eval_sse_cluster = [res['EvalSSECluster'] for res in result]
        eval_clusters_sse_opt_params = [res['EvalClustersSSEOptParams'] for res in result]

        eval_sse_cluster_df = pd.DataFrame(eval_sse_cluster)
        eval_sse_cluster_df.rename_axis('Run', inplace=True)

        eval_clusters_sse_opt_params_df = pd.concat(eval_clusters_sse_opt_params,
                                                    keys=range(len(eval_clusters_sse_opt_params)))
        eval_clusters_sse_opt_params_df.index = eval_clusters_sse_opt_params_df.index.set_names(["Run", "Metric"])

        results[k] = sc.odict(Parameter=f"{para_name}: {float(k)}",
                              SigClusSize=sig_cluster_sizes,
                              EvalSSECluster=eval_sse_cluster_df,
                              EvalClustersSSEOptParam=eval_clusters_sse_opt_params_df,
                              Simulations=sims,
                              )

    timer.toc()
    return results


def main():
    pass


if __name__ == "__main__":
    main()

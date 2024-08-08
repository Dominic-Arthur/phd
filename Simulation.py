import numpy as np
import pandas as pd
import sciris as sc
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import copy

rng = np.random.default_rng(12345)


def generate_transmission_tree(size_sse, R0, k, n_generations):
    # Calculate the parameter for the negative binomial distribution
    p = k / (k + R0)
    sse_infectees = np.arange(1, size_sse + 1)
    offsprings = {0: set(sse_infectees)}
    current_infectors = sse_infectees
    susceptible = np.arange(size_sse + 1, 100_000)
    generations = [current_infectors]
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


class SimTransmission:
    def __init__(self, size_sse, dur_sse, R0, k, n_generations, gen_length, clock_rate, incubation_period,
                 latent_period, generation_time, noise_exp_inf, prop_sampled, scale_sym_test, shape_sym_test,
                 noise_inf_sym, scale_inf_trans, shape_inf_trans, alpha, betas, base):

        self.size_sse = size_sse  # size of SSE (number of people infected by superspreader)
        self.dur_sse = dur_sse  # duration of the SSE
        self.R0 = R0  # basic reproduction number R0
        self.k = k  # over dispersion parameter
        self.T = generation_time
        self.n_generations = n_generations  # average number of generations in a transmission chain
        self.gen_length = gen_length  # length of the pathogen genome
        self.prop_sampled = prop_sampled  # proportion of cases tested/sampled

        self.clock_rate = clock_rate  # substitutions per genome per day
        self.base = base  # Whether to Poisson process in sequence evolution

        self.Ip = incubation_period
        self.Lp = latent_period
        self.inf_sym_mu = incubation_period - latent_period  # Onset of Infectiousness to Onset of Symptom
        self.alpha = alpha
        self.betas = betas
        self.noise_lp = noise_exp_inf + self.alpha / self.betas["noise_exp_inf"]
        self.inf_sym_noise = noise_inf_sym + self.alpha / self.betas["noise_inf_sym"]

        self.oit_scale = scale_inf_trans + self.alpha / self.betas["scale_inf_trans"]
        self.oit_shape = shape_inf_trans

        self.testing_scale = scale_sym_test + self.alpha / self.betas["scale_sym_test"]
        self.testing_shape = shape_sym_test

        tree, generations, offsprings = generate_transmission_tree(size_sse=self.size_sse, R0=self.R0,
                                                                   k=self.k, n_generations=self.n_generations)
        self.tree = tree
        self.offsprings = offsprings
        self.generations = generations

        self.result = None

    def __repr__(self):
        return (
            f"SimTransmission("
            f"size_sse={self.size_sse}, dur_sse={self.dur_sse}, R0={self.R0}, k={self.k}, "
            f"n_generations={self.n_generations}, gen_length={self.gen_length}, "
            f"clock_rate={self.clock_rate}, incubation_period={self.Ip}, latent_period={self.Lp}, "
            f"noise_exp_inf={self.noise_lp - self.alpha / 5}, prop_tested={self.prop_sampled}, "
            f"scale_sym_test={self.testing_scale - self.alpha}, shape_sym_test={self.testing_shape}, "
            f"noise_inf_sym={self.inf_sym_noise - self.alpha / 5}, scale_inf_trans={self.oit_scale - self.alpha / 2}, "
            f"shape_inf_trans={self.oit_shape}, alpha={self.alpha}, base={self.base})"
            f"result={self.result}"
        )

    def update_dependent_attributes(self):
        self.noise_lp = self.noise_lp + self.alpha / 5
        self.inf_sym_noise = self.inf_sym_noise + self.alpha / 5
        self.oit_scale = self.oit_scale + self.alpha / 2
        self.oit_shape = self.oit_shape
        self.testing_scale = self.testing_scale + self.alpha

    def update_attribute(self, attr_name, new_value):
        setattr(self, attr_name, new_value)
        if attr_name == "alpha":
            self.update_dependent_attributes()

    def run(self):
        global rng
        rng = np.random.default_rng(12345)
        result = self.generate_epi_data()

        pairwise_data = self.calculate_hamming_distance(result.sequences)

        pairwise_data["TemporalDistance"] = pairwise_data.apply(self.sampling_interval, axis=1,
                                                                G=self.tree)

        pairwise_data["IsTransPair"] = pairwise_data.apply(self.check_trans_pair, axis=1, G=self.tree)

        pairwise_data["BothSampled"] = pairwise_data.apply(self.check_both_sampled, axis=1, G=self.tree)

        child_parent_dict = {child: parent for parent, child in self.tree.edges}
        pairwise_data["ShareParent"] = pairwise_data.apply(self.share_parent, axis=1, c_p_dict=child_parent_dict)

        pairwise_data["Probability"] = self.calculate_log_likelihood(pairwise_data["HammingDistance"].values,
                                                                     pairwise_data["TemporalDistance"].values,
                                                                     T=self.T, clock_rate=self.clock_rate)

        generation_times = [np.abs(self.tree.nodes[s]["exp_date"] - self.tree.nodes[t]["exp_date"])
                            for s, t in self.tree.edges]
        result["generation_times"] = generation_times
        result["pairwise_data"] = pairwise_data

        self.update_attribute("result", result)
        return self

    def probability_network(self, threshold):
        edges_df = self.result.pairwise_data[self.result.pairwise_data["Probability"] >= threshold]
        probability_network = nx.from_pandas_edgelist(edges_df, source='SequenceID1', target='SequenceID2',
                                                      edge_attr=True, create_using=nx.Graph())

        probability_network.add_nodes_from(list(self.tree.nodes))
        return probability_network

    def snp_network(self, threshold):
        edges_df = self.result.pairwise_data[self.result.pairwise_data["HammingDistance"] <= threshold]

        snp_network = nx.from_pandas_edgelist(edges_df, source='SequenceID1', target='SequenceID2',
                                              edge_attr=True, create_using=nx.Graph())

        snp_network.add_nodes_from(list(self.tree.nodes))
        return snp_network

    def generate_epi_data(self):
        num_to_assign = int(self.prop_sampled * len(self.tree))
        nodes = list(self.tree.nodes)
        rng.shuffle(nodes)  # Shuffle nodes to ensure random assignment
        sampled_nodes = nodes[:num_to_assign]
        nucleotides = np.array(['A', 'T', 'C', 'G'])

        def mutate_sequence(period, source_seq):
            num_mutations = round(self.clock_rate * period) if self.base else rng.poisson(lam=self.clock_rate * period)
            mutation_positions = rng.choice(range(len(source_seq)), num_mutations, replace=False)
            mutated_sequence = source_seq.copy()
            for pos in mutation_positions:
                current_nucleotide = mutated_sequence[pos]
                new_nucleotide = rng.choice([n for n in nucleotides if n != current_nucleotide])
                mutated_sequence[pos] = new_nucleotide  # Replace nucleotides at mutation indices with random choices
            return mutated_sequence

        ref_seq = rng.choice(nucleotides, size=self.gen_length)
        sequences = {}
        generation_times = []
        index_exp_to_inf = rng.lognormal(np.log(self.Lp), np.log(self.noise_lp))
        index_exp_date = -index_exp_to_inf
        index_inf_sym = rng.lognormal(np.log(self.inf_sym_mu), np.log(self.inf_sym_noise))
        index_sym_test = rng.gamma(shape=self.testing_shape, scale=self.testing_scale)
        index_testing_date = index_inf_sym + index_sym_test
        evolve_period = np.abs(index_testing_date - index_exp_date)
        sequences[0] = mutate_sequence(evolve_period, ref_seq)

        # Iterate over edges in the DFS traversal
        for predecessor, successor in nx.dfs_edges(self.tree, source=0):
            self.tree.nodes[predecessor]["sampled"] = 1 if predecessor in sampled_nodes else 0
            self.tree.nodes[successor]["sampled"] = 1 if successor in sampled_nodes else 0

            if predecessor == 0:
                self.tree.nodes[predecessor]["exp_date"] = round(index_exp_date)
                self.tree.nodes[predecessor]["sample_date"] = round(index_testing_date)

                primary_exp_date = rng.choice(list(range(self.dur_sse)))
                primary_exp_to_inf = rng.lognormal(np.log(self.Lp), np.log(self.noise_lp))
                primary_inf_sym = rng.lognormal(np.log(self.inf_sym_mu), np.log(self.inf_sym_noise))
                primary_inf_trans = rng.gamma(shape=self.oit_shape, scale=self.oit_scale)
                primary_sym_test = rng.gamma(shape=self.testing_shape, scale=self.testing_scale)
                primary_testing_date = primary_exp_date + primary_exp_to_inf + primary_inf_sym + primary_sym_test
                self.tree.nodes[successor]["exp_date"] = round(primary_exp_date)
                self.tree.nodes[successor]["sample_date"] = round(primary_testing_date)
                self.tree.nodes[successor]["trans_date"] = round(primary_exp_date + primary_exp_to_inf +
                                                                 primary_inf_trans)

                evolve_period = primary_testing_date - index_exp_date
                generation_times.append(primary_exp_date - index_exp_date)
                sequences[successor] = mutate_sequence(evolve_period, ref_seq)
            else:
                suc_exp_date = self.tree.nodes[predecessor]["trans_date"]
                suc_exp_to_inf = rng.lognormal(np.log(self.Lp), np.log(self.noise_lp))
                suc_inf_sym = rng.lognormal(np.log(self.inf_sym_mu), np.log(self.inf_sym_noise))
                suc_inf_trans = rng.gamma(shape=self.oit_shape, scale=self.oit_scale)
                suc_sym_test = rng.gamma(shape=self.testing_shape, scale=self.testing_scale)
                suc_testing_date = suc_exp_date + suc_exp_to_inf + suc_inf_sym + suc_sym_test
                self.tree.nodes[successor]["exp_date"] = round(suc_exp_date)
                self.tree.nodes[successor]["sample_date"] = round(suc_testing_date)
                self.tree.nodes[successor]["trans_date"] = round(suc_exp_date + suc_exp_to_inf +
                                                                 suc_inf_trans)

                evolve_period = suc_testing_date - self.tree.nodes[predecessor]["exp_date"]
                generation_times.append(suc_exp_date - self.tree.nodes[predecessor]["exp_date"])
                sequences[successor] = mutate_sequence(evolve_period, sequences[predecessor])

        result = sc.objdict(sequences=sequences, ref_seq=ref_seq, generation_times=generation_times)
        return result

    @staticmethod
    def calculate_hamming_distance(sequences):
        """
        Calculate the Pairwise Hamming distance between sequences. Help from ChatGPT.

        :param sequences: Dictionary of sequences as np arrays
        :return: A DataFrame with Pairwise Hamming distances between sequences
        """

        keys = list(sequences.keys())
        seq_matrix = np.array(list(sequences.values()))

        # Convert nucleotide sequences to numerical values for Hamming distance calculation
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        seq_matrix_num = np.vectorize(mapping.get)(seq_matrix)

        # Calculate the pairwise Hamming distances
        hamming_distances = pdist(seq_matrix_num, metric='hamming')
        hamming_matrix = squareform(hamming_distances) * seq_matrix.shape[1]  # convert to number of differing positions

        indices = np.triu_indices(len(keys), k=1)
        data = {
            "SequenceID1": np.array(keys)[indices[0]],
            "SequenceID2": np.array(keys)[indices[1]],
            "HammingDistance": hamming_matrix[indices]
        }

        return pd.DataFrame(data)

    @staticmethod
    def sampling_interval(row, G):
        date1 = G.nodes[row["SequenceID1"]]["sample_date"]
        date2 = G.nodes[row["SequenceID2"]]["sample_date"]
        return np.abs(date1 - date2)

    @staticmethod
    def share_parent(row, c_p_dict):
        return c_p_dict.get(row["SequenceID1"]) == c_p_dict.get(row["SequenceID2"])

    @staticmethod
    def check_both_sampled(row, G):
        node1 = row["SequenceID1"]
        node2 = row["SequenceID2"]
        return 1 if G.nodes[node1]["sampled"] and G.nodes[node2]["sampled"] else 0

    @staticmethod
    def check_trans_pair(row, G):
        node1 = row["SequenceID1"]
        node2 = row["SequenceID2"]
        return 1 if G.has_edge(node1, node2) or G.has_edge(node2, node1) else 0

    @staticmethod
    def calculate_log_likelihood(genetic_distance, temporal_distance, T, clock_rate):
        log_likelihood = np.exp(-((genetic_distance + (clock_rate * temporal_distance)) / T))
        return log_likelihood


def get_data(base_sim):
    return base_sim.run()


def simulate_data(i, base_sim, param_name, num_sims):
    base_sim_copy = copy.deepcopy(base_sim)
    base_sim_copy.update_attribute(param_name, i)
    base_sims = [base_sim_copy] * num_sims
    results = sc.parallelize(get_data, base_sims, progress=True)
    return results


def run_simulations(param_dict, param_name, param_values, num_sims=1):
    results = sc.objdict()
    timer = sc.timer()
    base_sim = SimTransmission(**param_dict)

    for i in param_values:
        result = simulate_data(i, base_sim, param_name, num_sims)  # list
        results[f"{i}"] = result

    print(f"Simulation Done\n"
          f"{param_name} : {param_values}\n")
    timer.toc()
    return results


def main():
    param_dict = {
        "size_sse": 30,
        "R0": 1.5,
        "k": 0.5,
        "n_generations": 5,
        "generation_time": 6,
        "gen_length": 1000,  # Didn't use actual genome length 29903 to reduce computational time
        "clock_rate": 0.08,  # (1e-3 / 365) * 29903 substitution rate (number of mutations per genome per year)
        "dur_sse": 1,
        "base": False,
        "prop_sampled": 0.5,

        "latent_period": 4,
        "incubation_period": 6,

        "noise_exp_inf": 1.1,
        "noise_inf_sym": 1.1,

        "shape_sym_test": 2,
        "scale_sym_test": 1,

        "shape_inf_trans": 4,
        "scale_inf_trans": 1,
        "alpha": 0.0,
        "betas": {"noise_exp_inf": 5, "noise_inf_sym": 5, "scale_inf_trans": 2, "scale_sym_test": 1}
    }

    name = "R0"
    param_range = [1.5]

    df_results = run_simulations(param_dict, name, param_range)
    print(df_results)


if __name__ == "__main__":
    main()

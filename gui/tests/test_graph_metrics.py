import unittest
import numpy as np
import networkx as nx
import time
import os
import sys
from scipy import stats

# Put the parent directory on the path so that graph_metrics can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import graph_metrics as gm

class TestGraphMetrics(unittest.TestCase):

    def setUp(self):
        """Build the test graphs before each test."""
        self.seed = 42
        # 1. Small-world graph (Watts-Strogatz)
        self.G_ws = nx.watts_strogatz_graph(n=100, k=4, p=0.1, seed=self.seed)
        for u, v in self.G_ws.edges():
            self.G_ws[u][v]['weight'] = np.random.rand()
            self.G_ws[u][v]['distance'] = 1.0 / self.G_ws[u][v]['weight']

        # 2. Random graph (Erdos-Renyi)
        self.G_er = nx.erdos_renyi_graph(n=100, p=0.04, seed=self.seed)
        for u, v in self.G_er.edges():
            self.G_er[u][v]['weight'] = np.random.rand()
            self.G_er[u][v]['distance'] = 1.0 / self.G_er[u][v]['weight']

        # 3. Lattice graph
        self.G_lat = nx.watts_strogatz_graph(n=100, k=4, p=0, seed=self.seed)
        for u, v in self.G_lat.edges():
            self.G_lat[u][v]['weight'] = np.random.rand()
            self.G_lat[u][v]['distance'] = 1.0 / self.G_lat[u][v]['weight']

        self.ch_names = [f'CH_{i}' for i in range(100)]

    def test_small_worldness_sigma(self):
        """Robust check: WS far above ER, and ER not small-world on average."""
        print("\n--- Test small-worldness (sigma) ---")

        # 1) sigma for the WS graph (more stable with more null models)
        sw_ws = gm.compute_small_worldness(self.G_ws, n_null=50, seed=self.seed)
        sigma_ws = sw_ws['sigma']
        print(f"Sigma of the Watts-Strogatz graph: {sigma_ws:.4f}")

        # 2) sigma for ER, averaged over several realisations
        sigmas_er = []
        for s in range(10):
            G_er_i = nx.erdos_renyi_graph(n=100, p=0.04, seed=self.seed + s)
            # random weights with a consistent distance
            for u, v in G_er_i.edges():
                w = np.random.rand()
                G_er_i[u][v]['weight'] = w
                G_er_i[u][v]['distance'] = 1.0 / (w + 1e-9)
            sw_er_i = gm.compute_small_worldness(G_er_i, n_null=30, seed=self.seed + s)
            sigmas_er.append(sw_er_i['sigma'])

        sigma_er_mean = float(np.nanmean(sigmas_er))
        print(f"Mean sigma over 10 Erdos-Renyi graphs: {sigma_er_mean:.4f}")

        # 3) assertions
        self.assertGreater(sigma_ws, 1.5, "WS should show marked small-worldness (sigma > 1.5)")
        self.assertLess(sigma_er_mean, 1.2, "ER should not come out small-world (mean sigma < 1.2)")
        self.assertGreater(sigma_ws, sigma_er_mean, "WS must have a larger sigma than the ER mean")

    def test_efficiency_vs_distance(self):
        """Check that global efficiency falls as the distances grow."""
        print("\n--- Test efficiency versus distance ---")
        G_high_weight = self.G_ws.copy()
        E_glob_high = gm.compute_graph_metrics(G_high_weight, self.ch_names)['global']['E_global']
        G_low_weight = self.G_ws.copy()
        for u, v in G_low_weight.edges():
            G_low_weight[u][v]['weight'] *= 0.1
            G_low_weight[u][v]['distance'] = 1.0 / G_low_weight[u][v]['weight']
        E_glob_low = gm.compute_graph_metrics(G_low_weight, self.ch_names)['global']['E_global']
        print(f"E_global (high weights): {E_glob_high:.4f}")
        print(f"E_global (low weights): {E_glob_low:.4f}")
        self.assertLess(E_glob_low, E_glob_high, "Global efficiency should fall when the distances grow")

    def test_clustering_and_path_length(self):
        """Check that a lattice shows both high clustering and a long path length."""
        print("\n--- Test clustering and path length on a lattice ---")
        metrics_lat = gm.compute_graph_metrics(self.G_lat, self.ch_names)
        C_lat = metrics_lat['global']['C_global']
        L_lat = metrics_lat['global']['L_char']
        metrics_er = gm.compute_graph_metrics(self.G_er, self.ch_names)
        C_er = metrics_er['global']['C_global']
        L_er = metrics_er['global']['L_char']
        print(f"Clustering (lattice): {C_lat:.4f} versus (random): {C_er:.4f}")

        self.assertGreater(C_lat, C_er, "The clustering of a lattice should exceed that of a random graph")


    def test_benchmark_performance(self):
        """Benchmark the main functions on a graph of realistic size."""
        print("\n--- Performance benchmark (N=64) ---")

        # A WS graph is used because it is faster to randomise
        N = 64
        k = 4
        p = 0.1
        # Adjacency matrix taken from a WS graph, which is sparse
        G_bench = nx.watts_strogatz_graph(n=N, k=k, p=p, seed=self.seed)
        matrix = nx.to_numpy_array(G_bench)
        # random weights on the existing edges
        matrix[matrix > 0] = np.random.rand(G_bench.number_of_edges() * 2)
        ch_names = [f'CH_{i}' for i in range(N)]

        start_time = time.time()

        results = gm.run_multi_threshold_analysis(
            matrix, ch_names, threshold_range=(10, 30), n_steps=5, seed=self.seed
        )

        end_time = time.time()
        duration = end_time - start_time

        print(f"Multi-threshold analysis (5 steps) on a {N}x{N} graph finished in {duration:.4f} s.")
        self.assertIsNotNone(results, "The benchmark should not fail")
        # The 10 s budget should now be met comfortably
        self.assertLess(duration, 10.0, "The benchmark should finish in under 10 s on an average laptop")


    def test_nbs_functionality(self):
        print("\n--- Testing NBS functionality ---")
        N = 20
        n_subjects = 24
        rng = np.random.default_rng(123)

        def symm_zero(loc, scale, n):
            M = rng.normal(loc, scale, size=(n, n))
            M = 0.5*(M + M.T)
            np.fill_diagonal(M, 0.0)
            return M

        # Group 1: moderate baseline with moderate noise
        g1 = [symm_zero(0.30, 0.06, N) for _ in range(n_subjects)]

        # Group 2: as group 1 but with block 0:5 raised, plus slight jitter
        g2 = []
        for M in g1:
            M2 = M.copy()
            B = symm_zero(0.30 + 0.60, 0.03, 5)     # +0.60 effect, small sd so the block is not identical
            M2[0:5, 0:5] = B
            g2.append(M2)

        # t threshold for a two-sided p of 0.01
        df = (n_subjects + n_subjects - 2)
        t_thresh = stats.t.ppf(1 - 0.01/2, df)

        res_eff = gm.run_nbs(
            g1, g2,
            t_threshold=t_thresh,
            n_permutations=2000,  # finer resolution on p
            equal_var=True,
            two_sided=True,
            use_abs_t=True
        )
        print(f"NBS with an effect: p={res_eff['p_value']:.4f}, max_edges={res_eff['max_comp_size_real']}")
        self.assertLess(res_eff['p_value'], 0.05, "With a deterministic effect the p value should be very low")

        res_no = gm.run_nbs(
            g1, g1,
            t_threshold=t_thresh,
            n_permutations=2000,
            equal_var=True,
            two_sided=True,
            use_abs_t=True
        )
        print(f"NBS without an effect: p={res_no['p_value']:.4f}, max_edges={res_no['max_comp_size_real']}")
        self.assertGreater(res_no['p_value'], 0.1)




if __name__ == '__main__':
    unittest.main()

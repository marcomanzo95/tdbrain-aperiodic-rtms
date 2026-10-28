# graph_metrics.py

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Sequence
import logging
import config as cfg
import warnings
from scipy.stats import t
# np.trapz was removed in NumPy 2.0 and renamed np.trapezoid; support both
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

# Set up a logger for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_graph(
    matrix: np.ndarray,
    threshold_perc: Optional[int],
    weight_handling: str,
    symmetrize: str = 'average',
    directed: bool = False
) -> Optional[nx.Graph]:
    """Build a NetworkX graph from an adjacency matrix."""
    A = matrix.copy()
    if weight_handling == 'abs':
        A = np.abs(A)
    elif weight_handling == 'zero_neg':
        A[A < 0] = 0
    
    if not directed and symmetrize == 'average':
        A = 0.5 * (A + A.T)
    elif not directed and symmetrize == 'max':
        A = np.maximum(A, A.T)
        
    np.fill_diagonal(A, 0)

    if threshold_perc is not None and 0 < threshold_perc < 100:
        if A.size > 0:
            # For directed graphs every weight counts; for undirected ones, only the upper triangle.
            target_edges = A[np.triu_indices_from(A, k=1)] if not directed else A[A > 0]
            if target_edges.size > 0:
                threshold_val = np.percentile(target_edges, 100 - threshold_perc)
                A[A < threshold_val] = 0
        else:
            return None

    create_using = nx.DiGraph if directed else nx.Graph
    G = nx.from_numpy_array(A, create_using=create_using)
    
    isolated_nodes = list(nx.isolates(G))
    if isolated_nodes:
        logging.info(f"Removing {len(isolated_nodes)} isolated nodes.")
        G.remove_nodes_from(isolated_nodes)
        
    if G.number_of_nodes() == 0:
        logging.warning("No connected node is left after thresholding.")
        return None
    return G

def disparity_filter(G: nx.Graph, alpha: float = 0.05) -> nx.Graph:
    """Apply the disparity filter to extract the statistical backbone of a graph."""
    B = G.copy()
    for i in B.nodes():
        k = B.degree(i)
        if k > 1:
            sum_w = sum(B.edges[i, j]['weight'] for j in B.neighbors(i))
            if sum_w > 0:
                for j in B.neighbors(i):
                    p_ij = B.edges[i, j]['weight'] / sum_w
                    p_value = (1.0 - p_ij)**(k - 1)
                    if p_value > alpha:
                        B.edges[i, j]['weight'] = 0
    
    edges_to_remove = [(u, v) for u, v, w in B.edges(data='weight') if w == 0]
    B.remove_edges_from(edges_to_remove)
    B.remove_nodes_from(list(nx.isolates(B)))
    logging.info(f"Disparity filter with alpha={alpha} produced a backbone with {B.number_of_nodes()} nodes and {B.number_of_edges()} edges.")
    return B

def compute_small_worldness(G: nx.Graph, n_null: int = 100, seed: int = 42) -> Dict:
    """Compute the small-world coefficient (sigma) and the associated z scores."""
    nan_results = {k: np.nan for k in ["sigma", "C_real", "L_real", "C_rand_mean", "C_rand_std", "L_rand_mean", "L_rand_std", "C_z", "L_z"]}
    if G.is_directed():
        logging.warning("Small-worldness is not canonically defined for directed graphs. Skipping.")
        return nan_results

    if nx.is_connected(G):
        G_main = G
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        if len(largest_cc) < 2: return nan_results
        G_main = G.subgraph(largest_cc)

    C_real = nx.average_clustering(G_main, weight='weight')
    L_real = nx.average_shortest_path_length(G_main, weight='distance')

    # 2. Generate the null models and compute C_rand and L_rand
    C_null, L_null = [], []
    rng = np.random.default_rng(seed)

    # Take the weights of the original graph in order to reshuffle them
    original_weights = [d['weight'] for u, v, d in G_main.edges(data=True)]

    for i in range(n_null):
        try:
            # Build a null graph that preserves the degree sequence
            G_null_base = nx.double_edge_swap(G_main.copy(), nswap=len(G_main.edges()) * 5, max_tries=5000, seed=int(rng.integers(1e9)))
            
            if not nx.is_connected(G_null_base):
                continue

            # --- Assign the original weights at random to the edges of the null graph ---
            rng.shuffle(original_weights)
            for i, (u, v) in enumerate(G_null_base.edges()):
                weight = original_weights[i]
                G_null_base[u][v]['weight'] = weight
                G_null_base[u][v]['distance'] = 1.0 / (weight + 1e-9) if weight > 0 else 1e9
            # --- end of the new logic ---

            C_null.append(nx.average_clustering(G_null_base, weight='weight'))
            L_null.append(nx.average_shortest_path_length(G_null_base, weight='distance'))

        except nx.NetworkXAlgorithmError as e:
            logging.warning(f"Swap attempt {i+1}/{n_null} failed: {e}. Skipping this null model.")
            continue

    if not C_null or not L_null:
        logging.error("No valid null graph could be generated for the sigma estimate.")
        return nan_results

    C_rand_mean, C_rand_std = np.mean(C_null), np.std(C_null)
    L_rand_mean, L_rand_std = np.mean(L_null), np.std(L_null)

    gamma = C_real / C_rand_mean if C_rand_mean > 0 else np.inf
    lambda_ = L_real / L_rand_mean if L_rand_mean > 0 else np.inf
    sigma = gamma / lambda_ if lambda_ > 0 else np.inf
    C_z = (C_real - C_rand_mean) / C_rand_std if C_rand_std > 0 else np.inf
    L_z = (L_real - L_rand_mean) / L_rand_std if L_rand_std > 0 else np.inf

    return {
        "sigma": sigma, "C_real": C_real, "L_real": L_real,
        "C_rand_mean": C_rand_mean, "C_rand_std": C_rand_std,
        "L_rand_mean": L_rand_mean, "L_rand_std": L_rand_std,
        "C_z": C_z, "L_z": L_z
    }

def compute_mst_metrics(matrix: np.ndarray) -> Optional[Dict]:
    """Build a minimum spanning tree (MST) and compute its metrics."""
    if matrix is None or matrix.shape[0] < 2: return None
    with np.errstate(invalid='ignore'):
        distance_matrix = 1.0 - np.abs(matrix)
    np.fill_diagonal(distance_matrix, 0)
    if np.any(~np.isfinite(distance_matrix)):
        distance_matrix[~np.isfinite(distance_matrix)] = np.nanmax(distance_matrix[np.isfinite(distance_matrix)])
    G_full = nx.from_numpy_array(distance_matrix)
    if not nx.is_connected(G_full):
        largest_cc = max(nx.connected_components(G_full), key=len)
        G_full = G_full.subgraph(largest_cc)
    mst = nx.minimum_spanning_tree(G_full, weight='weight')
    if mst.number_of_nodes() == 0: return None
    degrees = np.array([d for _, d in mst.degree()])
    leaf_fraction = float(np.sum(degrees == 1) / mst.number_of_nodes())
    diameter = float(nx.diameter(mst))
    mean_edge_weight = float(np.mean([d['weight'] for _, _, d in mst.edges(data=True)])) if mst.number_of_edges() > 0 else 0.0
    return {"mst_leaf_fraction": leaf_fraction, "mst_diameter": diameter, "mst_mean_edge_weight": mean_edge_weight}

def compute_consensus_modularity(G: nx.Graph, n_reps: int = 50, seed: int = 42) -> Dict:
    """Compute a consensus community partition by running Louvain repeatedly."""
    all_partitions, all_q = [], []
    rng = np.random.default_rng(seed)
    adj_matrix = nx.to_numpy_array(G)
    for i in range(n_reps):
        rep_seed = int(rng.integers(1e9))
        try:
            import bct
            ci, q = bct.modularity_louvain_und(adj_matrix, seed=rep_seed)
            all_partitions.append(ci)
            all_q.append(q)
        except ImportError:
            communities = nx.community.louvain_communities(G, weight='weight', seed=rep_seed)
            q = nx.community.modularity(G, communities, weight='weight')
            all_q.append(q)
            ci_map = {node: i for i, com in enumerate(communities) for node in com}
            all_partitions.append(np.array([ci_map.get(n, -1) for n in G.nodes()]))
    if not all_partitions:
        return {"Q_mean": np.nan, "Q_std": np.nan, "consensus_partition": [], "n_modules_consensus": np.nan}
    n_nodes = G.number_of_nodes()
    agreement_matrix = np.zeros((n_nodes, n_nodes))
    for partition in all_partitions:
        agreement_matrix += (partition[:, None] == partition)
    agreement_matrix /= n_reps
    try:
        import bct
        consensus_partition, consensus_q = bct.modularity_louvain_und(agreement_matrix, seed=seed)
    except ImportError:
        G_agreement = nx.from_numpy_array(agreement_matrix)
        communities = nx.community.louvain_communities(G_agreement, weight='weight', seed=seed)
        ci_map = {node: i for i, com in enumerate(communities) for node in com}
        consensus_partition = np.array([ci_map.get(n, -1) for n in G_agreement.nodes()])
    return {"Q_mean": np.mean(all_q), "Q_std": np.std(all_q), "consensus_partition": consensus_partition, "n_modules_consensus": len(np.unique(consensus_partition))}



def compute_graph_metrics(G: nx.Graph, ch_names: List[str], seed: int = 42) -> Optional[Dict]:
    """Compute a set of global and per-node graph-theoretical metrics."""
    if G is None or G.number_of_nodes() == 0:
        return None
    
    is_directed = G.is_directed()

    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 0)
        G[u][v]['distance'] = 1.0 / (weight + 1e-9) if weight > 0 else 1e9

    density = nx.density(G)
    if np.any(nx.to_numpy_array(G) < 0):
        logging.warning("The matrix carries negative weights, so the clustering coefficient may lose its canonical interpretation.")
    
    C_local = nx.clustering(G, weight='weight')
    C_global = np.mean(list(C_local.values()))
    
    n = len(G)
    E_global = 0.0
    if n > 1 and not is_directed:
        all_pairs_shortest_path = dict(nx.all_pairs_dijkstra_path_length(G, weight='distance'))
        inv_path_lengths = [1.0 / dist for dists in all_pairs_shortest_path.values() for dist in dists.values() if dist > 0]
        if inv_path_lengths: E_global = np.mean(inv_path_lengths)

    L_char = np.nan
    if not is_directed:
        if nx.is_connected(G):
            L_char = nx.average_shortest_path_length(G, weight='distance')
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            if len(largest_cc) > 1:
                sub_G = G.subgraph(largest_cc)
                L_char = nx.average_shortest_path_length(sub_G, weight='distance')
                logging.warning(f"Graph not connected. L_char computed on the largest component ({len(largest_cc)} nodes).")

    consensus_results = compute_consensus_modularity(G, n_reps=50, seed=seed)
    q = consensus_results["Q_mean"]
    ci = consensus_results["consensus_partition"]
    n_modules = consensus_results["n_modules_consensus"]

    node_indices = list(G.nodes())
    if is_directed:
        node_metrics = pd.DataFrame({
            'channel_idx': node_indices,
            'in_degree': [d for _, d in G.in_degree(weight='weight')],
            'out_degree': [d for _, d in G.out_degree(weight='weight')],
            'clustering': [C_local[n] for n in node_indices],
            'community': ci,
        })
    else:
        E_local_dict = {node: nx.global_efficiency(nx.ego_graph(G, node, radius=1, center=False)) for node in G.nodes()}
        node_metrics = pd.DataFrame({
            'channel_idx': node_indices,
            'degree': [d for _, d in G.degree(weight='weight')],
            'clustering': [C_local[n] for n in node_indices],
            'efficiency_local': [E_local_dict[n] for n in node_indices],
            'community': ci,
        })
    
    # map node indices back onto channel labels
    node_map = {i: name for i, name in enumerate(ch_names)}
    node_metrics['channel'] = node_metrics['channel_idx'].map(node_map)
    
    # Drop the auxiliary column and move 'channel' to the front
    final_cols = ['channel'] + [col for col in node_metrics.columns if col not in ['channel', 'channel_idx']]
    node_metrics = node_metrics[final_cols]
    
    sw_metrics = compute_small_worldness(G, n_null=100, seed=seed)
    
    global_metrics = {"Q": q, "n_modules": n_modules, "C_global": C_global, "E_global": E_global, "L_char": L_char, "density": density}
    global_metrics.update(sw_metrics)

    return {"global": global_metrics, "node": node_metrics}

def run_multi_threshold_analysis(
    matrix: np.ndarray,
    ch_names: List[str],
    threshold_range: Tuple[int, int] = cfg.THRESHOLD_RANGE_DEFAULT,
    n_steps: int = cfg.N_STEPS_DEFAULT,
    seed: int = cfg.DEFAULT_SEED
) -> Optional[Dict]:
    """Run the graph-theoretical analysis over a range of thresholds and integrate the results."""
    thresholds_grid = np.linspace(threshold_range[0], threshold_range[1], n_steps)
    all_metrics = []
    used_thresholds = []

    for perc in thresholds_grid:
        G = prepare_graph(matrix, threshold_perc=int(perc), weight_handling='abs', directed=False)
        if G:
            metrics = compute_graph_metrics(G, ch_names, seed=seed)
            if metrics:
                all_metrics.append(metrics['global'])
                used_thresholds.append(perc)

    if not all_metrics:
        logging.warning("No metric could be computed over the requested threshold range.")
        return None

    df_metrics = pd.DataFrame(all_metrics)
    used_thresholds = np.asarray(used_thresholds, dtype=float)

    integrated_metrics = {}
    for col in df_metrics.columns:
        if pd.api.types.is_numeric_dtype(df_metrics[col]):
            valid = ~df_metrics[col].isnull()
            if valid.sum() > 1:
                x_axis = (used_thresholds[valid] - used_thresholds.min()) / (used_thresholds.max() - used_thresholds.min() + 1e-12)
                y = df_metrics.loc[valid, col].to_numpy(dtype=float)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    integrated_metrics[f"{col}_auc"] = _trapezoid(y, x=x_axis)

    return {
        "integrated": integrated_metrics,
        "steps": df_metrics,
        "thresholds": used_thresholds,
    }


def export_graph_analysis_results(
    output_dir: str,
    subject_id: str,
    params: Dict,
    global_metrics: Dict,
    node_metrics: pd.DataFrame
):
    """
    Export the graph-theoretical results to JSON and CSV files.

    Args:
        output_dir (str): destination folder for the files.
        subject_id (str): identifier of the subject or of the analysis.
        params (Dict): parameters used for the analysis.
        global_metrics (Dict): the global metrics.
        node_metrics (pd.DataFrame): the per-node metrics.
    """
    try:
        # Create the output folder if it does not exist
        os.makedirs(output_dir, exist_ok=True)
        
        base_filename = f"{subject_id}_graph_metrics"
        
        # 1. Write the parameters and the global metrics to a JSON file
        full_report = {
            "analysis_parameters": params,
            "global_metrics": {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in global_metrics.items()}
        }
        json_path = os.path.join(output_dir, f"{base_filename}.json")
        with open(json_path, 'w') as f:
            json.dump(full_report, f, indent=4)
            
        # 2. Write the per-node metrics to a CSV file
        csv_path = os.path.join(output_dir, f"{base_filename}_nodal.csv")
        node_metrics.to_csv(csv_path, index=False, float_format='%.4f')
        
        logging.info(f"Results exported to: {output_dir}")
        return True
    except Exception as e:
        logging.error(f"Error while exporting the results: {e}")
        return False

def run_nbs(
    group1_matrices: List[np.ndarray],
    group2_matrices: List[np.ndarray],
    t_threshold: Optional[float] = None,
    p_threshold_dynamic: float = cfg.P_VALUE_NBS_DEFAULT,
    n_permutations: int = cfg.N_PERMUTATIONS_NBS,
    seed: int = cfg.DEFAULT_SEED,
    equal_var: bool = True,
    two_sided: bool = True,
    use_abs_t: bool = True,
    component_stat: str = "extent"
) -> Optional[Dict]:
    """Network-based statistic (NBS) between two groups of undirected matrices."""
    from scipy.stats import ttest_ind

    if not group1_matrices or not group2_matrices:
        logging.error("Both groups must contain at least one matrix.")
        return None

    n1, n2 = len(group1_matrices), len(group2_matrices)

    def _prep(M):
        M = 0.5 * (M + M.T)
        np.fill_diagonal(M, 0.0)
        return M.astype(float)

    g1 = [_prep(m) for m in group1_matrices]
    g2 = [_prep(m) for m in group2_matrices]

    g1_stack = np.stack(g1, axis=2)  # (N,N,n1)
    g2_stack = np.stack(g2, axis=2)  # (N,N,n2)
    N = g1_stack.shape[0]
    iu = np.triu_indices(N, k=1)

    # observed t values (upper triangle only)
    t_real, _ = ttest_ind(
        g1_stack[iu[0], iu[1], :],
        g2_stack[iu[0], iu[1], :],
        axis=1, equal_var=equal_var, nan_policy='omit'
    )
    t_real = np.nan_to_num(t_real)

    # t threshold
    if t_threshold is None:
        df = (n1 + n2 - 2) if equal_var else min(n1 - 1, n2 - 1)
        alpha = 0.01
        t_threshold = t.ppf(1 - (alpha/2 if two_sided else alpha), df)

    thr_vec = np.abs(t_real) if use_abs_t else t_real
    mask_vec = thr_vec > t_threshold

    A = np.zeros((N, N), dtype=bool)
    A[iu] = mask_vec
    A = np.logical_or(A, A.T)
    G_real = nx.from_numpy_array(A.astype(int))

    comps = [c for c in nx.connected_components(G_real) if len(c) > 1]
    if not comps:
        return {"p_value": 1.0, "max_comp_size_real": 0, "components": [], "t_threshold": float(t_threshold)}

    # edge -> absolute t map, for the intensity statistic
    t_abs = np.zeros((N, N))
    t_abs[iu] = np.abs(t_real)
    t_abs = t_abs + t_abs.T

    def comp_stat(G, nodes):
        if component_stat == "intensity":
            sub_edges = G.subgraph(nodes).edges()
            return float(sum(t_abs[u, v] for u, v in sub_edges))
        else:  # extent
            return G.subgraph(nodes).number_of_edges()

    stat_real = max(comp_stat(G_real, c) for c in comps)

    # permutazioni
    all_stack = np.concatenate([g1_stack, g2_stack], axis=2)
    rng = np.random.default_rng(seed)
    null_stats = []

    for _ in range(n_permutations):
        perm = rng.permutation(n1 + n2)
        g1p = all_stack[:, :, perm[:n1]]
        g2p = all_stack[:, :, perm[n1:]]
        tp, _ = ttest_ind(
            g1p[iu[0], iu[1], :],
            g2p[iu[0], iu[1], :],
            axis=1, equal_var=equal_var, nan_policy='omit'
        )
        tp = np.nan_to_num(tp)
        tp_thr = np.abs(tp) if use_abs_t else tp

        Ap = np.zeros((N, N), dtype=bool)
        Ap[iu] = tp_thr > t_threshold
        Ap = np.logical_or(Ap, Ap.T)
        Gp = nx.from_numpy_array(Ap.astype(int))
        comps_p = [c for c in nx.connected_components(Gp) if len(c) > 1]
        if comps_p:
            null_stats.append(max(comp_stat(Gp, c) for c in comps_p))
        else:
            null_stats.append(0.0)

    null_stats = np.asarray(null_stats, dtype=float)
    p_value = (np.sum(null_stats >= stat_real) + 1) / (n_permutations + 1)

    return {
        "p_value": float(p_value),
        "max_comp_size_real": (int(stat_real) if component_stat == "extent" else float(stat_real)),
        "components": [list(c) for c in comps],
        "t_threshold": float(t_threshold),
        "component_stat": component_stat
    }

if __name__ == '__main__':
    # --- Quick smoke test ---
    print("Running the smoke test for graph_metrics.py...")
    
    # 1. Create the test data
    N = 32
    np.random.seed(42)
    random_matrix = np.random.rand(N, N)
    channel_names = [f'CH_{i+1}' for i in range(N)]
    
    # 2. Test prepare_graph
    print("\n--- Testing prepare_graph ---")
    G_test = prepare_graph(
        matrix=random_matrix,
        threshold_perc=20,
        weight_handling='abs'
    )
    assert G_test is not None
    print(f"Graph prepared with {G_test.number_of_nodes()} nodes and {G_test.number_of_edges()} edges.")
    
    # 3. Test compute_graph_metrics (include STEP 1, 2, 4)
    print("\n--- Testing compute_graph_metrics ---")
    metrics_test = compute_graph_metrics(G_test, channel_names)
    assert metrics_test is not None
    print("\nGlobal metrics computed:")
    for key, val in metrics_test['global'].items():
        print(f"  - {key}: {val:.4f}")
    print("\nPer-node metrics computed (first 5 rows):")
    print(metrics_test['node'].head())
    
    # 4. Test run_multi_threshold_analysis (STEP 3)
    print("\n--- Testing run_multi_threshold_analysis ---")
    multi_thresh_results = run_multi_threshold_analysis(random_matrix, channel_names)
    assert multi_thresh_results is not None
    print("\nIntegrated metrics (AUC):")
    for key, val in multi_thresh_results['integrated'].items():
        print(f"  - {key}: {val:.4f}")

    # 5. Test compute_mst_metrics (STEP 5)
    print("\n--- Testing compute_mst_metrics ---")
    mst_results = compute_mst_metrics(random_matrix)
    assert mst_results is not None
    print("MST metrics computed:")
    for key, val in mst_results.items():
        print(f"  - {key}: {val:.4f}")

    # 6. Test disparity_filter (STEP 6)
    print("\n--- Testing disparity_filter ---")
    G_full = nx.from_numpy_array(np.abs(random_matrix)) # a dense graph for the test
    backbone_graph = disparity_filter(G_full, alpha=0.05)
    assert backbone_graph is not None
    print(f"Original graph: {G_full.number_of_edges()} edges. Backbone: {backbone_graph.number_of_edges()} edges.")
    assert backbone_graph.number_of_edges() < G_full.number_of_edges()

    print("\nSmoke test completed successfully.")

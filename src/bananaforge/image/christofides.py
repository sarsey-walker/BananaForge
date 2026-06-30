"""Christofides algorithm for height map generation."""

import os
import random

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from skimage.color import rgb2lab
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score


def _compute_distinctiveness(centroids: np.ndarray) -> np.ndarray:
    """Return the minimum inter‑centroid distance for every centroid."""
    dmat = cdist(centroids, centroids, metric="euclidean")
    np.fill_diagonal(dmat, np.inf)
    return dmat.min(axis=1)


def two_stage_weighted_kmeans(
    target_lab: np.ndarray,
    H: int,
    W: int,
    overcluster_k: int = 200,
    final_k: int = 16,
    beta_distinct: float = 1.0,
    random_state: int | None = None,
):
    """Segment *target_lab* (reshaped (N,3)) into *final_k* clusters using a
    two‑stage weighted K‑Means.  Returns (final_centroids, final_labels).

    The pixel‑level data are *only* used in stage‑1; stage‑2 runs on the much
    smaller set of stage‑1 centroids which makes this fast and memory‑friendly.
    """

    # Stage 1: heavy over‑segmentation so that even tiny colour modes appear.
    kmeans1 = MiniBatchKMeans(
        n_clusters=overcluster_k,
        random_state=random_state,
        max_iter=300,
        n_init=10,
    )
    labels1 = kmeans1.fit_predict(target_lab)
    centroids1 = kmeans1.cluster_centers_
    counts1 = np.bincount(labels1, minlength=overcluster_k).astype(np.float64)

    # Stage 2 weighting: size * (1 + beta * normalised distinctiveness)
    distinct = _compute_distinctiveness(centroids1)
    if distinct.max() > 0:
        distinct /= distinct.max()
    weights = counts1 * (1.0 + beta_distinct * distinct)

    # Weighted K‑Means on the centroid set.
    kmeans2 = KMeans(
        n_clusters=final_k,
        random_state=random_state,
        n_init="auto",
    )
    kmeans2.fit(centroids1, sample_weight=weights)
    centroids_final = kmeans2.cluster_centers_

    # Assign every pixel to its nearest final centroid.
    # Use chunks to keep memory bounded for very large images.
    chunk = 2**18  # about 256k pixels ≈ 768 kB of float32 per chunk
    labels_final = np.empty(target_lab.shape[0], dtype=np.int32)
    for start in range(0, target_lab.shape[0], chunk):
        end = start + chunk
        d = cdist(target_lab[start:end], centroids_final, metric="euclidean")
        labels_final[start:end] = np.argmin(d, axis=1)

    labels_final = labels_final.reshape(H, W)
    return centroids_final, labels_final


def build_distance_matrix(labs, nodes):
    """
    Given an array labs (with shape (N, dims)) and a list of node indices,
    return a distance matrix (NumPy array) of shape (len(nodes), len(nodes)).
    """
    pts = labs[nodes]  # extract only the points corresponding to nodes
    # Use cdist for fast vectorized distance computation.
    return cdist(pts, pts, metric="euclidean")


def sample_pixels_for_silhouette(labels, sample_size=5000, random_state=None):
    """
    Flatten the label map, draw at most sample_size random positions,
    and return their (index, label) pairs ready for silhouette_score.
    """
    rng = np.random.default_rng(random_state)
    flat = labels.reshape(-1)
    n = flat.shape[0]

    if n <= sample_size:
        idx = np.arange(n)
    else:
        idx = rng.choice(n, size=sample_size, replace=False)

    return idx, flat[idx]


def segmentation_quality(
    target_lab_reshaped, labels, sample_size=5000, random_state=None
):
    """
    Compute the silhouette coefficient on a random pixel subset.
    Works in Lab because `target_lab_reshaped` is already weighted Lab.
    """
    idx, lbl_subset = sample_pixels_for_silhouette(labels, sample_size, random_state)
    X_subset = target_lab_reshaped[idx]
    # In rare cases (k == 1) sklearn will raise; catch and return -1
    try:
        return silhouette_score(X_subset, lbl_subset, metric="euclidean")
    except ValueError:
        return -1.0


def matrix_to_graph(matrix, nodes):
    """
    Convert a 2D NumPy array (matrix) into a dictionary-of-dicts graph,
    where graph[u][v] = matrix[i][j] for u = nodes[i], v = nodes[j].
    """
    graph = {}
    n = len(nodes)
    for i in range(n):
        u = nodes[i]
        graph[u] = {}
        for j in range(n):
            v = nodes[j]
            if u != v:
                graph[u][v] = matrix[i, j]
    return graph


# --- Christofides Helpers (same as before) ---


class UnionFind:
    def __init__(self):
        self.parents = {}
        self.weights = {}

    def __getitem__(self, obj):
        if obj not in self.parents:
            self.parents[obj] = obj
            self.weights[obj] = 1
            return obj
        path = [obj]
        root = self.parents[obj]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def union(self, *objects):
        roots = [self[x] for x in objects]
        heaviest = max(((self.weights[r], r) for r in roots))[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest


def minimum_spanning_tree(G):
    tree = []
    subtrees = UnionFind()
    # Build list of edges from graph dictionary.
    edges = sorted((G[u][v], u, v) for u in G for v in G[u])
    for W, u, v in edges:
        if subtrees[u] != subtrees[v]:
            tree.append((u, v, W))
            subtrees.union(u, v)
    return tree


def find_odd_vertexes(MST):
    degree = {}
    for u, v, _ in MST:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    return [v for v in degree if degree[v] % 2 == 1]


def minimum_weight_matching(MST, G, odd_vert):
    odd_vertices = odd_vert.copy()
    random.shuffle(odd_vertices)
    while odd_vertices:
        v = odd_vertices.pop()
        best_u = None
        best_dist = float("inf")
        for u in odd_vertices:
            if G[v][u] < best_dist:
                best_dist = G[v][u]
                best_u = u
        MST.append((v, best_u, G[v][best_u]))
        odd_vertices.remove(best_u)


def find_eulerian_tour(MST, G):
    graph = {}
    for u, v, _ in MST:
        graph.setdefault(u, []).append(v)
        graph.setdefault(v, []).append(u)
    start = next(iter(graph))
    tour = []
    stack = [start]
    while stack:
        v = stack[-1]
        if graph[v]:
            w = graph[v].pop()
            graph[w].remove(v)
            stack.append(w)
        else:
            tour.append(stack.pop())
    return tour


def christofides_tsp(graph):
    MST = minimum_spanning_tree(graph)
    odd_vertices = find_odd_vertexes(MST)
    minimum_weight_matching(MST, graph, odd_vertices)
    eulerian_tour = find_eulerian_tour(MST, graph)
    seen = set()
    path = []
    for v in eulerian_tour:
        if v not in seen:
            seen.add(v)
            path.append(v)
    path.append(path[0])
    return path


def prune_ordering(ordering, labs, bg, fg, min_length=3, improvement_factor=1.5):
    """
    Iteratively remove clusters from the ordering if doing so significantly reduces
    the total Lab-space distance. Only clusters that produce an improvement greater
    than improvement_factor * (median gap) are removed.

    Parameters:
      ordering: list of cluster indices (the current ordering)
      labs: Lab-space coordinates (indexed by cluster index)
      bg: background anchor (never removed)
      fg: foreground anchor (never removed)
      min_length: minimum allowed length of ordering
      improvement_factor: factor multiplied by the median gap to decide if a cluster is an outlier

    Returns:
      A pruned ordering that hopefully removes only extreme outliers.
    """
    current_order = ordering.copy()
    while len(current_order) > min_length:
        total_dist = compute_ordering_metric(current_order, labs)
        gaps = np.linalg.norm(
            labs[current_order[1:]] - labs[current_order[:-1]], axis=1
        )
        median_gap = np.median(gaps)
        best_reduction = 0
        best_idx = -1

        for i in range(1, len(current_order) - 1):
            if current_order[i] in (bg, fg):
                continue

            test_order = current_order[:i] + current_order[i + 1 :]
            new_dist = compute_ordering_metric(test_order, labs)
            reduction = total_dist - new_dist
            if reduction > best_reduction:
                best_reduction = reduction
                best_idx = i

        if best_reduction > improvement_factor * median_gap:
            current_order.pop(best_idx)
        else:
            break
    return current_order


def create_mapping(final_ordering, labs, all_labels):
    """
    Creates a mapping from each cluster (from all_labels) to a value in [0,1].
    Clusters in final_ordering get evenly spaced values.
    For clusters that were pruned (i.e. not in final_ordering), assign the value
    of the nearest cluster in final_ordering (based on Lab-space distance).

    Parameters:
      final_ordering: list of cluster indices (after pruning)
      labs: array of Lab-space coordinates (indexed by cluster index)
      all_labels: sorted list of all unique clusters produced by KMeans

    Returns:
      mapping: a dict mapping each cluster label in all_labels to a float in [0,1].
    """
    mapping = {}
    n_order = len(final_ordering)
    # If there's only one cluster in final_ordering, assign 0.5
    if n_order == 1:
        for label in all_labels:
            mapping[label] = 0.5
        return mapping

    # Assign evenly spaced values for clusters in final_ordering.
    for i, cluster in enumerate(final_ordering):
        mapping[cluster] = i / (n_order - 1)

    # For clusters not in final_ordering, find the nearest cluster (in Lab space)
    # from final_ordering and use its mapping value.
    for label in all_labels:
        if label not in mapping:
            lab_val = labs[label]
            best_cluster = None
            best_dist = float("inf")
            for cl in final_ordering:
                d = np.linalg.norm(labs[cl] - lab_val)
                if d < best_dist:
                    best_dist = d
                    best_cluster = cl
            mapping[label] = mapping[best_cluster]
    return mapping


def create_height_map_from_mapping(all_labels, value_map, max_layers=None):
    """
    Creates a height map from cluster labels and value mapping.
    This replaces the height map creation part of the old create_mapping function.
    """
    # Apply this mapping to the label image
    H, W = all_labels.shape
    final_height_map = np.zeros((H, W), dtype=np.float32)
    for cluster_idx, value in value_map.items():
        final_height_map[all_labels == cluster_idx] = value

    # Scale the height map to the full range for sigmoid activation
    # Inverse of sigmoid: log(p / (1 - p))
    eps = 1e-7
    # Ensure values are properly bounded to avoid invalid log operations
    final_height_map = np.clip(final_height_map, eps, 1.0 - eps)
    final_height_map_logits = np.log(final_height_map / (1 - final_height_map)).astype(
        np.float32
    )

    # --- Initialize global_logits ---
    # For now, we'll create a simple cycling pattern for material assignments
    # This will be overridden by the material color mapping in run_init_threads
    # We need to return a placeholder that will be replaced
    if max_layers is not None:
        global_logits = np.zeros((max_layers, 1), dtype=np.float32)  # Use max_layers
    else:
        global_logits = np.zeros(
            (len(value_map), 1), dtype=np.float32
        )  # Use num_clusters

    return final_height_map_logits, global_logits


def tsp_order_christofides_path(nodes, labs, bg, fg):
    """
    Orders the given nodes (cluster indices) by solving the Traveling Salesperson
    Problem (TSP) using the Christofides algorithm.
    The path is constrained to start at `bg` and end at `fg`.
    """
    if len(nodes) < 2:
        return nodes

    # Build the distance matrix and graph for all nodes
    dist_matrix = build_distance_matrix(labs, nodes)
    graph = matrix_to_graph(dist_matrix, nodes)

    # Run Christofides to get a full tour
    path = christofides_tsp(graph)

    # --- Align the path to start with bg and end with fg ---
    # Find the indices of bg and fg in the path
    try:
        bg_idx = path.index(bg)
        fg_idx = path.index(fg)
    except ValueError:
        # If bg or fg are not in the path (should not happen with good clustering),
        # just return the unaligned path.
        return path[:-1]

    # Reorder the path based on the direction that places fg after bg
    if (bg_idx + 1) % (len(path) - 1) == fg_idx:
        # Path is already in the correct direction, just rotate it
        final_ordering = path[bg_idx:-1] + path[:bg_idx]
    else:
        # Path is in the reverse direction, so reverse and rotate
        reversed_path = path[::-1]
        bg_idx_rev = reversed_path.index(bg)
        final_ordering = reversed_path[bg_idx_rev:-1] + reversed_path[:bg_idx_rev]

    # Ensure fg is the very last element, not just after bg
    try:
        # Move fg to the end if it's not already there
        final_ordering.remove(fg)
        final_ordering.append(fg)
    except ValueError:
        pass  # fg was not in the list, which is odd but we can proceed

    return final_ordering


def compute_ordering_metric(ordering, labs):
    """
    Computes the total Euclidean distance of a path defined by an ordering
    of cluster centroids in Lab space. A lower score is better.
    """
    if len(ordering) < 2:
        return 0.0
    # Calculate the sum of distances between consecutive points in the ordering
    path_points = labs[ordering]
    distances = np.linalg.norm(path_points[1:] - path_points[:-1], axis=1)
    return np.sum(distances)


def interpolate_arrays(value_array_pairs, num_points):
    # Sort pairs by the value (first element in each pair)
    sorted_pairs = sorted(value_array_pairs, key=lambda x: x[0])

    values = np.array([p[0] for p in sorted_pairs])
    arrays = np.array([p[1] for p in sorted_pairs])

    # Create the target values for interpolation
    target_values = np.linspace(values.min(), values.max(), num_points)

    # Interpolate each column of the arrays
    interpolated_arrays = np.zeros((num_points, arrays.shape[1]))
    for i in range(arrays.shape[1]):
        interpolated_arrays[:, i] = np.interp(target_values, values, arrays[:, i])

    return interpolated_arrays.astype(np.float32)


def init_height_map(
    target,
    max_layers,
    h,  # unused here but preserved for API compatibility
    background_tuple,
    eps=1e-6,
    random_seed=None,
    lab_weights=(1.0, 1.0, 1.0),
    init_method="quantize_maxcoverage",
    cluster_layers=None,
    lab_space=True,
    material_colors=None,
):
    """
    Initializes a height map by segmenting the target image into clusters,
    ordering them with a TSP solver (Christofides), and mapping that order to height.

    Returns:
        A tuple of (height_map_logits, global_logits, ordering_metric, labels)
    """
    H, W, C = target.shape

    # --- Image Segmentation ---
    # Convert to Lab space for more perceptually uniform clustering
    target_lab = rgb2lab(target)
    target_lab_reshaped = target_lab.reshape(-1, 3)

    # Weight the Lab channels if specified
    wL, wa, wb = lab_weights
    target_lab_reshaped *= np.array([wL, wa, wb])

    if init_method == "kmeans":
        centroids, all_labels = two_stage_weighted_kmeans(
            target_lab_reshaped,
            H,
            W,
            overcluster_k=min(200, cluster_layers * 4),
            final_k=cluster_layers,
            random_state=random_seed,
        )
        labs = centroids
    elif init_method == "quantize_maxcoverage":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown init_method: {init_method}")

    # --- TSP Color Ordering ---
    # Find the cluster index for the background and foreground colors.
    # The "foreground" is the color in the image furthest from the background.
    bg_lab = rgb2lab(np.array(background_tuple).reshape(1, 1, 3) / 255.0).flatten()
    bg_lab *= np.array([wL, wa, wb])

    # Find the node (cluster index) closest to the background color
    bg_node = np.argmin(np.linalg.norm(labs - bg_lab, axis=1))

    # Find the node furthest from the background color to act as an anchor
    fg_node = np.argmax(np.linalg.norm(labs - bg_lab, axis=1))

    all_nodes = list(range(len(labs)))
    final_ordering = tsp_order_christofides_path(all_nodes, labs, bg_node, fg_node)

    # Prune outliers from the ordering
    final_ordering = prune_ordering(final_ordering, labs, bg_node, fg_node)

    # Get the unique clusters (should be 0...cluster_layers-1 ideally)
    unique_clusters = sorted(np.unique(all_labels))

    # Create the proper height mapping
    new_values = create_mapping(final_ordering, labs, unique_clusters)

    # Create height map from the mapping
    final_height_map, global_logits = create_height_map_from_mapping(
        all_labels, new_values, max_layers=max_layers
    )

    ordering_metric = compute_ordering_metric(final_ordering, labs)

    return (
        final_height_map,
        global_logits,
        ordering_metric,
        all_labels,
        labs,
        final_ordering,
        new_values,
    )


def run_init_threads(
    target,
    max_layers,
    h,  # unused but preserved for API compatibility
    background_tuple,
    eps=1e-6,
    random_seed=None,
    num_threads=64,
    init_method="kmeans",
    cluster_layers=None,
    material_colors=None,
):
    """
    Runs `init_height_map` in parallel across multiple threads (jobs) with
    different random seeds and returns the results from the best run.

    The best run is determined by the `ordering_metric` from `init_height_map`,
    which measures the total length of the TSP path through the color clusters.
    A shorter path is generally better.
    """

    # If cluster_layers is not specified, use max_layers
    if cluster_layers is None or cluster_layers <= 0:
        cluster_layers = max_layers

    print("Choosing best ordering with metric:", end=" ")
    if random_seed is None:
        random_seed = 0
    seeds = [random_seed + i for i in range(num_threads)]
    results = Parallel(n_jobs=os.cpu_count())(
        delayed(init_height_map)(
            target,
            max_layers,
            h,
            background_tuple,
            eps,
            seed,
            init_method=init_method,
            cluster_layers=cluster_layers,
            material_colors=material_colors,
        )
        for seed in seeds
    )

    # --- Select the best result ---
    # The metric is the total length of the TSP path through cluster centroids.
    # A shorter path means the colors are arranged more smoothly.
    best_result_metric = float("inf")
    best_result_idx = -1
    for i, res in enumerate(results):
        if res is None:
            continue
        _, _, metric, _, _, _, _ = res
        if metric < best_result_metric:
            best_result_metric = metric
            best_result_idx = i

    if best_result_idx == -1:
        raise RuntimeError(
            "All initialization threads failed. This may be due to an issue with "
            "the input image or clustering parameters."
        )

    print(f"{best_result_metric:.4f}")
    best_result = results[best_result_idx]

    height_map, global_logits, _, labels, labs, final_ordering, new_values = best_result
    print(f"Best result number of cluster layers: {len(np.unique(labels))}")

    if material_colors is not None:
        # Assign materials based on cluster colors
        num_layers = max_layers
        num_materials = len(material_colors)

        # Convert material colors to Lab space for better color matching
        from skimage.color import rgb2lab

        material_colors_lab = rgb2lab(material_colors.reshape(1, -1, 3)).reshape(-1, 3)

        # Create material assignment logits based on cluster-to-material matching
        global_logits_list = []

        # Get unique cluster labels
        unique_clusters = np.unique(labels)

        # Use the already computed height mapping from init_height_map

        for cluster_label in unique_clusters:
            # Get the Lab color for this cluster (already computed in two_stage_weighted_kmeans)
            cluster_lab = labs[cluster_label]

            # Find closest material in Lab space
            distances = np.linalg.norm(material_colors_lab - cluster_lab, axis=1)
            best_material_idx = np.argmin(distances)

            # Create one-hot assignment for this cluster
            material_logit = np.ones(num_materials, dtype=np.float32) * -1.0
            material_logit[best_material_idx] = 1.0

            # Use the MAPPED height value - this handles pruned clusters correctly
            height_value = new_values[cluster_label]

            global_logits_list.append((height_value, material_logit))

        # Sort by height value and interpolate to max_layers
        global_logits_list.sort(key=lambda x: x[0])
        global_logits = interpolate_arrays(global_logits_list, num_layers)
    else:
        # If no material colors provided, create a simple cycling pattern
        # Always use max_layers to ensure it's respected
        num_layers = max_layers
        # Default to 4 materials if none specified
        num_materials = 4
        material_assignment_logits = np.zeros(
            (num_layers, num_materials), dtype=np.float32
        )

        for i in range(num_layers):
            material_idx = i % num_materials
            material_assignment_logits[i, material_idx] = 1.0

        global_logits = material_assignment_logits

    return height_map, global_logits, labels

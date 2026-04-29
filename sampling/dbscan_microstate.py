#!/usr/bin/env python3
"""
dbscan_microstate.py

Pipeline Overview:
  1. Load multiple PDB structures and extract C-alpha coordinates.
  2. Compute a pairwise RMSD matrix to quantify structural similarity.
  3. Run DBSCAN clustering with varying RMSD cutoffs to identify stable clusters (microstates).
  4. Determine a single representative structure for each microstate cluster.
  5. Save representative structures and membership mappings for downstream analyses.

Requirements:
  - Python packages: MDAnalysis, joblib, sklearn, pymultinest, numpy

Example Usage:
    python dbscan_microstate.py "path_to_PDBs/*.pdb" --min-samples 4 --n-points 50 --desired-min-clusters 2 --desired-max-clusters 6
"""

# -------------------------
# Imports & Logging
# -------------------------
import os
import glob
import argparse
import json
import logging
from collections import defaultdict
from typing import List, Dict
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN

# Configure module-level logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# -------------------------
# Global Default Parameters
# -------------------------
DEFAULT_MIN_SAMPLES = 5               # DBSCAN minimum samples per cluster
DEFAULT_N_POINTS = 100                # Number of RMSD cutoffs to scan
DEFAULT_DESIRED_MIN_CLUSTERS = 2      # Lower bound for desired number of clusters
DEFAULT_DESIRED_MAX_CLUSTERS = 6      # Upper bound for desired number of clusters
DEFAULT_OUTPUT_DIR = "./combined_output"

# ============================================================================
# SECTION 1. RMSD Utility Functions
# ============================================================================
def load_ca_coords(pdb_files: List[str]) -> np.ndarray:
    """
    Load C-alpha coordinates from multiple PDB files.
    Args:
        pdb_files: List of PDB file paths.
    Returns:
        coords: Shape (N_structures, N_CA_atoms, 3)
    Raises:
        ValueError: If structures contain inconsistent atom counts.
    """
    coords_list = []
    atom_count = None
    for pdb in pdb_files:
        u = mda.Universe(pdb)
        ca = u.select_atoms("name CA")
        if atom_count is None:
            atom_count = len(ca)
            logger.debug("Detected %d CA atoms", atom_count)
        elif len(ca) != atom_count:
            raise ValueError(f"{pdb} has {len(ca)} CA atoms; expected {atom_count}.")
        coords_list.append(ca.positions.copy())
    return np.array(coords_list)


def pairwise_rmsd_upper(coords_array: np.ndarray, n_jobs: int = -1) -> np.ndarray:
    """
    Compute the upper-triangle of the RMSD matrix in parallel.
    RMSD measures structural similarity between pairs of structures.
    """
    N = coords_array.shape[0]
    if N < 2:
        return np.array([])
    triu = np.triu_indices(N, k=1)

    def _pair(i, j):
        return float(rms.rmsd(coords_array[i], coords_array[j], center=True))

    return np.array(Parallel(n_jobs=n_jobs)(delayed(_pair)(i, j) for i, j in zip(*triu)))


def upper_to_full_matrix(upper_rmsd: np.ndarray, N: int) -> np.ndarray:
    """
    Convert upper-triangle RMSD data to a full symmetric matrix.
    """
    mat = np.zeros((N, N), dtype=float)
    if upper_rmsd.size == 0:
        return mat
    tri = np.triu_indices(N, k=1)
    mat[tri] = upper_rmsd
    mat[(tri[1], tri[0])] = upper_rmsd
    return mat


def force_n_clusters(pdb_files):
    """
    Force each structure to its own microstate cluster.
    """
    return {f"ms_{i}": [i] for i in range(len(pdb_files))}


def force_two_clusters(pdb_files):
    """
    Special handling for the case where there are exactly two PDBs.
    Automatically assign each to its own cluster.
    """
    return {
        "ms_0": [0],
        "ms_1": [1]
    }

# ============================================================================
# SECTION 2. Clustering (DBSCAN)
# ============================================================================
def cluster_dbscan(rmsd_matrix: np.ndarray, rmsd_cutoff: float, min_samples: int = 2):
    """
    Run DBSCAN clustering using the precomputed RMSD matrix.
    """
    clustering = DBSCAN(eps=rmsd_cutoff, min_samples=min_samples, metric="precomputed")
    labels = clustering.fit_predict(rmsd_matrix)
    clusters = defaultdict(list)
    for idx, lab in enumerate(labels):
        clusters["noise" if lab == -1 else int(lab)].append(idx)
    return dict(clusters), labels


def summarize_clusters(clusters, labels, n_structures):
    """
    Summarize clustering results for logging and evaluation.
    Returns:
        {
          "n_clusters": Number of clusters (excluding noise),
          "cluster_sizes": List of cluster sizes,
          "noise": Number of noise points,
          "coverage": Fraction of points assigned to clusters
        }
    """
    cluster_sizes = [len(v) for k, v in clusters.items() if k != "noise"]
    coverage = float(np.sum(labels != -1) / n_structures) if n_structures > 0 else 0.0
    return {
        "n_clusters": len(cluster_sizes),
        "cluster_sizes": cluster_sizes,
        "noise": len(clusters.get("noise", [])),
        "coverage": coverage
    }


def find_cutoff(rmsd_matrix, n_points, min_samples, desired_min, desired_max):
    """
    Automatically search for the best RMSD cutoff for DBSCAN by scanning a range of cutoffs.
    """
    N = rmsd_matrix.shape[0]
    if N < 2:
        raise ValueError("At least 2 structures are required for clustering.")
    dists = rmsd_matrix[np.triu_indices_from(rmsd_matrix, k=1)]
    min_rmsd = float(np.min(dists))
    max_rmsd = float(np.max(dists))
    cutoffs = np.linspace(min_rmsd, max_rmsd, n_points) if max_rmsd > min_rmsd else np.array([min_rmsd])

    best_candidate = None
    best_score = None

    for cutoff in cutoffs:
        clusters, labels = cluster_dbscan(rmsd_matrix, rmsd_cutoff=cutoff, min_samples=min_samples)
        summary = summarize_clusters(clusters, labels, N)
        logger.info(
            "cutoff=%.3f -> %d clusters, coverage=%.1f%%",
            cutoff, summary["n_clusters"], summary["coverage"] * 100.0
        )

        if desired_min <= summary["n_clusters"] <= desired_max:
            return cutoff, clusters, labels

        dist = max(desired_min - summary["n_clusters"], summary["n_clusters"] - desired_max, 0)
        score = (dist, -summary["coverage"])
        if best_score is None or score < best_score:
            best_score, best_candidate = score, (cutoff, clusters, labels, summary)

    logger.warning(
        "No cutoff produced desired cluster range, using fallback cutoff %.3f",
        best_candidate[0]
    )
    return best_candidate[0], best_candidate[1], best_candidate[2]


def clusters_to_microstates(clusters, include_noise=False):
    """
    Convert DBSCAN clusters into a microstate dictionary.
    Each cluster is labeled as 'ms_<cluster_id>'.
    """
    microstates = {}
    for k, members in clusters.items():
        if k == "noise":
            if include_noise:
                microstates["ms_noise"] = list(members)
            continue
        microstates[f"ms_{k}"] = list(members)
    return microstates

# ============================================================================
# SECTION 3. Save Representative Structures
# ============================================================================
def save_microstate_representatives(microstates: Dict[str, List[int]], pdb_files: List[str], output_txt_path: str):
    """
    Select and save a single representative PDB path for each microstate.
    Representative = central member of the cluster.
    """
    with open(output_txt_path, 'w') as f:
        for cluster_id, indices in microstates.items():
            if not indices:
                continue
            # Choose central element (median index in sorted cluster list)
            central_idx = len(indices) // 2
            rep_idx = indices[central_idx]
            rep_path = pdb_files[rep_idx]
            f.write(rep_path + '\n')

    logger.info(f"Representative PDB paths saved to: {output_txt_path}")

# ============================================================================
# SECTION 4. Main Orchestration
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Cluster PDBs by RMSD and identify microstates.")
    parser.add_argument("pdb_files", nargs="+", help="PDB files or glob patterns (e.g. '*.pdb')")
    parser.add_argument("--outdir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES, help="DBSCAN minimum samples per cluster")
    parser.add_argument("--n-points", type=int, default=DEFAULT_N_POINTS, help="Number of RMSD cutoffs to scan")
    parser.add_argument("--desired-min-clusters", type=int, default=DEFAULT_DESIRED_MIN_CLUSTERS, help="Lower bound for desired cluster count")
    parser.add_argument("--desired-max-clusters", type=int, default=DEFAULT_DESIRED_MAX_CLUSTERS, help="Upper bound for desired cluster count")
    parser.add_argument("--force-n-states", type=int, default=None, help="If set and equal to the number of PDBs, force each structure into its own microstate.")

    args = parser.parse_args()
    force_n = args.force_n_states

    # Expand PDB file globs into full list
    pdb_files = sorted(set(sum([glob.glob(p) for p in args.pdb_files], [])))
    if not pdb_files:
        raise ValueError("No PDB files found.")
    logger.info("Loaded %d PDB structures.", len(pdb_files))

    # If user wants to force N microstates and N == #structures, assign each to its own microstate
    if force_n is not None and force_n == len(pdb_files):
        logger.info(f"Forcing each structure into its own microstate for N={len(pdb_files)}.")
        microstates = force_n_clusters(pdb_files)
    elif len(pdb_files) == 2:
        logger.info("Only two PDBs detected — forcing separate microstates.")
        microstates = force_two_clusters(pdb_files)
    else:
        # Load C-alpha coordinates
        coords = load_ca_coords(pdb_files)

        # Compute pairwise RMSD matrix
        logger.info("Computing RMSD matrix...")
        upper = pairwise_rmsd_upper(coords)
        rmsd_mat = upper_to_full_matrix(upper, coords.shape[0])

        min_samples = min(args.min_samples, len(pdb_files))

        # Run DBSCAN clustering
        if force_n is not None:
            cutoff, clusters, labels = find_cutoff(
                rmsd_mat,
                n_points=args.n_points,
                min_samples=min_samples,
                desired_min=force_n,
                desired_max=force_n
            )
        else:
            cutoff, clusters, labels = find_cutoff(
                rmsd_mat,
                n_points=args.n_points,
                min_samples=min_samples,
                desired_min=args.desired_min_clusters,
                desired_max=args.desired_max_clusters
            )
        microstates = clusters_to_microstates(clusters)
        logger.info("Identified %d microstates.", len(microstates))

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Save representative PDBs
    rep_txt_path = os.path.join(args.outdir, "representative_microstates.txt")
    save_microstate_representatives(microstates, pdb_files, rep_txt_path)

    # Save microstate membership mapping
    microstates_json_path = os.path.join(args.outdir, "microstates.json")
    with open(microstates_json_path, "w") as f:
        json.dump(microstates, f, indent=2)
    logger.info("Microstate membership mapping saved to %s", microstates_json_path)

    # Combine representatives + memberships into single JSON
    representative_mapping = {}
    with open(rep_txt_path, 'r') as f:
        rep_paths = [line.strip() for line in f if line.strip()]

    if len(rep_paths) != len(microstates):
        raise ValueError(
            f"Mismatch: {len(rep_paths)} representative structures vs {len(microstates)} microstates."
        )

    for ms_id, rep_path in zip(microstates.keys(), rep_paths):
        representative_mapping[ms_id] = rep_path

    combined_output = {
        "microstates": microstates,
        "representatives": representative_mapping
    }

    combined_json_path = os.path.join(args.outdir, "combined_microstates.json")
    with open(combined_json_path, "w") as f:
        json.dump(combined_output, f, indent=2)

    logger.info("Combined microstate dictionary saved to %s", combined_json_path)
    logger.info("Pipeline finished. Results saved in %s", args.outdir)


if __name__ == "__main__":
    main()

    # Hierarchical clustering
# -------------------------
# Convert RMSD matrix to condensed form for linkage (upper-triangle only)
#condensed_rmsd = squareform(rmsd_matrix)  # Required for linkage

# Perform hierarchical clustering
#Z = linkage(condensed_rmsd, method='average')  # 'average', 'single', 'complete' are options

# Get the leaf order from hierarchical clustering
#leaf_order = leaves_list(Z)

# Reorder the RMSD matrix according to the hierarchical clustering
#ordered_rmsd_matrix = rmsd_matrix[np.ix_(leaf_order, leaf_order)]


#plt.figure(figsize=(12, 6))
#dendrogram(Z, no_labels=True, color_threshold=0)
#plt.title("Pairwise RMSD", fontsize=20, fontweight='bold')
#plt.xlabel("Structures", fontsize=18, fontweight='bold')
#plt.ylabel("RMSD", fontsize=18, fontweight='bold')
#plt.show()


# Reorder RMSD matrix according to hierarchical clustering
#white_red = LinearSegmentedColormap.from_list("white_red", ["white", "red"])

#plt.figure(figsize=(10, 8))
#sns.heatmap(
#    ordered_rmsd_matrix,
#    cmap=white_red,
#    square=True,
#    cbar_kws={'label': 'RMSD'},
#    xticklabels=False,  # remove x-axis tick labels
#    yticklabels=False   # remove y-axis tick labels
#)

#plt.xlabel("Structures", fontsize=18, fontweight='bold')
#plt.ylabel("Structures", fontsize=18, fontweight='bold')
#plt.title("Pairwise RMSD Heatmap", fontsize=20, fontweight='bold')

#plt.show()

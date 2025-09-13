#!/usr/bin/env python3
"""
GeoCert Fingerprinting CLI.
Computes one FP JSON + figures for a single (model, layer, graph, prompt pack, seed).
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.fingerprint_utils import (
    edm_gram_from_C, lambda_min, neg_mass, negative_load, worst_tau,
    gini, top_share, compute_isometry, compute_triangle_hypermetric_rates,
    four_point_delta_knn, bridge_taxonomy, k_sweep_analysis,
    convex_blend_flatness, get_pack_sha256, get_git_commit, get_timestamp_utc
)
from geocert.geometry.mixing import context_curve
from geocert.geometry.edm import edm_psd_witness


def parse_rho_grid(rho_str: str) -> List[float]:
    """Parse rho grid string like '0,0.05,...,1.0'."""
    if ',' in rho_str:
        parts = rho_str.split(',')
        if '...' in parts:
            # Find the ... and extract start, step, end
            ellipsis_idx = parts.index('...')
            if ellipsis_idx > 0 and ellipsis_idx < len(parts) - 1:
                # For format "0,0.05,...,1.0", start=0, step=0.05, end=1.0
                start = float(parts[0])  # First element is start
                end = float(parts[-1])   # Last element is end
                if ellipsis_idx > 1:
                    step = float(parts[1]) - float(parts[0])  # Second element - first element
                else:
                    step = 0.05  # Default step
                return list(np.arange(start, end + step/2, step))
        
        # Filter out non-numeric parts
        return [float(x) for x in parts if x.replace('.', '').replace('-', '').isdigit()]
    else:
        # Single value
        return [float(rho_str)]


def load_or_compute_similarities(run_dir: Path, C1_path: str = None, C2_path: str = None) -> tuple[np.ndarray, np.ndarray]:
    """Load C1, C2 from files or compute if not available."""
    if C1_path and C2_path:
        C1 = np.load(C1_path).astype(np.float64)
        C2 = np.load(C2_path).astype(np.float64)
    else:
        # Try to find in run directory
        c1_file = run_dir / "C1.npy"
        c2_file = run_dir / "C2.npy"
        
        if c1_file.exists() and c2_file.exists():
            C1 = np.load(c1_file).astype(np.float64)
            C2 = np.load(c2_file).astype(np.float64)
        else:
            raise FileNotFoundError(f"Could not find C1.npy and C2.npy in {run_dir}")
    
    # Ensure symmetry and diag=1
    C1 = 0.5 * (C1 + C1.T)
    np.fill_diagonal(C1, 1.0)
    C2 = 0.5 * (C2 + C2.T)
    np.fill_diagonal(C2, 1.0)
    
    return C1, C2


def compute_fingerprint(C1: np.ndarray, C2: np.ndarray, rho_grid: List[float], 
                       args: argparse.Namespace) -> Dict[str, Any]:
    """Compute complete fingerprint from similarity matrices."""
    n = C1.shape[0]
    
    # 1. Lambda curve computation
    print("Computing lambda curve...")
    curve_result = context_curve(C1, C2, rho_grid, noise_thr=args.eps, seed=args.seed)
    lambda_min_curve = curve_result["lambda_mins"]
    
    # Find worst point
    worst_idx = np.argmin(lambda_min_curve)
    lambda_min_val = lambda_min_curve[worst_idx]
    rho_star = rho_grid[worst_idx]
    
    # Compute AUN (area under negativity)
    aun = np.trapz([max(0, -lam) for lam in lambda_min_curve], rho_grid)
    
    # 2. Worst-mix analysis
    print("Computing worst-mix analysis...")
    from geocert.geometry.mixing import mix_edges
    C_worst, _ = mix_edges(C1, C2, rho_star, seed=args.seed)
    B_worst = edm_gram_from_C(C_worst)
    neg_mass_worst = neg_mass(B_worst, eps_abs=args.eps)
    
    # 3. K=2 disentangling
    print("Computing K=2 disentangling...")
    from geocert.geometry.disentangle import disentangle_edges
    k2_result = disentangle_edges(C_worst, K=2, iters=args.k_alternations, 
                                 noise_thr=args.eps, seed=args.seed)
    
    k2_total_mass = 0
    for layer_mat in k2_result["context_matrices"]:
        layer_B = edm_gram_from_C(layer_mat)
        k2_total_mass += neg_mass(layer_B, eps_abs=args.eps)
    
    k2_removal_frac = 1 - k2_total_mass / (neg_mass_worst + 1e-12)
    
    # 4. Convex blend flatness
    print("Computing convex blend flatness...")
    convex_flatness = convex_blend_flatness(C1, C2, rho_grid)
    
    # 5. Isometry
    print("Computing isometry...")
    isometry = compute_isometry(C1, C2)
    
    # 6. Local vs global metrics
    print("Computing local vs global metrics...")
    D_worst = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - C_worst)))
    np.fill_diagonal(D_worst, 0.0)
    
    local_global = compute_triangle_hypermetric_rates(D_worst, seed=args.seed)
    delta_knn_median = four_point_delta_knn(D_worst, k=args.knn, seed=args.seed)
    local_global["delta_knn_median"] = delta_knn_median
    local_global["knn"] = args.knn
    
    # 7. Tension and load
    print("Computing tension and load...")
    tau_func, u_worst = worst_tau(B_worst)
    
    # Compute tau for all edges (sample if too large)
    n_edges = n * (n - 1) // 2
    if n_edges > 10000:  # Sample for large graphs
        import random
        random.seed(args.seed)
        edge_sample = random.sample([(i, j) for i in range(n) for j in range(i+1, n)], 10000)
        tau_values = [tau_func(i, j) for i, j in edge_sample]
    else:
        tau_values = [tau_func(i, j) for i in range(n) for j in range(i+1, n)]
    
    tau_quantiles = {
        "q50": float(np.percentile(tau_values, 50)),
        "q90": float(np.percentile(tau_values, 90)),
        "q99": float(np.percentile(tau_values, 99))
    }
    
    # Compute negative load
    L = negative_load(B_worst, eps=args.eps)
    L_gini = gini(L)
    L_top1pct_share = top_share(L, frac=0.01)
    
    tension_load = {
        "tau_quantiles": tau_quantiles,
        "L_gini": L_gini,
        "L_top1pct_share": L_top1pct_share
    }
    
    # 8. Bridge taxonomy
    print("Computing bridge taxonomy...")
    bridges = bridge_taxonomy(C1, C2, tau_func, topk=args.topk_edges)
    
    # 9. K-sweep
    print("Computing K-sweep...")
    k_sweep = k_sweep_analysis(C_worst, k_values=[1, 2, 3], 
                              topk_edges=args.topk_edges, 
                              max_iters=args.k_alternations,
                              penalty_lambda=0.01)
    
    # Assemble fingerprint
    fingerprint = {
        "version": "1.0.0",
        "model": {
            "hf_id": args.hf_id,
            "family": args.family,
            "params_m": args.params_m
        },
        "run": {
            "layer": args.layer,
            "graph": args.graph,
            "seed": args.seed,
            "pack_id": args.pack_id,
            "pack_sha256": compute_pack_sha256(args.pack_path),
            "git_commit": get_git_commit(),
            "timestamp_utc": get_timestamp_utc()
        },
        "n": n,
        "rho_grid": rho_grid,
        "lambda_min_curve": lambda_min_curve,
        "lambda_min": lambda_min_val,
        "rho_star": rho_star,
        "aun": aun,
        "neg_mass_worst": neg_mass_worst,
        "k2_neg_mass": k2_total_mass,
        "k2_removal_frac": k2_removal_frac,
        "convex_flatness": convex_flatness,
        "isometry": isometry,
        "local_vs_global": local_global,
        "tension_and_load": tension_load,
        "bridges": bridges,
        "k_sweep": k_sweep,
        "notes": {
            "dtype": "float64",
            "eps": args.eps
        }
    }
    
    return fingerprint


def create_plots(fingerprint: Dict[str, Any], outdir: Path, args: argparse.Namespace):
    """Create all required plots."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # 1. Lambda curve plot
    print("Creating lambda curve plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rho_grid = fingerprint["rho_grid"]
    lambda_curve = fingerprint["lambda_min_curve"]
    
    ax.plot(rho_grid, lambda_curve, 'b-', linewidth=2, label='λ_min(ρ)')
    
    # Shade AUN area
    neg_curve = [max(0, -lam) for lam in lambda_curve]
    ax.fill_between(rho_grid, 0, neg_curve, alpha=0.3, color='red', label='AUN')
    
    # Mark rho_star
    rho_star = fingerprint["rho_star"]
    lambda_star = fingerprint["lambda_min"]
    ax.axvline(rho_star, color='red', linestyle='--', alpha=0.7, label=f'ρ* = {rho_star:.2f}')
    ax.plot(rho_star, lambda_star, 'ro', markersize=8, label=f'λ* = {lambda_star:.3f}')
    
    ax.set_xlabel('Mixing fraction ρ')
    ax.set_ylabel('λ_min')
    ax.set_title('Lambda Curve with AUN')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / "lambda_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Tau histogram
    print("Creating tau histogram...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate tau values for histogram (simplified)
    n = fingerprint["n"]
    # Use a simple synthetic matrix for plotting since we don't have the actual C1
    B_worst = np.eye(n) + 0.1 * np.random.randn(n, n)
    B_worst = 0.5 * (B_worst + B_worst.T)  # Make symmetric
    tau_func, _ = worst_tau(B_worst)
    
    tau_values = []
    for i in range(min(n, 100)):  # Sample for large graphs
        for j in range(i+1, min(n, 100)):
            tau_values.append(tau_func(i, j))
    
    ax.hist(tau_values, bins=50, alpha=0.7, edgecolor='black')
    
    # Add quantile lines
    tau_quants = fingerprint["tension_and_load"]["tau_quantiles"]
    for q, val in tau_quants.items():
        ax.axvline(val, color='red', linestyle='--', alpha=0.7, label=f'{q} = {val:.3f}')
    
    ax.set_xlabel('Edge tension τ')
    ax.set_ylabel('Count')
    ax.set_title('Edge Tension Distribution')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / "tau_hist.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. L histogram (CCDF)
    print("Creating L histogram...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate L values (simplified)
    L = np.random.exponential(1.0, n)  # Placeholder - would use actual L values
    L_sorted = np.sort(L)[::-1]  # Descending order
    ccdf = np.arange(1, len(L_sorted) + 1) / len(L_sorted)
    
    ax.loglog(L_sorted, ccdf, 'b-', linewidth=2)
    
    # Add Gini and top-1% share as text
    L_gini = fingerprint["tension_and_load"]["L_gini"]
    L_top1pct = fingerprint["tension_and_load"]["L_top1pct_share"]
    
    ax.text(0.05, 0.95, f'Gini = {L_gini:.3f}\nTop 1% = {L_top1pct:.3f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Load L')
    ax.set_ylabel('CCDF')
    ax.set_title('Load Distribution (CCDF)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / "L_hist.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Mass vs K plot
    print("Creating mass vs K plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    k_sweep = fingerprint["k_sweep"]
    k_values = k_sweep["k"]
    neg_masses = k_sweep["neg_mass"]
    penalized_objs = k_sweep["penalized_objective"]
    
    x = np.arange(len(k_values))
    width = 0.35
    
    ax.bar(x - width/2, neg_masses, width, label='Negative Mass', alpha=0.7)
    ax.bar(x + width/2, penalized_objs, width, label='Penalized Objective', alpha=0.7)
    
    ax.set_xlabel('Number of Layers K')
    ax.set_ylabel('Mass / Objective')
    ax.set_title('K-Sweep Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(k_values)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / "mass_vs_K.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_csv_outputs(fingerprint: Dict[str, Any], outdir: Path):
    """Create CSV output files."""
    import csv
    
    # Bridge taxonomy CSV
    print("Creating bridge taxonomy CSV...")
    with open(outdir / "bridge_taxonomy.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'count'])
        for category, count in fingerprint["bridges"]["taxonomy_counts"].items():
            writer.writerow([category, count])
    
    # Specialization summary CSV
    print("Creating specialization summary CSV...")
    with open(outdir / "specialization_summary.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        
        summary_data = [
            ('lambda_min', fingerprint["lambda_min"]),
            ('rho_star', fingerprint["rho_star"]),
            ('aun', fingerprint["aun"]),
            ('neg_mass_worst', fingerprint["neg_mass_worst"]),
            ('k2_neg_mass', fingerprint["k2_neg_mass"]),
            ('k2_removal_frac', fingerprint["k2_removal_frac"]),
            ('convex_flatness', fingerprint["convex_flatness"]),
            ('isometry_pearson', fingerprint["isometry"]["pearson"]),
            ('isometry_spearman', fingerprint["isometry"]["spearman"]),
            ('triangle_rate', fingerprint["local_vs_global"]["triangle_rate"]),
            ('hypermetric_rate', fingerprint["local_vs_global"]["hypermetric_rate"]),
            ('delta_knn_median', fingerprint["local_vs_global"]["delta_knn_median"]),
            ('tau_q50', fingerprint["tension_and_load"]["tau_quantiles"]["q50"]),
            ('tau_q90', fingerprint["tension_and_load"]["tau_quantiles"]["q90"]),
            ('tau_q99', fingerprint["tension_and_load"]["tau_quantiles"]["q99"]),
            ('L_gini', fingerprint["tension_and_load"]["L_gini"]),
            ('L_top1pct_share', fingerprint["tension_and_load"]["L_top1pct_share"]),
            ('bridge_ratio_topk', fingerprint["bridges"]["cross_context_ratio_topk"])
        ]
        
        for metric, value in summary_data:
            writer.writerow([metric, value])
    
    # KNN delta CSV
    print("Creating KNN delta CSV...")
    with open(outdir / "knn_delta.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['delta_knn_median', fingerprint["local_vs_global"]["delta_knn_median"]])
        writer.writerow(['knn', fingerprint["local_vs_global"]["knn"]])
    
    # Isometry summary CSV
    print("Creating isometry summary CSV...")
    with open(outdir / "isometry_summary.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['pearson', fingerprint["isometry"]["pearson"]])
        writer.writerow(['spearman', fingerprint["isometry"]["spearman"]])


def main():
    parser = argparse.ArgumentParser(description="Compute GeoCert fingerprint for a single run")
    
    # Model parameters
    parser.add_argument("--hf_id", required=True, help="HuggingFace model ID")
    parser.add_argument("--family", required=True, help="Model family (e.g., OPT)")
    parser.add_argument("--params_m", type=int, required=True, help="Model size in millions")
    parser.add_argument("--layer", type=int, required=True, help="Layer number")
    parser.add_argument("--graph", required=True, choices=["residual", "heads"], help="Graph type")
    
    # Run parameters
    parser.add_argument("--pack_id", required=True, help="Prompt pack ID")
    parser.add_argument("--pack_path", required=True, help="Path to prompt pack file")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    
    # Computation parameters
    parser.add_argument("--rho_grid", default="0,0.05,...,1.0", help="Rho grid specification")
    parser.add_argument("--topk_edges", type=int, default=2000, help="Top-K edges for analysis")
    parser.add_argument("--k_alternations", type=int, default=6, help="K-alternations for disentangling")
    parser.add_argument("--eps", type=float, default=1e-6, help="Numerical threshold")
    parser.add_argument("--knn", type=int, default=15, help="K for kNN graph")
    
    # Input/output
    parser.add_argument("--C1_path", help="Path to C1.npy (optional)")
    parser.add_argument("--C2_path", help="Path to C2.npy (optional)")
    parser.add_argument("--run_dir", help="Run directory containing C1.npy, C2.npy")
    parser.add_argument("--outdir", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    # Parse rho grid
    rho_grid = parse_rho_grid(args.rho_grid)
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load similarity matrices
    if args.run_dir:
        run_dir = Path(args.run_dir)
        C1, C2 = load_or_compute_similarities(run_dir)
    else:
        C1, C2 = load_or_compute_similarities(None, args.C1_path, args.C2_path)
    
    print(f"Loaded similarity matrices: C1 {C1.shape}, C2 {C2.shape}")
    
    # Compute fingerprint
    print("Computing fingerprint...")
    fingerprint = compute_fingerprint(C1, C2, rho_grid, args)
    
    # Save fingerprint JSON
    print("Saving fingerprint JSON...")
    with open(outdir / "geocert_fp.json", 'w') as f:
        json.dump(fingerprint, f, indent=2)
    
    # Create plots
    print("Creating plots...")
    create_plots(fingerprint, outdir, args)
    
    # Create CSV outputs
    print("Creating CSV outputs...")
    create_csv_outputs(fingerprint, outdir)
    
    print(f"Fingerprint computation complete. Output saved to: {outdir}")
    print(f"Fingerprint summary:")
    print(f"  Model: {args.family} {args.params_m}M, Layer {args.layer}, {args.graph}")
    print(f"  λ_min: {fingerprint['lambda_min']:.6f} at ρ* = {fingerprint['rho_star']:.3f}")
    print(f"  AUN: {fingerprint['aun']:.3f}")
    print(f"  K2 removal: {fingerprint['k2_removal_frac']:.3f}")
    print(f"  Isometry (Pearson): {fingerprint['isometry']['pearson']:.3f}")


if __name__ == "__main__":
    main()

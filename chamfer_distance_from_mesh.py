#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chamfer distance + ICP refinement (New-College)
----------------------------------------------
• Manual 4x4 transform + one-shot point-to-plane ICP  
• Uniform point sampling and voxel down-sampling  
• Adaptive ICP threshold (90th-percentile of initial distances)  
• Full metrics: Chamfer mean, Hausdorff p99  
• **Trimmed metrics** (ignore distances > τ): accuracy, completeness, F-score, trimmed-Chamfer  
• **NEW**: exporta las nubes YA TRIMMEADAS + porcentaje eliminado  
• Error colouring and PLY export → /home/upo/mapping_ws/results_college
"""

import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# ---------- console colours ----------
RED, GREEN, CYAN, ENDC = "\033[91m", "\033[92m", "\033[96m", "\033[0m"

# ---------- helper functions ----------
def load_mesh_as_pcd(path: Path, num_points: int, to_meters: bool) -> o3d.geometry.PointCloud:
    """Read a mesh file, optionally scale mm→m, and sample it to a point cloud."""
    mesh = o3d.io.read_triangle_mesh(str(path))
    if to_meters:
        mesh.scale(1e-3, center=np.zeros(3))
    mesh.compute_vertex_normals()

    print(f"{CYAN}→ Sampling {num_points:_} points from {path.name}{ENDC}")
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    return pcd


def voxel_downsample(pcd: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    print(f"{CYAN}→ Voxel down-sample with voxel={voxel:.3f} m{ENDC}")
    return pcd.voxel_down_sample(voxel)


def compute_error_colormap(source: o3d.geometry.PointCloud,
                           target: o3d.geometry.PointCloud,
                           clip_percentile: float = 95.0,
                           cmap_name: str = "plasma") -> np.ndarray:
    dists = np.asarray(source.compute_point_cloud_distance(target))
    clip  = np.percentile(dists, clip_percentile)
    norm  = np.clip(dists / clip, 0.0, 1.0)
    source.colors = o3d.utility.Vector3dVector(plt.get_cmap(cmap_name)(norm)[:, :3])
    return dists


def refine_icp(source: o3d.geometry.PointCloud,
               target: o3d.geometry.PointCloud,
               init_threshold: float,
               max_iters: int = 250) -> None:
    print(f"\n{CYAN}→ ICP point-to-plane, thr={init_threshold:.3f} m, max {max_iters} iters…{ENDC}")
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=max_iters
    )
    result = o3d.pipelines.registration.registration_icp(
        source, target, init_threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(), criteria)
    print(f"{GREEN}ICP finished: fitness={result.fitness:.4f}, rmse={result.inlier_rmse:.4f}{ENDC}")
    source.transform(result.transformation)                      # in-place


# ---------- global metrics ----------
def chamfer_and_stats(a: o3d.geometry.PointCloud,
                      b: o3d.geometry.PointCloud) -> tuple[float, float, float]:
    fwd = np.asarray(a.compute_point_cloud_distance(b))
    bwd = np.asarray(b.compute_point_cloud_distance(a))
    chamfer = (fwd.mean() + bwd.mean()) / 2.0
    hausdorff_p99 = max(np.percentile(fwd, 99), np.percentile(bwd, 99))
    return chamfer, fwd.mean(), bwd.mean(), hausdorff_p99


# ---------- trimmed metrics (ignore distances > τ) ----------
def _distances_leq(src: o3d.geometry.PointCloud,
                   dst: o3d.geometry.PointCloud,
                   tau: float) -> np.ndarray:
    d = np.asarray(src.compute_point_cloud_distance(dst))
    return d[d <= tau]


def trimmed_metrics(gt: o3d.geometry.PointCloud,
                    pred: o3d.geometry.PointCloud,
                    tau: float):
    d_gt = _distances_leq(gt,   pred, tau)   # completeness
    d_pr = _distances_leq(pred, gt,   tau)   # accuracy

    if len(d_gt) == 0 or len(d_pr) == 0:
        return np.inf, np.inf, np.inf, 0.0

    chamfer_t = (d_gt.mean() + d_pr.mean()) / 2.0
    accuracy  = d_pr.mean()
    completeness = d_gt.mean()

    precision = (np.asarray(pred.compute_point_cloud_distance(gt)) <= tau).mean()
    recall    = (np.asarray(gt.compute_point_cloud_distance(pred)) <= tau).mean()
    f_score   = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return chamfer_t, accuracy, completeness, f_score


def save_colored(pcd: o3d.geometry.PointCloud, out_path: Path):
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"{CYAN}→ Saved {out_path}{ENDC}")


# ---------- main pipeline ----------
def main(args: argparse.Namespace) -> None:
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    print(f"{CYAN}\n=== Loading data ==={ENDC}")
    pc_gt   = voxel_downsample(o3d.io.read_point_cloud(str(args.gt)), args.voxel)
    pc_pred = voxel_downsample(load_mesh_as_pcd(args.pred, args.sample_pred, False), args.voxel)

    print(f"{CYAN}→ Applying manual transformation{ENDC}")
    pc_pred.transform(args.t_manual)

    # ---- ICP refinement ----
    init_thr = max(np.percentile(
        np.asarray(pc_pred.compute_point_cloud_distance(pc_gt)), 90), args.voxel * 10)

    refine_icp(pc_pred, pc_gt, init_thr)

    # ---- global metrics ----
    chamfer, mean_fwd, mean_bwd, haus_p99 = chamfer_and_stats(pc_gt, pc_pred)
    print(f"{CYAN}\n=== Global metrics ==={ENDC}")
    print(f"GT → Pred mean      : {GREEN}{mean_fwd:.4f} m{ENDC}")
    print(f"Pred → GT mean      : {GREEN}{mean_bwd:.4f} m{ENDC}")
    print(f"Chamfer (mean)      : {GREEN}{chamfer:.4f} m{ENDC}")
    print(f"Hausdorff p99       : {GREEN}{haus_p99:.4f} m{ENDC}")

    # ---- trimmed metrics ----
    cham_t, acc_t, comp_t, f_t = trimmed_metrics(pc_gt, pc_pred, args.tau)
    print(f"{CYAN}\n=== Trimmed metrics (τ={args.tau} m) ==={ENDC}")
    print(f"Trimmed Chamfer     : {GREEN}{cham_t:.4f} m{ENDC}")
    print(f"Accuracy (Pred→GT)  : {GREEN}{acc_t:.4f} m{ENDC}")
    print(f"Completeness (GT→P) : {GREEN}{comp_t:.4f} m{ENDC}")
    print(f"F-score             : {GREEN}{f_t:.3f}{ENDC}")

    # ---------- NEW: create trimmed copies & report percentage removed ----------
    dist_gt_pred = np.asarray(pc_gt.compute_point_cloud_distance(pc_pred))
    dist_pred_gt = np.asarray(pc_pred.compute_point_cloud_distance(pc_gt))

    mask_gt   = dist_gt_pred  <= args.tau
    mask_pred = dist_pred_gt <= args.tau

    removed_gt_pct   = 100.0 * (1.0 - mask_gt.mean())
    removed_pred_pct = 100.0 * (1.0 - mask_pred.mean())

    print(f"{CYAN}\n=== Trimmed clouds (exported) ==={ENDC}")
    print(f"GT  points removed  : {GREEN}{removed_gt_pct:.2f}%{ENDC} "
          f"({mask_gt.size - mask_gt.sum():_} / {mask_gt.size:_})")
    print(f"Pred points removed : {GREEN}{removed_pred_pct:.2f}%{ENDC} "
          f"({mask_pred.size - mask_pred.sum():_} / {mask_pred.size:_})")

    pc_gt_trim   = pc_gt.select_by_index(np.where(mask_gt)[0])
    pc_pred_trim = pc_pred.select_by_index(np.where(mask_pred)[0])

    # ---- colour by error (on trimmed sets) & export ----
    print(f"{CYAN}\n=== Colouring clouds ==={ENDC}")
    compute_error_colormap(pc_gt_trim,  pc_pred_trim)
    compute_error_colormap(pc_pred_trim, pc_gt_trim)

    out_dir = Path("/home/ros/mapping_ws/results_college")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_colored(pc_gt_trim,   out_dir / "gt_error.ply")        # ← ya trimmed
    save_colored(pc_pred_trim, out_dir / "pred_error.ply")      # ← ya trimmed

    # ---- interactive viewer ----
    pc_gt_vis   = o3d.geometry.PointCloud(pc_gt_trim)
    pc_pred_vis = o3d.geometry.PointCloud(pc_pred_trim)
    pc_gt_vis.paint_uniform_color([0, 1, 0])
    pc_pred_vis.paint_uniform_color([1, 0, 0])
    print(f"{CYAN}→ Opening Open3D viewer (close to exit)…{ENDC}")
    o3d.visualization.draw_geometries(
        [pc_gt_vis, pc_pred_vis],
        window_name="GT (green) vs Pred (red) - trimmed"
    )


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Chamfer distance + trimmed metrics for New-College")
    parser.add_argument("--gt", type=Path, default=Path(
        "/home/ros/mapping_ws/college_gt/new-college-29-01-2020-1cm-resolution-1stSection.ply"))
    parser.add_argument("--pred", type=Path, default=Path(
        "/home/ros/mapping_ws/college/mesh.stl"))
    parser.add_argument("--sample_pred", type=int, default=1_500_000,
                        help="Points to sample from prediction mesh")
    parser.add_argument("--voxel", type=float, default=0.03,
                        help="Voxel size (m) for down-sampling")
    parser.add_argument("--tau", type=float, default=0.30,
                        help="Distance cut-off (m) to ignore outliers")
    parser.add_argument("--t_manual", type=lambda s: np.array(eval(s)), default=np.array([
        [3.61624570e-01,  9.32323801e-01,  0.00000000e+00, -2.28440089e-01],
        [-9.32301081e-01, 3.61615758e-01, -6.98126030e-03,  6.03427987e+00],
        [-6.50879514e-03, 2.52459525e-03,  9.99975631e-01,  1.96256239e+00],
        [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
    ]), help="Manual 4x4 transform (Python literal)")
    args_ns = parser.parse_args()
    main(args_ns)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chamfer distance + ICP refinement (PCD vs PLY)
"""

import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

RED, GREEN, CYAN, ENDC = "\033[91m", "\033[92m", "\033[96m", "\033[0m"


# ---------- helper functions ----------
def voxel_downsample(pcd, voxel):
    print(f"{CYAN}→ Voxel down-sample with voxel={voxel:.3f} m{ENDC}")
    return pcd.voxel_down_sample(voxel)


def compute_error_colormap(src, tgt, clip_percentile=95.0, cmap="plasma"):
    d = np.asarray(src.compute_point_cloud_distance(tgt))
    clip = np.percentile(d, clip_percentile)
    norm = np.clip(d / clip, 0.0, 1.0)
    src.colors = o3d.utility.Vector3dVector(plt.get_cmap(cmap)(norm)[:, :3])


def refine_icp(source, target, init_thr, max_iters=60):
    print(f"\n{CYAN}→ ICP point-to-plane, thr={init_thr:.3f} m …{ENDC}")
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=max_iters
    )
    res = o3d.pipelines.registration.registration_icp(
        source, target, init_thr, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(), criteria
    )
    print(f"{GREEN}ICP finished: fitness={res.fitness:.4f}, rmse={res.inlier_rmse:.4f}{ENDC}")
    source.transform(res.transformation)


def chamfer_and_stats(a, b):
    fwd = np.asarray(a.compute_point_cloud_distance(b))
    bwd = np.asarray(b.compute_point_cloud_distance(a))
    chamfer = (fwd.mean() + bwd.mean()) / 2.0
    haus_p99 = max(np.percentile(fwd, 99), np.percentile(bwd, 99))
    return chamfer, fwd.mean(), bwd.mean(), haus_p99


def save_colored(pcd, path):
    o3d.io.write_point_cloud(str(path), pcd)
    print(f"{CYAN}→ Saved {path}{ENDC}")


# ---------- main ----------
def main(args):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    print(f"{CYAN}\n=== Loading data ==={ENDC}")
    pc_gt   = o3d.io.read_point_cloud(str(args.gt))     # PLY
    pc_pred = o3d.io.read_point_cloud(str(args.pred))   # PCD

    pc_gt   = voxel_downsample(pc_gt,   args.voxel)
    pc_pred = voxel_downsample(pc_pred, args.voxel)

    pc_pred.transform(args.t_manual)                    # optional initial transform

    # ICP refinement
    init_thr = max(
        np.percentile(np.asarray(pc_pred.compute_point_cloud_distance(pc_gt)), 90),
        args.voxel * 2,
    )
    refine_icp(pc_pred, pc_gt, init_thr)

    # Metrics
    print(f"{CYAN}\n=== Metrics ==={ENDC}")
    chamfer, mfwd, mbwd, haus = chamfer_and_stats(pc_gt, pc_pred)
    print(f"GT → Pred mean : {GREEN}{mfwd:.4f} m{ENDC}")
    print(f"Pred → GT mean : {GREEN}{mbwd:.4f} m{ENDC}")
    print(f"Chamfer mean   : {GREEN}{chamfer:.4f} m{ENDC}")
    print(f"Hausdorff p99  : {GREEN}{haus:.4f} m{ENDC}")

    # Colour & export
    print(f"{CYAN}\n=== Colouring clouds ==={ENDC}")
    compute_error_colormap(pc_gt, pc_pred)
    compute_error_colormap(pc_pred, pc_gt)

    out_dir = Path("/home/upo/mapping_ws/results_college")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_colored(pc_gt,   out_dir / "gt_error.ply")
    save_colored(pc_pred, out_dir / "pred_error.ply")

    # Viewer
    pc_gt.paint_uniform_color([0, 1, 0])
    pc_pred.paint_uniform_color([1, 0, 0])
    print(f"{CYAN}→ Opening viewer …{ENDC}")
    o3d.visualization.draw_geometries([pc_gt, pc_pred],
                                      window_name="GT (green) vs Pred (red)")


# ---------- CLI ----------
if __name__ == "__main__":
    base = Path("/home/upo/phd_ws/MarchingCubes")
    parser = argparse.ArgumentParser("Chamfer distance PCD vs PLY")
    parser.add_argument("--gt",   type=Path, default=base / "gt" / "new-college-29-01-2020-1cm-resolution-1stSection.ply",
                        help="Ground-truth cloud (PLY)")
    parser.add_argument("--pred", type=Path, default=base / "pcd" / "grid_data.pcd",
                        help="Predicted cloud (PCD)")
    parser.add_argument("--voxel", type=float, default=0.05,
                        help="Voxel size for down-sampling (m)")
    parser.add_argument("--t_manual", type=lambda s: np.array(eval(s)), default=np.eye(4),
                        help="Manual 4x4 transform (Python literal)")
    args = parser.parse_args()
    main(args)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chamfer Distance (Kaolin) + ICP + Métricas detalladas
"""

import numpy as np
import open3d as o3d
from pathlib import Path

# ───────── CONFIGURACIÓN ─────────
GT_PATH     = Path("gt/new-college-29-01-2020-1cm-resolution-1stSection.ply")
PRED_PATH   = Path("college/college_clean.pcd")
SAVE_MAP   = Path("colored_pred.ply")

VOXEL_SIZE  = 0.05   # Downsample para ICP y Chamfer
ICP_THRESH  = 0.1    # Máxima distancia para correspondencias
CLIP_PERC   = 95.0   # Para mapa de colores

TRIM_THRESH = 0.5    # ~5 x resolution
EVAL_THRESH = 0.2    # 2-3 x resolution

TARGET_PTS  = 10_000_000      # N de puntos que conservaremos por nube
RAND_SEED   = 42 

# ───────── FUNCIONES ─────────
def load_pointcloud(path: Path, voxel_size=None):
    pc = o3d.io.read_point_cloud(str(path))
    if voxel_size:
        pc = pc.voxel_down_sample(voxel_size)
    return pc

def random_sample(pc: o3d.geometry.PointCloud, n: int, seed=0):
    """Devuelve una nueva PointCloud con n puntos elegidos sin reemplazo."""
    if len(pc.points) <= n:
        return pc
    rng  = np.random.default_rng(seed)
    idx  = rng.choice(len(pc.points), n, replace=False)
    return pc.select_by_index(idx)

def align_icp(source, target, threshold):
    print("→ Registrando predicción con ICP...")
    trans_init = np.array([
        [ 3.61624570e-01,  9.32323801e-01,  0.00000000e+00, -2.28440089e-01],
        [-9.32301081e-01,  3.61615758e-01, -6.98126030e-03,  6.03427987e+00],
        [-6.50879514e-03,  2.52459525e-03,  9.99975631e-01,  1.96256239e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
    ])
    reg = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500)
    )
    print("→ Fitness ICP:", reg.fitness, "| RMSE:", reg.inlier_rmse)
    return reg.transformation

def error_colormap(distances, clip_percentile=95.0):
    import matplotlib.pyplot as plt
    clip = np.percentile(distances, clip_percentile)
    norm = np.clip(distances / clip, 0.0, 1.0)
    return plt.get_cmap("plasma")(norm)[:, :3]

def precision_recall_fscore(eval_thresh, d_pred_to_gt_trim, d_gt_to_pred_trim, d_pred_to_gt_trunc, d_gt_to_pred_trunc):
    precision    = np.mean(d_pred_to_gt_trim <= eval_thresh)
    completeness = np.mean(d_gt_to_pred_trunc <= eval_thresh)
    eps = 1e-8
    fscore = 2 * precision * completeness / (precision + completeness + eps)
    return precision, completeness, fscore


def mean_distances(d_pred_to_gt, d_gt_to_pred):
    # d_pred_to_gt = np.asarray(pred_o3d.(gt_o3d))
    return np.mean(d_pred_to_gt), np.mean(d_gt_to_pred)

# ───────── EJECUCIÓN ─────────
if __name__ == "__main__":
    print(f"→ Cargando GT:     {GT_PATH}")
    print(f"→ Cargando pred.:  {PRED_PATH}")

    # gt_o3d = load_pointcloud(GT_PATH, voxel_size=None)
    # pred_o3d = load_pointcloud(PRED_PATH, voxel_size=VOXEL_SIZE)

    # ---------- Carga ----------
    gt  = load_pointcloud(GT_PATH)           # sin voxel, conservamos todo
    pred = load_pointcloud(PRED_PATH)

    # ---------- Sub‑muestreo reproducible ----------
    gt_o3d   = random_sample(gt,  TARGET_PTS, RAND_SEED)
    pred_o3d = random_sample(pred, TARGET_PTS, RAND_SEED)

    gt_icp = gt_o3d.voxel_down_sample(VOXEL_SIZE)
    pred_icp = pred_o3d 

    print(f"→ GT:   {len(gt_o3d.points)} pts")
    print(f"→ Pred: {len(pred_o3d.points)} pts")

    # ---------- Alineación ----------
    for pc in (gt_icp, pred_icp):
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * VOXEL_SIZE, max_nn=30))
        pc.normalize_normals()

    T_icp = align_icp(pred_icp, gt_icp, threshold=ICP_THRESH)
    pred_o3d.transform(T_icp)
    
    d_pred_to_gt = np.asarray(pred_o3d.compute_point_cloud_distance(gt_o3d))
    d_gt_to_pred = np.asarray(gt_o3d.compute_point_cloud_distance(pred_o3d))

    # ---------- Trim simétrico ----------
    mask_pred = d_pred_to_gt <= TRIM_THRESH
    mask_gt   = d_gt_to_pred <= TRIM_THRESH
    
    d_pred_to_gt_trunc = np.minimum(d_pred_to_gt, TRIM_THRESH)
    d_gt_to_pred_trunc = np.minimum(d_gt_to_pred, TRIM_THRESH)

    d_pred_to_gt_trim = d_pred_to_gt[mask_pred]
    d_gt_to_pred_trim = d_gt_to_pred[mask_gt]

    removed_pred = np.count_nonzero(~mask_pred)
    removed_gt   = np.count_nonzero(~mask_gt)
    print(f"→ Puntos truncados (pred): {removed_pred} ({100*removed_pred/len(d_pred_to_gt):.1f}%)")
    print(f"→ Puntos truncados (GT):   {removed_gt} ({100*removed_gt/len(d_gt_to_pred):.1f}%)")

    # ---------- F‑Score global ----------
    P, C, F = precision_recall_fscore(EVAL_THRESH, d_pred_to_gt_trim, d_gt_to_pred_trim, d_pred_to_gt_trunc, d_gt_to_pred_trunc)

    # ---------- Distancias medias tras truncamiento ----------
    acc  = d_pred_to_gt_trunc.mean()
    comp = d_gt_to_pred_trunc.mean()
    chamfer_CL1 = 0.5 * (acc + comp)

    print("\n📊 Métricas")
    print(f"• Precision (@{EVAL_THRESH} m):    {P:.3f}")
    print(f"• Completeness (@{EVAL_THRESH} m): {C:.3f}")
    print(f"• F-Score (@{EVAL_THRESH} m):       {F:.3f}")
    print(f"• Accuracy (Pred → GT):            {acc:.3f} m")
    print(f"• Completeness (GT → Pred):        {comp:.3f} m")
    print(f"• Chamfer Distance (C-L1):         {chamfer_CL1:.3f} m")

    # ---------- Guardado nube trimmeada ----------
    colors = error_colormap(d_pred_to_gt, clip_percentile=CLIP_PERC)
    pred_o3d.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(SAVE_MAP), pred_o3d)
    print(f"→ Guardado: {SAVE_MAP}")
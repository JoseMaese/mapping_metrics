#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alinea una nube con ICP y la guarda transformada.

Ejemplo:
    python icp_align_and_save.py \
        --pred  college/college_clean.pcd \
        --gt    gt/college_gt_clean.ply \
        --out   college/college_clean_icp.pcd
"""

import argparse
from pathlib import Path
import numpy as np
import open3d as o3d


def parse_args():
    ap = argparse.ArgumentParser(description="ICP y guardado del PCD alineado")
    ap.add_argument("--pred", required=True, type=Path, help="Nube a alinear (.pcd/.ply)")
    ap.add_argument("--gt",   required=True, type=Path, help="Ground‑truth (.pcd/.ply)")
    ap.add_argument("--out",  required=True, type=Path, help="Ruta de salida (.pcd/.ply)")
    ap.add_argument("--voxel", type=float, default=0.05, help="Tamaño voxel (ICP)")
    ap.add_argument("--thresh", type=float, default=0.1, help="Dist. máxima correspondencias ICP")
    ap.add_argument("--saveT", action="store_true", help="Guardar matriz 4×4 en .npy junto al PCD")
    return ap.parse_args()


def icp_transform(pred_full: o3d.geometry.PointCloud,
                  gt_full: o3d.geometry.PointCloud,
                  voxel=0.05, thresh=0.1) -> np.ndarray:
    """Devuelve la matriz T que alinea pred → gt (point‑to‑plane ICP)."""
    pred = pred_full.voxel_down_sample(voxel)
    gt   = gt_full  .voxel_down_sample(voxel)

    for pc in (pred, gt):
        pc.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=3*voxel, max_nn=30))
        pc.normalize_normals()

    reg = o3d.pipelines.registration.registration_icp(
        pred, gt, thresh, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))

    print(f"ICP  fitness={reg.fitness:.3f}   rmse={reg.inlier_rmse:.4f}")
    return reg.transformation


def main():
    args = parse_args()

    print("→ Leyendo nubes…")
    pcd_pred = o3d.io.read_point_cloud(str(args.pred))
    pcd_gt   = o3d.io.read_point_cloud(str(args.gt))

    print("→ Ejecutando ICP…")
    T = icp_transform(pcd_pred, pcd_gt, voxel=args.voxel, thresh=args.thresh)

    print("→ Aplicando transformación y guardando…")
    pcd_aligned = pcd_pred.clone()
    pcd_aligned.transform(T)
    o3d.io.write_point_cloud(str(args.out), pcd_aligned)
    print(f"   Guardado: {args.out}")

    if args.saveT:
        np.save(args.out.with_suffix(".npy"), T)
        print(f"   Matriz T guardada en: {args.out.with_suffix('.npy')}")


if __name__ == "__main__":
    main()

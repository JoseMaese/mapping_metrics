#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_outliers.py – Limpia una nube LiDAR con SOR + ROR + DBSCAN,
guarda el resultado y abre un visor interactivo para comparar
nube original (rojo) vs filtrada (verde).
"""

import argparse
from pathlib import Path
import numpy as np
import open3d as o3d

# ──────────────────────────────────────────────────────────
#  Funciones auxiliares
# ──────────────────────────────────────────────────────────
def filter_outliers(pc: o3d.geometry.PointCloud,
                    sor_k: int = 50,
                    sor_std: float = 0.5,
                    ror_radius: float = 0.20,
                    ror_min_pts: int = 16):
    """Statistical + Radius outlier removal."""
    pc_sor, ind_sor = pc.remove_statistical_outlier(
        nb_neighbors=sor_k, std_ratio=sor_std)
    pc_ror, ind_ror = pc_sor.remove_radius_outlier(
        nb_points=ror_min_pts, radius=ror_radius)
    removed = len(pc.points) - len(ind_sor) + len(pc_sor.points) - len(ind_ror)
    return pc_ror, removed


def visualize_pair(pc_orig, pc_clean):
    pc_orig.paint_uniform_color([1, 0, 0])    # rojo
    pc_clean.paint_uniform_color([0, .8, 0])  # verde
    o3d.visualization.draw_geometries([pc_orig, pc_clean],
        window_name="Original (rojo)  vs  Filtrada (verde)")

# ──────────────────────────────────────────────────────────
#  Programa principal
# ──────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Filtra outliers con SOR+ROR y conserva "
                    "solo el clúster principal (DBSCAN).")
    p.add_argument("--input",  type=Path, default="college/grid_data.ply")
    p.add_argument("--output", type=Path, default="college/grid_data_test.pcd")
    p.add_argument("--sor_k",      type=int,   default=25)
    p.add_argument("--sor_std",    type=float, default=1.8)
    p.add_argument("--ror_radius", type=float, default=0.30)
    p.add_argument("--ror_min",    type=int,   default=6)
    p.add_argument("--no_vis",     action="store_true")
    args = p.parse_args()

    # 1 · Carga
    print(f"→ Leyendo nube: {args.input}")
    pcd = o3d.io.read_point_cloud(str(args.input))
    n_before = len(pcd.points)

    # 2 · SOR + ROR
    pcd_clean, n_removed = filter_outliers(
        pcd,
        sor_k=args.sor_k,
        sor_std=args.sor_std,
        ror_radius=args.ror_radius,
        ror_min_pts=args.ror_min)
    print(f"→ SOR/ROR: {n_removed} pts eliminados "
          f"({100*n_removed/n_before:.1f} %), quedan {len(pcd_clean.points)}")

    # --- DBSCAN agresivo -----------------------------------
    labels = np.array(pcd_clean.cluster_dbscan(
        eps=0.15,        # radio 15 cm
        min_points=16,   # densidad mínima
        print_progress=False))

    if labels.max() >= 0:
        counts  = np.bincount(labels[labels >= 0])
        main_id = counts.argmax()
        keep    = np.where(labels == main_id)[0]
        removed_db = len(pcd_clean.points) - len(keep)
        pcd_clean = pcd_clean.select_by_index(keep)
        print(f"→ DBSCAN: {removed_db} pts de clústeres secundarios eliminados "
              f"({100*removed_db/(removed_db+len(keep)):.1f} %)")
    else:
        print("⚠️  DBSCAN no detectó clústeres densos; "
              "ajusta eps/min_points si fuese necesario.")

    # 3 · Guardado
    o3d.io.write_point_cloud(str(args.output), pcd_clean)
    ply_path = args.output.with_suffix(".ply")
    o3d.io.write_point_cloud(str(ply_path), pcd_clean)
    print(f"→ Nube filtrada guardada en {args.output} y {ply_path}")

    # 4 · Visualización
    if not args.no_vis:
        visualize_pair(pcd, pcd_clean)

if __name__ == "__main__":
    main()

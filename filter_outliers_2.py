#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_outliers.py – Limpieza adaptativa de nubes:
SOR suave + kNN-percentil (sustituye ROR) + DBSCAN adaptativo.
Guarda resultado y permite comparar original (rojo) vs filtrada (verde).
"""

import argparse
from pathlib import Path
import numpy as np
import open3d as o3d


# ──────────────────────────────────────────────────────────
#  Utilidades de filtrado
# ──────────────────────────────────────────────────────────
def sor_soft(pc: o3d.geometry.PointCloud, k: int = 25, std: float = 2.0):
    """Statistical Outlier Removal suave."""
    pc2, _ = pc.remove_statistical_outlier(nb_neighbors=k, std_ratio=std)
    return pc2


def knn_percentile_filter(pc: o3d.geometry.PointCloud,
                          k: int = 16,
                          pct: float = 97.0,
                          scale: float = 1.1,
                          z_relax: bool = False,
                          z_split: float = 2.0,
                          z_scale_hi: float = 1.3):
    """
    Mantiene puntos cuyo radio al k-ésimo vecino está por debajo
    de un umbral percentil global (con escalado opcional por altura).
    """
    P = np.asarray(pc.points)
    if len(P) == 0:
        return pc

    kdt = o3d.geometry.KDTreeFlann(pc)
    dk = np.empty(len(P), dtype=np.float32)
    for i, p in enumerate(P):
        _, _, ds = kdt.search_knn_vector_3d(p, k + 1)  # incluye el propio punto
        dk[i] = np.sqrt(ds[-1])

    thr = np.percentile(dk, pct) * scale
    if z_relax:
        # Relajación por altura (región superior más laxa)
        hi = P[:, 2] > z_split
        thr_hi = np.percentile(dk[hi], pct) * scale * z_scale_hi if np.any(hi) else thr
        keep = np.where((~hi & (dk <= thr)) | (hi & (dk <= thr_hi)))[0]
    else:
        keep = np.where(dk <= thr)[0]

    return pc.select_by_index(keep)


def dbscan_keep_central(pc: o3d.geometry.PointCloud,
                        k: int = 16,
                        mult: float = 1.6,
                        min_pts: int = 22,
                        keep_extra: int = 1,      # 0 => solo central; 1 => central + 1 mayor
                        ref: str = "bbox",        # "bbox" (centro de la nube) o "origin"
                        xy_only: bool = True):
    P = np.asarray(pc.points)
    if len(P) == 0:
        return pc

    # eps ~ mult * mediana(distancia al k-ésimo vecino)
    kdt = o3d.geometry.KDTreeFlann(pc)
    dk = np.empty(len(P), dtype=np.float32)
    for i, p in enumerate(P):
        _, _, ds = kdt.search_knn_vector_3d(p, k + 1)
        dk[i] = np.sqrt(ds[-1])
    eps = mult * float(np.median(dk))

    labels = np.array(pc.cluster_dbscan(eps=eps, min_points=min_pts, print_progress=False))
    max_lbl = labels.max()
    if max_lbl < 0:
        return pc  # todo ruido

    # índices por clúster
    clusters = [np.where(labels == l)[0] for l in range(max_lbl + 1)]
    sizes = np.array([len(idx) for idx in clusters])

    # centro de referencia
    if ref == "origin":
        c_ref = np.zeros(3, dtype=np.float64)
    else:
        bb = pc.get_axis_aligned_bounding_box()
        c_ref = (bb.get_min_bound() + bb.get_max_bound()) / 2.0

    # centroides y distancia al centro de referencia (en XY si xy_only=True)
    centroids = np.array([P[idx].mean(axis=0) for idx in clusters])
    if xy_only:
        d2 = np.sum((centroids[:, :2] - c_ref[:2])**2, axis=1)
    else:
        d2 = np.sum((centroids - c_ref)**2, axis=1)

    central_id = int(np.argmin(d2))

    # mantener: central + (keep_extra) clústeres más grandes (excluyendo el central)
    order_by_size = np.argsort(sizes)[::-1]
    keep_ids = [central_id]
    for l in order_by_size:
        if l == central_id:
            continue
        keep_ids.append(int(l))
        if len(keep_ids) >= 1 + int(keep_extra):
            break

    keep_idx = np.concatenate([clusters[l] for l in keep_ids]) if keep_ids else np.array([], dtype=int)
    return pc.select_by_index(keep_idx)


def visualize_pair(pc_orig, pc_clean):
    pc_orig = o3d.geometry.PointCloud(pc_orig)
    pc_clean = o3d.geometry.PointCloud(pc_clean)
    pc_orig.paint_uniform_color([1, 0, 0])    # rojo
    pc_clean.paint_uniform_color([0, .8, 0])  # verde
    o3d.visualization.draw_geometries(
        [pc_orig, pc_clean],
        window_name="Original (rojo)  vs  Filtrada (verde)"
    )


# ──────────────────────────────────────────────────────────
#  Programa principal
# ──────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Filtra outliers con SOR suave + kNN-percentil + DBSCAN adaptativo."
    )
    p.add_argument("--input",  type=Path, default="college/grid_data.ply")
    p.add_argument("--output", type=Path, default="college/grid_data_clean.pcd")
    # Pre-voxel opcional para estabilizar densidad antes de clustering
    p.add_argument("--pre_voxel", type=float, default=0.0, help="voxel preprocesado en metros (0=off)")

    # Parámetros SOR
    p.add_argument("--sor_k",   type=int,   default=25)
    p.add_argument("--sor_std", type=float, default=2.0)

    # Parámetros kNN-percentil (sustituye ROR)
    p.add_argument("--knn_k",     type=int,   default=16)
    p.add_argument("--knn_pct",   type=float, default=95.0)
    p.add_argument("--knn_scale", type=float, default=1.0)
    p.add_argument("--z_relax",   action="store_true")
    p.add_argument("--z_split",   type=float, default=2.0)
    p.add_argument("--z_scale",   type=float, default=1.3)

    # Parámetros DBSCAN adaptativo
    p.add_argument("--db_k",       type=int,   default=16)
    p.add_argument("--db_mult",    type=float, default=1.6)
    p.add_argument("--db_min_pts", type=int,   default=22)
    p.add_argument("--db_keep",    type=int,   default=1)
    p.add_argument("--db_min_sz",  type=int,   default=3000)

    p.add_argument("--db_keep_extra", type=int, default=0)     # 0 solo central; 1 central+1
    p.add_argument("--db_ref",        type=str, default="bbox", choices=["bbox","origin"])
    p.add_argument("--db_xy_only",    action="store_true")

    # Visualización
    p.add_argument("--no_vis", action="store_true")
    args = p.parse_args()

    # 1 · Carga
    print(f"→ Leyendo nube: {args.input}")
    pcd = o3d.io.read_point_cloud(str(args.input))
    n0 = len(pcd.points)
    print(f"  puntos: {n0}")

    # 2 · Pre-voxel opcional
    if args.pre_voxel > 0:
        pcd = pcd.voxel_down_sample(args.pre_voxel)
        print(f"→ Pre-voxel @ {args.pre_voxel:.3f} m → {len(pcd.points)} pts")

    # 3 · SOR suave
    pcd = sor_soft(pcd, k=args.sor_k, std=args.sor_std)
    print(f"→ SOR: {len(pcd.points)} pts")

    # 4 · kNN-percentil (adaptativo, opc. relajación por altura)
    pcd = knn_percentile_filter(
        pcd, k=args.knn_k, pct=args.knn_pct, scale=args.knn_scale,
        z_relax=args.z_relax, z_split=args.z_split, z_scale_hi=args.z_scale
    )
    print(f"→ kNN-percentil: {len(pcd.points)} pts")

    # 5 · DBSCAN adaptativo (conserva varios clústeres útiles)
    pcd = dbscan_keep_central(
        pcd, k=args.db_k, mult=args.db_mult, min_pts=args.db_min_pts,
        keep_extra=args.db_keep_extra, ref=args.db_ref, xy_only=args.db_xy_only
    )
    print(f"→ DBSCAN adaptativo: {len(pcd.points)} pts")

    # 6 · Guardado
    o = Path(args.output)
    o.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(o), pcd)
    o_ply = o.with_suffix(".ply")
    o3d.io.write_point_cloud(str(o_ply), pcd)
    print(f"→ Nube filtrada guardada en: {o}  y  {o_ply}")

    # 7 · Visualización
    if not args.no_vis:
        try:
            pcd_in = o3d.io.read_point_cloud(str(args.input))
            visualize_pair(pcd_in, pcd)
        except Exception as e:
            print(f"⚠️  Visualización no disponible: {e}")

if __name__ == "__main__":
    main()

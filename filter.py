#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filtra del MAPA los puntos que estén a >0.3 cm del GT.
- Sin argumentos: rutas y parámetros fijos abajo.
- Soporta GT como nube o malla (muestrea si es malla).
"""

from pathlib import Path
import numpy as np
import open3d as o3d

# ========= CONFIG =========
MAP_PATH  = Path("college/grid_data.ply")      # nube/mapa a filtrar
GT_PATH   = Path("/workspace/gt/gt_map_pc_mai.ply")                # ground truth (PCD/PLY o malla)
OUT_PATH  = Path("college/mesh_filtrado.ply")  # salida

# MAP_PATH  = Path("college/grid_data.ply")      # nube/mapa a filtrar
# GT_PATH   = Path("/workspace/gt/ncd_quad_gt.ply")                # ground truth (PCD/PLY o malla)
# OUT_PATH  = Path("college/mesh_filtrado_0p3cm_new.ply")  # salida

THR_M     = 0.1   # 0.3 m 
VOXEL     = 0.0    # downsample opcional para acelerar (0=off)
GT_SAMPLES_IF_MESH = 1_000_000  # pts a muestrear si GT es malla
# ==========================

def read_gt_as_pcd(path: Path, mesh_samples: int):
    """Lee GT como PointCloud; si es malla, muestrea puntos uniformemente."""
    gt = o3d.io.read_point_cloud(str(path))
    if len(gt.points) > 0:
        return gt
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise RuntimeError(f"GT vacío o no legible: {path}")
    return mesh.sample_points_uniformly(number_of_points=mesh_samples)

def main():
    print(f"Umbral distancia: {THR_M*100:.1f} cm  ({THR_M:.4f} m)")
    # Leer GT
    print(f"→ Leyendo GT:   {GT_PATH}")
    gt = read_gt_as_pcd(GT_PATH, GT_SAMPLES_IF_MESH)
    print(f"   GT puntos: {len(gt.points)}")

    # Leer mapa
    print(f"→ Leyendo MAPA: {MAP_PATH}")
    mp = o3d.io.read_point_cloud(str(MAP_PATH))
    if len(mp.points) == 0:
        raise RuntimeError("Mapa vacío o no legible.")

    # Voxel opcional
    if VOXEL > 0:
        gt  = gt.voxel_down_sample(VOXEL)
        n0  = len(mp.points)
        mp  = mp.voxel_down_sample(VOXEL)
        print(f"   Voxel @ {VOXEL:.3f} m → mapa {n0} → {len(mp.points)} pts ; GT {len(gt.points)} pts")

    # Recorte por bbox del GT expandida (acelera)
    bb = gt.get_axis_aligned_bounding_box()
    minb = bb.get_min_bound(); maxb = bb.get_max_bound()
    minb -= THR_M; maxb += THR_M
    mp = mp.crop(o3d.geometry.AxisAlignedBoundingBox(minb, maxb))
    print(f"   Mapa tras bbox-expandida: {len(mp.points)} pts")

    if len(mp.points) == 0:
        print("Nada que filtrar tras bbox. Guardando nube vacía.")
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(OUT_PATH), mp)
        return

    # KDTree en GT
    kdt = o3d.geometry.KDTreeFlann(gt)

    # Mantener puntos del mapa con al menos un vecino GT dentro de THR_M
    P = np.asarray(mp.points)
    keep_idx = []
    for i, p in enumerate(P):
        c, _, _ = kdt.search_radius_vector_3d(p, THR_M)
        if c > 0:
            keep_idx.append(i)
        if (i+1) % 200000 == 0:
            print(f"   chequeados {i+1}/{len(P)}...")

    mp_f = mp.select_by_index(keep_idx)
    print(f"→ Filtrado: {len(keep_idx)}/{len(P)} puntos conservados "
          f"({100.0*len(keep_idx)/max(1,len(P)):.2f}%)")

    # Guardar
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(OUT_PATH), mp_f)
    print(f"→ Guardado: {OUT_PATH}")

if __name__ == "__main__":
    main()

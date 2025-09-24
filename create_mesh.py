#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Requisitos: pip install open3d numpy scipy scikit-image trimesh

import numpy as np
import open3d as o3d
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import marching_cubes
import trimesh
import os
import tempfile
import gc

# ====== CONFIGURACIÓN ======
IN_PATH  = "/workspace/gt/gt_map_pc_mai.ply"   # o .pcd
OUT_PATH = "/workspace/map.stl"
VOXEL    = 0.05     # m (objetivo de resolución)
DS       = 0.00     # m (downsample; 0 = sin downsample)
SMOOTH   = 0.00     # m (sigma gauss; 0 = sin suavizado)  # desactiva para ahorrar RAM
MARGIN   = 0.05     # m de margen
MC_STEP  = 1        # ≥2 reduce vértices y RAM en MC
# ===========================

def pc_to_stl():
    if not os.path.isfile(IN_PATH):
        raise FileNotFoundError(IN_PATH)

    pcd = o3d.io.read_point_cloud(IN_PATH)
    if len(pcd.points) == 0:
        raise ValueError("Nube vacía")

    if DS > 0:
        pcd = pcd.voxel_down_sample(DS)

    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.shape[0] < 50:
        raise ValueError("Muy pocos puntos tras preprocesado")

    mn = pts.min(axis=0) - MARGIN
    mx = pts.max(axis=0) + MARGIN
    size = mx - mn

    v = float(VOXEL)
    dims = np.ceil(size / v).astype(int) + 1
    nx, ny, nz = map(int, dims)
    cells_m = nx*ny*nz/1e6
    print(f"Grid: {nx} x {ny} x {nz}  (~{cells_m:.2f} M celdas)")

    # Rasterización a ocupación uint8 para ahorrar (bool usa 1 byte igual, pero uint8 facilita memmap)
    idx = np.floor((pts - mn) / v).astype(np.int32)
    idx = np.clip(idx, [0, 0, 0], [nx - 1, ny - 1, nz - 1])

    # Memoria: occ en disco (memmap) para no duplicar en RAM
    tmpdir = tempfile.mkdtemp(prefix="sdf_")
    occ_path = os.path.join(tmpdir, "occ.uint8")
    occ = np.memmap(occ_path, dtype=np.uint8, mode="w+", shape=(nx, ny, nz))
    occ[:] = 0
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    del idx
    gc.collect()

    # SDF unsigned exterior en memmap float32
    dist_out_path = os.path.join(tmpdir, "dist_out.f32")
    dist_out = np.memmap(dist_out_path, dtype=np.float32, mode="w+", shape=(nx, ny, nz))
    # SciPy devuelve float64; asignamos sobre float32 para truncar sin duplicar grandes buffers
    _do = distance_transform_edt(occ == 0, sampling=v)  # float64 temporal
    dist_out[:] = _do
    del _do
    gc.collect()

    # SDF unsigned interior
    dist_in_path = os.path.join(tmpdir, "dist_in.f32")
    dist_in = np.memmap(dist_in_path, dtype=np.float32, mode="w+", shape=(nx, ny, nz))
    _di = distance_transform_edt(occ == 1, sampling=v)
    dist_in[:] = _di
    del _di, occ
    gc.collect()

    # SDF firmado en el mismo buffer de dist_out para evitar otro array
    sdf_path = os.path.join(tmpdir, "sdf.f32")
    # Reutiliza dist_out como SDF
    sdf = dist_out
    sdf[:] = sdf - dist_in  # in-place
    del dist_in
    gc.collect()

    # Suavizado opcional in-place (consume memoria si sigma>0)
    if SMOOTH > 0.0:
        gaussian_filter(sdf, sigma=SMOOTH / v, output=sdf)

    # Marching Cubes con step_size>1 reduce RAM/CPU
    verts, faces, _, _ = marching_cubes(sdf, level=0.0, spacing=(v, v, v), step_size=MC_STEP)
    # Libera SDF inmediatamente
    del sdf, dist_out
    gc.collect()

    verts = verts.astype(np.float32, copy=False)
    verts += mn.astype(np.float32)

    if verts.shape[0] == 0 or faces.shape[0] == 0:
        raise RuntimeError("Marching Cubes no encontró superficie")

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(OUT_PATH)
    print(f"OK -> {OUT_PATH}  |  V={len(verts)} F={len(faces)}  voxel={v:.4f} m  step={MC_STEP}")

    # Limpia temporales en disco
    try:
        for f in (occ_path, dist_out_path, dist_in_path, sdf_path):
            if os.path.exists(f):
                os.remove(f)
        os.rmdir(tmpdir)
    except Exception:
        pass

if __name__ == "__main__":
    pc_to_stl()

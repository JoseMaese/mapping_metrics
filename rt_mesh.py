#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reconstrucción por bloques con VTK FlyingEdges + ExtractVOI y visor en “tiempo real”.
Modo banda estrecha (negativo sólo en centros ocupados). Sin EDT.

Requisitos:
  pip install open3d numpy vtk
  (opcional para visor rápido) pip install pyvista

Si no hay PyVista, el visor se desactiva automáticamente.
"""

import os
import numpy as np
import open3d as o3d
import vtk
from vtk.util import numpy_support as vtknp

# ======================= CONFIG =======================
IN_PATH   = "/workspace/gt/gt_map_pc_mai.ply"   # PLY/PCD
OUT_STL   = "/workspace/map_2.stl"

VOXEL     = 0.03    # m
MARGIN    = 0.03    # m
DS        = 0.00    # m (0=off)

# Banda estrecha
BAND_FACTOR  = 1.10      # ~1.1*VOXEL
KEEP_RATIO   = 0.00      # 0..1 (0=cualquier hit)
BAND_SMOOTH  = 0.00      # m (Gauss leve en VOI, 0=off)

# Tiling
BLOCK_VOX    = 64        # voxels por eje
PAD_VOX      = 1         # padding del VOI

# Limpieza
CLEAN_TOL    = 1e-9

# Visor
LIVE_VIEW    = True      # activar visor
BG_COLOR     = (0.08, 0.08, 0.1)
WINDOW_SIZE  = (960, 600)
# ======================================================

# PyVista opcional
try:
    import pyvista as pv
    pv.global_theme.allow_empty_mesh = True   
    PV_OK = True
except Exception:
    PV_OK = False
    LIVE_VIEW = False


def load_points(path, ds_voxel):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    p = o3d.io.read_point_cloud(path)
    if len(p.points) == 0:
        raise ValueError("Nube vacía")
    if ds_voxel and ds_voxel > 0:
        p = p.voxel_down_sample(ds_voxel)
    pts = np.asarray(p.points, dtype=np.float32)
    if pts.shape[0] < 50:
        raise ValueError("Muy pocos puntos")
    return pts


def grid_from_points(pts, voxel, margin):
    mn = pts.min(axis=0) - margin
    mx = pts.max(axis=0) + margin
    size = mx - mn
    v = float(voxel)
    nx, ny, nz = (np.ceil(size / v).astype(int) + 1).tolist()
    print(f"Grid: {nx} x {ny} x {nz}  (~{nx*ny*nz/1e6:.2f} M celdas)")
    idx = np.floor((pts - mn) / v).astype(np.int32)
    idx[:, 0] = np.clip(idx[:, 0], 0, nx - 1)
    idx[:, 1] = np.clip(idx[:, 1], 0, ny - 1)
    idx[:, 2] = np.clip(idx[:, 2], 0, nz - 1)
    return (nx, ny, nz), mn, v, idx


def make_vtk_image(nx, ny, nz, spacing, origin, fill_value):
    img = vtk.vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetSpacing(spacing, spacing, spacing)
    img.SetOrigin(origin[0], origin[1], origin[2])
    img.AllocateScalars(vtk.VTK_FLOAT, 1)
    arr = vtknp.vtk_to_numpy(img.GetPointData().GetScalars())
    arr[:] = np.float32(fill_value)
    return img, arr


def linear_indices_from_xyz(idx, dims):
    nx, ny, nz = dims
    return idx[:, 0].astype(np.int64) + idx[:, 1].astype(np.int64) * nx + idx[:, 2].astype(np.int64) * (nx * ny)


def select_occupied_voxels(idx, dims, keep_ratio):
    lin = linear_indices_from_xyz(idx, dims)
    uniq, counts = np.unique(lin, return_counts=True)
    if counts.size == 0:
        return np.empty((0,), dtype=np.int64)
    max_hits = int(counts.max())
    thr = 1 if (not keep_ratio or keep_ratio <= 0.0) else max(1, int(round(max_hits * float(keep_ratio))))
    return uniq[counts >= thr]


def group_by_blocks(selected_lin, dims, block_vox):
    nx, ny, nz = dims
    blocks = {}
    if selected_lin.size == 0:
        return blocks
    x = selected_lin % nx
    y = (selected_lin // nx) % ny
    z = (selected_lin // (nx * ny)) % nz
    bx = (x // block_vox).astype(np.int32)
    by = (y // block_vox).astype(np.int32)
    bz = (z // block_vox).astype(np.int32)
    for i in range(selected_lin.size):
        k = (int(bx[i]), int(by[i]), int(bz[i]))
        if k not in blocks:
            blocks[k] = []
        blocks[k].append(int(selected_lin[i]))
    for k in list(blocks.keys()):
        blocks[k] = np.array(blocks[k], dtype=np.int64)
    return blocks


def lin_extents(lin_idx, dims):
    nx, ny, nz = dims
    x = lin_idx % nx
    y = (lin_idx // nx) % ny
    z = (lin_idx // (nx * ny)) % nz
    return int(x.min()), int(x.max()), int(y.min()), int(y.max()), int(z.min()), int(z.max())


def main():
    pts = load_points(IN_PATH, DS)
    (nx, ny, nz), origin, h, ijk = grid_from_points(pts, VOXEL, MARGIN)
    band = float(BAND_FACTOR) * h

    img, buf_flat = make_vtk_image(nx, ny, nz, h, origin, fill_value=+band)
    buf3d = buf_flat.reshape((nz, ny, nx))
    selected_lin = select_occupied_voxels(ijk, (nx, ny, nz), KEEP_RATIO)
    if selected_lin.size == 0:
        raise RuntimeError("Sin ocupación tras umbral")

    # marcar negativos
    x = selected_lin % nx
    y = (selected_lin // nx) % ny
    z = (selected_lin // (nx * ny)) % nz
    buf3d[z, y, x] = -band

    blocks = group_by_blocks(selected_lin, (nx, ny, nz), BLOCK_VOX)

    # pipeline de acumulación
    app = vtk.vtkAppendPolyData()

    # visor en vivo (pyvista)
    if LIVE_VIEW and PV_OK:
        plotter = pv.Plotter(window_size=WINDOW_SIZE)
        plotter.set_background(color=BG_COLOR)

        # crea un placeholder vacío
        mesh_pv = pv.PolyData()
        actor = plotter.add_mesh(
            mesh_pv, color="white", smooth_shading=False,
            lighting=True, show_edges=False
        )
        plotter.add_axes()
        plotter.show(auto_close=False)
        plotter.enable_parallel_projection()

    # por bloque
    for (bx, by, bz), lin_in_block in blocks.items():
        # bbox del bloque
        x0 = bx * BLOCK_VOX; y0 = by * BLOCK_VOX; z0 = bz * BLOCK_VOX
        x1 = min((bx + 1) * BLOCK_VOX - 1, nx - 1)
        y1 = min((by + 1) * BLOCK_VOX - 1, ny - 1)
        z1 = min((bz + 1) * BLOCK_VOX - 1, nz - 1)
        # bbox ajustado al contenido + padding
        bx0, bx1, by0, by1, bz0, bz1 = lin_extents(lin_in_block, (nx, ny, nz))
        x0 = max(x0, bx0 - PAD_VOX); x1 = min(x1, bx1 + PAD_VOX)
        y0 = max(y0, by0 - PAD_VOX); y1 = min(y1, by1 + PAD_VOX)
        z0 = max(z0, bz0 - PAD_VOX); z1 = min(z1, bz1 + PAD_VOX)

        voi = vtk.vtkExtractVOI()
        voi.SetInputData(img)
        voi.SetVOI(int(x0), int(x1), int(y0), int(y1), int(z0), int(z1))
        voi.Update()

        if BAND_SMOOTH and BAND_SMOOTH > 0.0:
            sigma_px = BAND_SMOOTH / h
            smoother = vtk.vtkImageGaussianSmooth()
            smoother.SetInputConnection(voi.GetOutputPort())
            smoother.SetStandardDeviation(sigma_px)
            smoother.SetDimensionality(3)
            smoother.Update()
            source = smoother.GetOutputPort()
        else:
            source = voi.GetOutputPort()

        fe = vtk.vtkFlyingEdges3D()
        fe.SetInputConnection(source)
        fe.ComputeNormalsOff()
        fe.SetValue(0, 0.0)
        fe.Update()

        poly = vtk.vtkPolyData()
        poly.ShallowCopy(fe.GetOutput())
        if poly.GetNumberOfPoints() == 0:
            continue

        app.AddInputData(poly)
        app.Update()

        # actualizar visor
        if LIVE_VIEW and PV_OK:
            # reutiliza actor: actualiza fuente del mapper
            updated = app.GetOutput()
            actor.mapper.SetInputData(updated)
            plotter.render()

    # limpieza y guardado
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(app.GetOutputPort())
    clean.SetTolerance(CLEAN_TOL)
    clean.Update()

    tris = vtk.vtkTriangleFilter()
    tris.SetInputConnection(clean.GetOutputPort())
    tris.Update()

    stl = vtk.vtkSTLWriter()
    stl.SetFileName(OUT_STL)
    stl.SetFileTypeToBinary()
    stl.SetInputConnection(tris.GetOutputPort())
    ok = stl.Write()
    if not ok:
        raise RuntimeError("Error al escribir STL")

    if LIVE_VIEW and PV_OK:
        plotter.update()  # frame final
        # deja la ventana abierta para inspección
        plotter.show(auto_close=True)

    print(f"OK -> {OUT_STL}")


if __name__ == "__main__":
    main()

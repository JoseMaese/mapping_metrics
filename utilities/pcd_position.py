#!/usr/bin/env python3
import open3d as o3d
import numpy as np

# --- Cargar datos ---
pc_gt  = o3d.io.read_point_cloud("gt/new-college-29-01-2020-1cm-resolution-1stSection.ply")
mesh   = o3d.io.read_triangle_mesh("college/mesh.stl")
pc_pred = mesh.sample_points_uniformly(5_000)  # pred para mover

# --- ROI y downsample para visualización ---
bb = pc_pred.get_axis_aligned_bounding_box()
bb = bb.scale(2.0, bb.get_center())            # ampliar 2× la caja
pc_gt_view = pc_gt.crop(bb)
pc_gt_view = pc_gt_view.voxel_down_sample(0.05)  # 5 cm para ir rápido

# Colores
pc_gt_view.paint_uniform_color([0, 1, 0])   # verde
pc_pred.paint_uniform_color([1, 0, 0])      # rojo

T = np.eye(4)

# --- Utilidades ---
def rot_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis
    R = np.array([
        [c + x*x*(1-c),     x*y*(1-c) - z*s, x*z*(1-c) + y*s, 0],
        [y*x*(1-c) + z*s,   c + y*y*(1-c),   y*z*(1-c) - x*s, 0],
        [z*x*(1-c) - y*s,   z*y*(1-c) + x*s, c + z*z*(1-c),   0],
        [0, 0, 0, 1]
    ])
    return R

def translation_matrix(dx=0, dy=0, dz=0):
    T = np.eye(4)
    T[:3, 3] = [dx, dy, dz]
    return T

# --- Interactivo ---
def run_interactive():
    global pc_pred, T, pc_gt_view

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Mover PCD", width=960, height=540)
    vis.add_geometry(pc_gt_view)
    vis.add_geometry(pc_pred)
    opt = vis.get_render_option()
    opt.point_size = 2.0

    def update_and_refresh(transform):
        nonlocal vis
        global T, pc_pred
        pc_pred.transform(transform)
        T = transform @ T
        vis.update_geometry(pc_pred)
        vis.poll_events()
        vis.update_renderer()
        return False

    step = 0.05
    angle = np.deg2rad(1)

    # Traducción
    vis.register_key_callback(ord("W"), lambda v: update_and_refresh(translation_matrix(0, 0, +step)))
    vis.register_key_callback(ord("S"), lambda v: update_and_refresh(translation_matrix(0, 0, -step)))
    vis.register_key_callback(ord("A"), lambda v: update_and_refresh(translation_matrix(-step, 0, 0)))
    vis.register_key_callback(ord("D"), lambda v: update_and_refresh(translation_matrix(+step, 0, 0)))
    vis.register_key_callback(ord("Q"), lambda v: update_and_refresh(translation_matrix(0, +step, 0)))
    vis.register_key_callback(ord("E"), lambda v: update_and_refresh(translation_matrix(0, -step, 0)))

    # Rotación
    vis.register_key_callback(ord("J"), lambda v: update_and_refresh(rot_matrix([0, 1, 0], +angle)))
    vis.register_key_callback(ord("L"), lambda v: update_and_refresh(rot_matrix([0, 1, 0], -angle)))
    vis.register_key_callback(ord("I"), lambda v: update_and_refresh(rot_matrix([1, 0, 0], +angle)))
    vis.register_key_callback(ord("K"), lambda v: update_and_refresh(rot_matrix([1, 0, 0], -angle)))
    vis.register_key_callback(ord("U"), lambda v: update_and_refresh(rot_matrix([0, 0, 1], +angle)))
    vis.register_key_callback(ord("O"), lambda v: update_and_refresh(rot_matrix([0, 0, 1], -angle)))

    # Imprimir matriz y salir
    vis.register_key_callback(ord("Z"), lambda v: (print("\nMatriz actual:\n", T), False)[1])
    vis.register_key_callback(ord("X"), lambda v: (print("\nSaliendo. Matriz final:\n", T), vis.destroy_window())[1])

    print("""
Controles:
  w/s → mover eje Z
  a/d → mover eje X
  q/e → mover eje Y
  j/l → rotar en Y
  i/k → rotar en X
  u/o → rotar en Z
  z   → imprimir matriz
  x   → salir
""")
    vis.run()

run_interactive()

#!/usr/bin/env python3
"""
Convierte /workspace/college/mesh.stl a:
- malla PLY:      /workspace/college/mesh.ply
- nube 1 cm PLY:  /workspace/college/mesh_points_1cm.ply
- nube 1 cm PCD:  /workspace/college/mesh_points_1cm.pcd
Requiere: pip install open3d
"""

from pathlib import Path
import open3d as o3d

# --- Parámetros fijos ---
INPUT_STL = Path("/workspace/college/mesh.stl")
SCALE_TO_METERS = 1.0   # usa 0.001 si el STL está en mm, 0.01 si está en cm
RES_CM = 1.0            # resolución objetivo en cm
OVERSAMPLE = 10.0        # factor de sobre-muestreo antes del voxel downsample
MAX_POINTS = 8_000_000  # límite para evitar picos de RAM
# ------------------------

def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh

def main():
    if not INPUT_STL.exists():
        raise FileNotFoundError(f"No existe: {INPUT_STL}")

    mesh = o3d.io.read_triangle_mesh(str(INPUT_STL))
    if mesh.is_empty():
        raise RuntimeError("La malla está vacía o no se pudo leer.")

    if SCALE_TO_METERS != 1.0:
        mesh.scale(SCALE_TO_METERS, center=mesh.get_center())

    mesh = clean_mesh(mesh)

    out_dir = INPUT_STL.parent
    base = INPUT_STL.stem

    # Guardar malla en PLY
    mesh_ply = out_dir / f"{base}.ply"
    o3d.io.write_triangle_mesh(str(mesh_ply), mesh, write_ascii=False, print_progress=True)

    # Muestreo de superficie -> nube densa
    voxel_size = RES_CM / 100.0  # metros
    area = mesh.get_surface_area()  # m^2
    est_points = int(min(max(area / (voxel_size**2) * OVERSAMPLE, 100_000), MAX_POINTS))
    dense_pcd = mesh.sample_points_uniformly(number_of_points=est_points)

    # Voxel downsample a la resolución objetivo
    pcd_1cm = dense_pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_1cm.estimate_normals()

    # Guardar nubes
    pcd_ply = out_dir / f"{base}_points_5cm.ply"
    pcd_pcd = out_dir / f"{base}_points_5cm.pcd"
    o3d.io.write_point_cloud(str(pcd_ply), pcd_1cm, write_ascii=False, print_progress=True)
    o3d.io.write_point_cloud(str(pcd_pcd), pcd_1cm, write_ascii=False, print_progress=True)

    print(f"Área superficial: {area:.4f} m^2")
    print(f"Puntos previos: {len(dense_pcd.points)}")
    print(f"Puntos 1 cm:    {len(pcd_1cm.points)}")
    print(f"PLY malla:      {mesh_ply}")
    print(f"PLY puntos:     {pcd_ply}")
    print(f"PCD puntos:     {pcd_pcd}")

if __name__ == "__main__":
    main()

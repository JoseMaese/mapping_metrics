import open3d as o3d

pc = o3d.io.read_point_cloud("college/college_00.pcd")

print(f"→ Nube cargada con {len(pc.points)} puntos")
print("→ Visualizando ventana interactiva...")
o3d.visualization.draw_geometries(
    [pc],
    window_name="Colored Prediction",
    width=1280,
    height=720,
    point_show_normal=False
)

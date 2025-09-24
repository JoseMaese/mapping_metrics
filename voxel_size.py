import numpy as np
import open3d as o3d

# ---------- 1. Cargar la nube ----------
pcd_file = "college/college.pcd"
pcd = o3d.io.read_point_cloud(pcd_file)
points = np.asarray(pcd.points)
n_total = points.shape[0]

print(f"Número total de puntos en la nube: {n_total}")

# ---------- 2. Muestreo aleatorio ----------
N_SAMPLE = 23879486
replace = n_total < N_SAMPLE          # Rellenar si la nube tiene <100k puntos
sample_idx = np.random.choice(n_total, N_SAMPLE, replace=replace)
sample_pts = points[sample_idx]

# ---------- 3. KD‑Tree y distancias ----------
pcd_sample = o3d.geometry.PointCloud()
pcd_sample.points = o3d.utility.Vector3dVector(sample_pts)
kdtree = o3d.geometry.KDTreeFlann(pcd_sample)

dists = np.empty(N_SAMPLE)
for i in range(N_SAMPLE):
    # search_knn_vector_3d devuelve el punto en sí (k=1) y el vecino más cercano (k=2)
    _, idx, dist2 = kdtree.search_knn_vector_3d(pcd_sample.points[i], 2)
    dists[i] = np.sqrt(dist2[1])      # dist^2 → dist

# ---------- 4. Estadísticas ----------
mean_dist = dists.mean()
median_dist = np.median(dists)
std_dist = dists.std()

print(f"Distancia media al vecino más cercano : {mean_dist:.4f} m")
print(f"Distancia mediana                      : {median_dist:.4f} m")
print(f"Desviación estándar                    : {std_dist:.4f} m")

# ---------- 5. Interpretación rápida ----------
# Para nubes voxelizadas de forma regular, la distancia a NN tiende a ser ≈ (0.5–0.75)·voxel_size
voxel_est = mean_dist / 0.6           # valor heurístico intermedio
print(f"Voxel size aproximado ≈ {voxel_est:.4f} m (heurístico)")

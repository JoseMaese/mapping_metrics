import numpy as np
import matplotlib
matplotlib.use("Agg")                 # evita X11
import matplotlib.pyplot as plt

# Resoluciones para el eje (m, se rotulan como cm)
res_all = np.array([0.3, 0.2, 0.1, 0.05])   # 4 ticks
res_vdb = np.array([0.3, 0.2, 0.1])         # solo 3 puntos

# --- VDB-GPDF ---
mu_vdb = np.array([187, 317, 656]) 
sd_vdb = np.array([34, 65, 132])

# --- DB-TSDF ---
mu_no = np.array([137.8, 138.4, 140.8, 154.7])
sd_no = np.array([20.6, 12.7, 8.6, 16.9])

mu_ds2 = np.array([66.8, 67.7, 70.0, 91.1])
sd_ds2 = np.array([7.2, 7.4, 6.2, 5.9])

# mu_ds3 = np.array([44.6, 48.2, 56.6, 70.8])
# sd_ds3 = np.array([2.9, 3.8, 11.9, 14.1])

plt.figure(figsize=(7,4))

# VDB-GPDF (azul continuo, solo 3 puntos)
plt.errorbar(res_vdb, mu_vdb, yerr=sd_vdb, marker='o', ms=4,
             capsize=3, lw=1.8, color='C0', label="VDB-GPDF")

# DB-TSDF (rojo continuo y discontinuo)
plt.errorbar(res_all, mu_no,  yerr=sd_no,  marker='o', ms=4,
             capsize=3, lw=1.5, color='C3', linestyle='-',  label="DB-TSDF No DS")
plt.errorbar(res_all, mu_ds2, yerr=sd_ds2, marker='o', ms=4,
             capsize=3, lw=1.5, color='C3', linestyle='--', label="DB-TSDF DS=2")
# plt.errorbar(res_all, mu_ds3, yerr=sd_ds3, marker='o', ms=4,
#              capsize=3, lw=1.5, color='C3', linestyle='-.', label="DB-TSDF DS=3")

plt.xlabel("Voxel size (cm)")
plt.ylabel("Update time (ms)")
plt.title("Runtime vs. voxel size")
plt.grid(True, alpha=0.3)

# Ticks en cm y eje X invertido
plt.xticks(res_all, [f"{x:.2f}" for x in res_all])
plt.gca().invert_xaxis()

plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.98), frameon=False)
plt.tight_layout()
plt.savefig("runtime_vs_downsampling.png", dpi=600)
print("Figura guardada en runtime_vs_downsampling.png")

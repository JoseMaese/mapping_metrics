import open3d as o3d
from pathlib import Path

# Rutas
inp = Path("/workspace/college/diag_mapa_1m.pcd")
out = inp.with_suffix(".ply")

# Leer con API tensor (conserva atributos extra)
tpc = o3d.t.io.read_point_cloud(str(inp))

# Unificar/asegurar el atributo 'intensity'
candidate_names = [
    "intensity", "intensities", "Intensity", "reflectance",
    "scalar_Intensity", "ScalarField", "I"
]

int_arr = None
for name in candidate_names:
    try:
        int_arr = tpc.point[name]
        if name != "intensity":
            tpc.point["intensity"] = int_arr
        break
    except (KeyError, RuntimeError):
        pass

# Si existe, normalízalo a Float32 y forma (N,1)
if int_arr is not None:
    arr = tpc.point["intensity"]
    if arr.dtype != o3d.core.Dtype.Float32:
        arr = arr.to(o3d.core.Dtype.Float32)
    if arr.ndim == 1:
        arr = arr.reshape((-1, 1))
    elif arr.shape[-1] != 1:
        arr = arr[:, :1]  # forzar a 1 canal
    tpc.point["intensity"] = arr
else:
    print("Aviso: el PCD no trae un campo de intensidad reconocible; se guardará solo XYZ.")

# Escribir PLY (binario) conservando atributos
o3d.t.io.write_point_cloud(str(out), tpc, write_ascii=False)
print(f"Guardado: {out}")

# Opcional: muestra atributos presentes
print(tpc)  # imprime algo como: PointCloud with TensorMap:{positions, intensity, ...}

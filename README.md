# Entorno virtual para métricas 3D (Chamfer, ICP, Open3D)

Este proyecto necesita versiones coherentes de NumPy, SciPy, scikit‑learn y
Open3D.  Para evitar conflictos con el Python del sistema, se incluye
`setup_env.sh`, que crea un **entorno virtual** completamente aislado.

---

## Prerrequisitos

| Requisito | Versión mínima |
|-----------|----------------|
| Python    | 3.8 – 3.11     |
| Paquetes de compilación (solo si compilas desde fuente) | `build-essential`, `python3-dev` |

---

## Instalación rápida

```bash
chmod +x setup_env.sh        # hacer ejecutable
./setup_env.sh               # crea ~/env_mapping
# ó bien ./setup_env.sh /ruta/a/mi_env

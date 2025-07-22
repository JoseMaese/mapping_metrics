#!/usr/bin/env bash
# ------------------------------------------------------------
# Crea un entorno virtual reproducible para las métricas 3D
# Uso:
#   chmod +x setup_env.sh
#   ./setup_env.sh                 # -> ~/env_mapping
#   ./setup_env.sh /ruta/otro_env  # -> env personalizado
# ------------------------------------------------------------
set -e

# Carpeta del entorno (por defecto ~/env_mapping)
ENV_DIR="${1:-$HOME/env_mapping}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "📦 Creando entorno virtual en: $ENV_DIR"
$PYTHON_BIN -m venv "$ENV_DIR"

echo "⏳ Activando entorno …"
# shellcheck source=/dev/null
source "$ENV_DIR/bin/activate"

echo "⬆️  Actualizando pip / wheel / setuptools …"
pip install --upgrade pip wheel setuptools

echo "🔧 Instalando dependencias numéricas compatibles …"
pip install \
  numpy==1.26.4 \
  scipy==1.11.4 \
  scikit-learn==1.3.2 \
  open3d==0.18.0 \
  matplotlib==3.8.4

echo -e "\n✅ Entorno listo."
echo "Para activarlo manualmente más tarde:"
echo "  source \"$ENV_DIR/bin/activate\""

# -------------------- Dockerfile --------------------
# PyTorch + CUDA 11.8 + cuDNN9
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

# Dependencias básicas + git
RUN apt-get update && apt-get install -y \
        libgl1-mesa-glx libegl1 libxext6 libxrender1 \
        libglvnd0 libglib2.0-0 x11-apps git wget && \
    rm -rf /var/lib/apt/lists/*

# Python: Kaolin 0.17 + Open3D
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir open3d && \
    pip install --no-cache-dir kaolin==0.17.0 \
      -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu118.html && \
    pip cache purge

WORKDIR /workspace

# === Descargar mapping_metrics automáticamente ===
RUN git clone --depth 1 https://github.com/JoseMaese/mapping_metrics.git /workspace/mapping_metrics && \
    if [ -f /workspace/mapping_metrics/requirements.txt ]; then \
        pip install --no-cache-dir -r /workspace/mapping_metrics/requirements.txt; \
    fi

# Opcional: añadir al PYTHONPATH
ENV PYTHONPATH="/workspace/mapping_metrics:${PYTHONPATH}"

CMD ["/bin/bash"]
# ----------------------------------------------------

<h1 align="center"><a href="https://JoseMaese.github.io/mapping_metrics/" style="text-decoration:none;color:inherit;">Mapping Metrics</a></h1>

<p align="center">
  <a href="https://JoseMaese.github.io/mapping_metrics/" style="
    display:inline-block;padding:10px 18px;
    border:2px solid #003771;border-radius:999px;
    color:#003771;text-decoration:none;font-weight:600;">
    Visit website →
  </a>
</p>

This repository provides Python utilities to evaluate **runtime efficiency** and **3D reconstruction quality** for TSDF-based mapping methods.
The [DB-TSDF paper](https://arxiv.org/html/2509.20081v1) uses these evaluation utilities, and the source code is available on [GitHub](https://github.com/robotics-upo/DB-TSDF).
They are adapted from the metric-evaluation procedure introduced in [SHINE-Mapping](https://github.com/PRBonn/SHINE_mapping).

## Evaluation code

The directory includes an extended evaluation script for mapping accuracy derived from Atlas and modified for DB-TSDF:

The script reads predicted meshes (`ply`, `pcd`, `stl`) and ground-truth point clouds, aligns them using multiscale ICP, applies optional bounding-box cropping, and computes metrics such as **Chamfer L1/L2**, **mean absolute error**, **precision**, **recall**, and **F-score**. Helper functions handle KD-tree nearest-neighbor queries, random sampling, and colored visualization of errors.

## Install using Docker

Follow these steps to build and run mapping_metrics inside a Docker container:

```bash
Clone the repository:
git clone https://github.com/robotics-upo/mapping_metrics.git
cd mapping_metrics
```

Build the Docker image:

```bash
docker build -t mapping_metrics .
```

Allow Docker to access the X server if GUI visualization is required:
```bash
xhost +local:docker
```

Run the container:

```bash
docker run -it
--env="DISPLAY"
--env="QT_X11_NO_MITSHM=1"
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"
--name mapping_metrics_container
mapping_metrics
```

The Dockerfile sets up the full Python environment and downloads all dependencies automatically, allowing direct execution of the evaluation scripts inside the container.
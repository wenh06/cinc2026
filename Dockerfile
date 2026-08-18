# https://hub.docker.com/r/pytorch/pytorch
# pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime: Python 3.11.14, Ubuntu 22.04.5 LTS
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
# NOTE historical base images (for reference):
# pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime  Python 3.10.13  Ubuntu 20.04.6 LTS
# pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime  Python 3.10.14  Ubuntu 22.04.4 LTS
# pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime  Python 3.11.10  Ubuntu 22.04.5 LTS
# pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime  Python 3.11.13  Ubuntu 22.04.3 LTS
# pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime  Python 3.11.14  Ubuntu 22.04.5 LTS
# pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel    Python 3.11.14  Ubuntu 22.04.5 LTS (nvcc + gcc included)
#
# runtime images have NO nvcc / gcc; devel images include the full CUDA toolkit.
# We use runtime because we do not compile custom CUDA extensions.

# ── Build-time toggle ─────────────────────────────────────────────────────────
# TORCH_ECG_SOURCE=pypi    (default) → pip install torch-ecg  (latest PyPI release)
# TORCH_ECG_SOURCE=github            → pip install git+https://…@dev  (dev branch)
# Usage:
#   docker build .                                          # use PyPI release
#   docker build --build-arg TORCH_ECG_SOURCE=github .     # use dev branch
ARG TORCH_ECG_SOURCE=github
ARG PHI_MODEL_DOWNLOAD=1
ARG PHI_MEGA_URL=

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ── Cache / model directory paths ─────────────────────────────────────────────
# NOTE: Since CinC 2025 the Challenge runs submissions via Apptainer, which does
# NOT inherit Dockerfile ENV at runtime.  Runtime paths in team_code.py are
# resolved programmatically; these ENV vars are kept for local Docker convenience.
ENV HUGGINGFACE_HUB_CACHE=/challenge/cache/revenger_model_dir
ENV HF_HUB_CACHE=/challenge/cache/revenger_model_dir
ENV MODEL_CACHE_DIR=/challenge/cache/revenger_model_dir
ENV DATA_CACHE_DIR=/challenge/cache/revenger_data_dir
ENV TEST_DATA_CACHE_DIR=/challenge/cache/revenger_action_test_data_dir
ENV GIT_CLONE_DIR=/challenge/cache/git_clone_dir

ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PHI_MODEL_DOWNLOAD=$PHI_MODEL_DOWNLOAD
ENV PHI_MEGA_URL=$PHI_MEGA_URL


# ── Diagnostics ───────────────────────────────────────────────────────────────
RUN cat /etc/os-release
RUN python --version
# nvcc / gcc absent in runtime images — these lines are no-ops if not found
RUN if command -v nvcc >/dev/null 2>&1; then nvcc --version; fi
RUN if command -v gcc  >/dev/null 2>&1; then gcc  --version; fi

# Since 2025/2026 the Challenge GPU pool includes:
#   NVidia Ampere A30 (compute capability 8.0, 24 GB VRAM)
#   NVidia RTX 6000 Ada Generation (compute capability 8.6, 48 GB VRAM)


## The MAINTAINER instruction sets the author field of the generated images.
LABEL maintainer="wenh06@gmail.com"


# ── System packages ───────────────────────────────────────────────────────────
# build-essential + ninja-build: for C-extension pip packages (pycurl, pyedflib …)
# libsm6 libxext6 libxrender1: required by opencv-python on headless servers
# libsndfile1: required by biosppy and other audio libs
# awscli dependencies (curl, unzip) are already included below
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ninja-build \
        git \
        ffmpeg libsm6 libxext6 libxrender1 \
        libsndfile1 \
        megatools \
        unzip wget curl \
        vim tree \
    && rm -rf /var/lib/apt/lists/*


## DO NOT EDIT the 3 lines.
RUN mkdir /challenge
COPY ./requirements-docker.txt /challenge
WORKDIR /challenge


RUN mkdir -p $MODEL_CACHE_DIR \
    && mkdir -p $DATA_CACHE_DIR \
    && mkdir -p $TEST_DATA_CACHE_DIR \
    && mkdir -p $GIT_CLONE_DIR


RUN which python

# list packages pre-installed in the base image
RUN pip list

# torch and related packages (torchvision, torchaudio, etc.) are already in the base image

RUN python -m pip install --upgrade pip setuptools wheel

# ── Install torch-ecg ─────────────────────────────────────────────────────────
RUN if [ "$TORCH_ECG_SOURCE" = "github" ]; then \
        echo "Installing torch-ecg from GitHub (dev branch) …" \
        && pip install git+https://github.com/DeepPSP/torch_ecg.git@dev; \
    else \
        echo "Installing torch-ecg from PyPI …" \
        && pip install torch-ecg; \
    fi

# ── Other dependencies ────────────────────────────────────────────────────────
RUN pip install -r requirements-docker.txt

# list packages after installing all requirements
RUN pip list

# ── AWS CLI v2 ────────────────────────────────────────────────────────────────
# https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html
RUN curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip -q awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws
RUN aws --version && which aws


# ── Copy project ──────────────────────────────────────────────────────────────
COPY ./ /challenge


# ── Vendored third-party source and feature caches ────────────────────────────
# third_party/philosophers-stone/src, data/spectral_features and
# data/phi_cache are ordinary tracked files.  The official build context has
# no .git (git submodules fail there) and no guaranteed MEGA/HF access, so they
# are baked into the image by the COPY above instead of being fetched at build
# time.  The Phi checkpoint itself is still downloaded at build time
# (PHI_MODEL_DOWNLOAD=1) so cache-miss records can be computed on the fly.


# ── Post-build environment check ─────────────────────────────────────────────
RUN python post_docker_build.py
RUN du -sh $DATA_CACHE_DIR $TEST_DATA_CACHE_DIR $MODEL_CACHE_DIR
RUN tree $MODEL_CACHE_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Local test / run commands (not executed during image build):
#
#   docker build -t cinc2026 .
#   docker build --build-arg TORCH_ECG_SOURCE=github -t cinc2026 .
#
#   docker run -it --shm-size=10240m --gpus all \
#     -v /path/to/model:/challenge/model \
#     -v /path/to/data:/challenge/data:ro \
#     -v /path/to/output:/challenge/output \
#     cinc2026 bash
#
#   python train_model.py /challenge/data /challenge/model
#   python run_model.py   /challenge/model /challenge/data /challenge/output
#   python evaluate_model.py labels outputs scores.csv
# ─────────────────────────────────────────────────────────────────────────────

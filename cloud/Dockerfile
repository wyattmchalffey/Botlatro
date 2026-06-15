# Reproducible, provider-agnostic image for the Botlatro Phase-B pipeline.
# Builds both native extensions (Rust balatro_core, Cython hand_evaluator) and
# CPU-only torch. Runs anywhere (Hetzner, AWS, GCP, RunPod, local docker).
#
#   docker build -f cloud/Dockerfile -t botlatro:cpu .
#   docker run --rm -v "$PWD/out:/work/.data" botlatro:cpu \
#       bash cloud/run_iteration.sh --n-runs 50000 --shards 1 --shard-id 0
#
# The bottleneck is the CPU simulator, so size the HOST for cores, not GPU.
FROM python:3.12-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/work/src \
    BALATRO_NO_FORESIGHT=shuffle

# System + Rust toolchain
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        build-essential curl git pkg-config ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /work

# Python deps first (layer-cached across code changes)
RUN pip install --upgrade pip wheel \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY cloud/requirements.txt /work/cloud/requirements.txt
RUN pip install -r /work/cloud/requirements.txt

# Native extensions: copy the Rust crate + build it, then the package
COPY botlatro-core /work/botlatro-core
RUN pip install /work/botlatro-core            # maturin -> balatro_core .so
COPY . /work
RUN pip install -e .                           # cython hand_evaluator + package

# Verify the native builds at image-build time (fail fast, not on the cloud bill)
RUN python -c "import balatro_core; from balatro_ai.rules import hand_evaluator; import torch; print('image OK', torch.__version__)"

CMD ["bash"]

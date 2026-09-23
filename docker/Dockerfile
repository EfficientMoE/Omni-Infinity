FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG VDN_REF=e9204ce2330febbf36d8fee462f04a321e8217b6
ARG MOE_STORE_REF=9724e4681fe1474bdb5f4ade9906f9d8ab734c9b
ARG INSTALL_FLASH_ATTN=0

ENV CUDA_HOME=/usr/local/cuda \
    HF_HOME=/cache/huggingface \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        ninja-build \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade pip setuptools wheel

# Keep the large, stable CUDA/PyTorch layer ahead of source checkouts.
# torchvision must come from the same cu129 index: the PyPI default is cu130.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu129 \
        torch==2.13.0+cu129 \
        torchvision==0.28.0+cu129 \
    && python -c "import torch, torchvision; assert torch.__version__ == '2.13.0+cu129'; assert torchvision.__version__ == '0.28.0+cu129'"

RUN git clone https://github.com/OpenVDN/vdn-minimax-h3.git /opt/vdn-minimax-h3 \
    && git -C /opt/vdn-minimax-h3 checkout --detach "${VDN_REF}" \
    && test "$(git -C /opt/vdn-minimax-h3 rev-parse HEAD)" = "${VDN_REF}"

# Install the upstream dependency set explicitly so flash-attn-4 remains an
# opt-in build for Hopper and data-center Blackwell. Consumer Blackwell sm120
# uses FlexAttention/PyTorch kernels and does not load it.
RUN pip install --no-cache-dir --pre \
        accelerate==1.14.0 \
        av==18.1.0 \
        einops==0.8.2 \
        huggingface-hub==1.32.0 \
        numpy==2.5.2 \
        nvtx==0.2.16 \
        omegaconf==2.3.1 \
        peft==0.20.0 \
        pillow==12.3.0 \
        pytest==9.1.1 \
        pytest-timeout==2.4.0 \
        pyyaml==6.0.3 \
        safetensors==0.8.0 \
        torchao==0.18.0 \
        tqdm==4.70.1 \
        transformers==5.15.0 \
        triton==3.7.1 \
    && if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then \
         pip install --no-cache-dir --pre --no-build-isolation flash-attn-4==4.0.0b26; \
       elif [ "${INSTALL_FLASH_ATTN}" != "0" ]; then \
         echo "INSTALL_FLASH_ATTN must be 0 or 1" >&2; exit 2; \
       fi \
    && pip install --no-cache-dir --no-deps -e /opt/vdn-minimax-h3

# setup_diffusers.sh clones its pinned base, applies the VDN patch with
# `git am`, and installs that patched checkout editable.
RUN git config --global user.name "Omni-Infinity Docker Build" \
    && git config --global user.email "docker-build@omni-infinity.invalid" \
    && cd /opt/vdn-minimax-h3 \
    && PIP=pip bash scripts/setup_diffusers.sh

# moe-store is not published on PyPI. Pin the verified Git revision rather
# than tracking a moving branch; it is required by the h3-dense store path.
RUN pip install --no-cache-dir --no-deps \
        "git+https://github.com/EfficientMoE/moe-store.git@${MOE_STORE_REF}"

WORKDIR /workspace/omni-infinity
COPY . .
RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OMNI_INFINITY=0.1.0 \
    pip install --no-cache-dir --no-deps -e . \
    && python -c "import torch, diffusers, torchao, omni_infinity; from omni_infinity.arch import VdnRunner; assert torch.__version__ == '2.13.0+cu129'; print(torch.__version__, diffusers.__version__)"

CMD ["python"]

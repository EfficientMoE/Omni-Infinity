#!/usr/bin/env bash
# Container entrypoint for .github/workflows/nightly-gpu.yml. Runs as a
# non-root user with the repo mounted at /workspace, the model snapshot
# and moe-store mounted read-only under /models, and /output mounted from
# the host for reports and debugging artifacts.
set -euo pipefail

python - <<'EOF'
import os

import diffusers
import torch

assert torch.__version__ == "2.12.0+cu130", torch.__version__
assert diffusers.__version__ == "0.40.0", diffusers.__version__
assert torch.cuda.is_available(), "no CUDA device visible"
name = torch.cuda.get_device_name(0)
expected = os.environ.get("OMNI_CI_EXPECT_GPU")
assert not expected or name == expected, name
print(f"stack OK on {name}")
EOF

pip install -q --user --no-build-isolation --no-deps -e .

python -m pytest tests/ \
    -m "gpu or weights" \
    -q -rs \
    --timeout 3900 \
    --basetemp /output/pytest-tmp \
    --junitxml /output/nightly-gpu-report.xml

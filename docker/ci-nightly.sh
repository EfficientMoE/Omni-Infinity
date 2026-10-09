#!/usr/bin/env bash
# Container entrypoint for .github/workflows/nightly-gpu.yml. Runs as a
# non-root user with the repo mounted at /workspace, the model snapshot
# and moe-store mounted read-only under /models, and /output mounted from
# the host for reports and debugging artifacts.
#
# Each test file runs in its own pytest process: the gpu/weights tests
# hold hundreds of GB of host RAM (AdaLN host cache, pinned streaming
# buffers), and one long-lived interpreter accumulated enough across
# files to draw the host OOM killer.
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

mapfile -t files < <(
    python -m pytest tests/ -m "gpu or weights" --collect-only -q 2>/dev/null \
        | sed -n 's/::.*//p' | sort -u
)
echo "selected files: ${files[*]}"

mkdir -p /output/pytest-tmp

status=0
for file in "${files[@]}"; do
    name=$(basename "$file" .py)
    echo "::group::$name"
    python -m pytest "$file" \
        -m "gpu or weights" \
        -q -rs \
        --timeout 3900 \
        --basetemp "/output/pytest-tmp/$name" \
        --junitxml "/output/junit-$name.xml" \
        || status=1
    echo "::endgroup::"
done
exit "$status"

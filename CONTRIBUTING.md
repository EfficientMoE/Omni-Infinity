# Contributing to Omni-Infinity

Thanks for your interest in Omni-Infinity. This guide covers the development
setup, the lint/test gates, and the conventions the project follows. For a map
of the codebase see [ARCHITECTURE.md](ARCHITECTURE.md); for the longer design
notes see the [Documentation Hub](docs/README.md).

## Development setup

Omni-Infinity targets **Python 3.12** in CI (the package declares
`requires-python >=3.10`) and depends on `torch>=2.0` and `moe-store>=0.2.1`.

```bash
# CPU-only development environment (mirrors the CI unit job)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-deps 'moe-store @ git+https://github.com/EfficientMoE/moe-store.git@v0.2.1'
pip install diffusers transformers
pip install -e '.[dev]'

python -c "import omni_infinity; print(omni_infinity.__version__)"
```

The `[dev]` extra installs the test tooling and the serving dependencies
(FastAPI, uvicorn, websockets, httpx, av, soundfile, pillow), so the streaming
test modules — which import FastAPI at module scope — can be collected.

## Linting and formatting

CI pins **Ruff 0.6.9**. Run both gates before pushing:

```bash
ruff check .
ruff format --check .
```

Ruff is configured in `pyproject.toml`: `line-length = 80`, lint rules
`E, F, I, W`, and `third_party/` is excluded from both lint and format.

## Tests

```bash
pytest tests/ -m "not gpu and not weights" -q --timeout 180 \
  --cov=omni_infinity --cov-report=term-missing --cov-fail-under=80
```

- Two markers gate hardware-dependent tests: **`gpu`** (needs a CUDA device)
  and **`weights`** (needs a local checkpoint or moe-store). GitHub-hosted CI
  excludes both; coverage must stay at or above **80%**.
- Several **doc-contract tests** pin exact strings and paths in the docs —
  `tests/test_operator_docs.py` and `tests/test_cache_bench_docs.py` read
  `README.md`, and `tests/test_*_cache*.py` / `tests/test_cache_c4_doc.py`
  read the per-level `docs/caches_c*.md` files by path. When you change those
  docs, update the matching test in the same change, and keep the doc files
  where they are (benchmarks reference them by path too).
- GPU/weights reproduction recipes and the Docker-based gates live in
  [docker/README.md](docker/README.md).

## Coding standards

- New source files carry the SPDX header used throughout the tree:

  ```python
  # Copyright (c) EfficientMoE.
  # SPDX-License-Identifier: Apache-2.0
  ```

- **Never modify `third_party/`** (the vendored VDN-Minimax-H3 and patched
  Diffusers). Integration lives in `omni_infinity/` adapters (for example
  `step_overlap.py`), which fail closed if an upstream topology changes.
- Keep measurement docs honest: numbers in the docs reflect recorded runs, not
  estimates. Do not invent benchmark figures.
- Prefer small, focused changes that match the existing module layout
  described in [ARCHITECTURE.md](ARCHITECTURE.md).

## Commits and pull requests

- Follow the repository's **Conventional Commits** style with a scope, e.g.
  `feat(caches): …`, `fix(smoke): …`, `test(parity): …`, `docs: …`.
- Keep pull requests focused and make sure the **lint** and **CPU unit** jobs
  in `.github/workflows/ci.yml` pass. CI runs on pushes to `main`, on every
  pull request, and on manual dispatch.
- Reference the relevant tracking issue
  ([Omni-Infinity issues](https://github.com/EfficientMoE/Omni-Infinity/issues),
  or the RFC [MoE-Infinity#222](https://github.com/EfficientMoE/MoE-Infinity/issues/222))
  where applicable.

## License

By contributing, you agree that your contributions are licensed under the
project's [Apache-2.0](LICENSE) license.

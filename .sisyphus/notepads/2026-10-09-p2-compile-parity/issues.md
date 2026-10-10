# Issues / infra state — P2 compile-parity follow-up

## [2026-10-09] elm/gpt-5.6-sol BUDGET EXCEEDED
`AI_APICallError: API key budget exceeded`. openai/* keys also dead. Use
**anthropic/claude-fable-5** (verified working) for per-commit reviews and any
`opencode run` delegation. Task-tool subagent delegation is BROKEN here — delegate GPU
runs via `nohup opencode run -m anthropic/claude-fable-5 "$(cat PROMPTFILE)" > LOG 2>&1 &`
+ poll, or run short probes directly.

## [2026-10-09] ENVIRONMENT ISOLATION (critical for Phase 1)
Phase 1 builds NEWER torch/diffusers. You MUST NOT pip-install or mutate the shared pinned
env `/home/leyang/anaconda3` — other live sessions (e.g. p7-multigpu) and the recorded
goldens depend on torch 2.12.0+cu130 / diffusers 0.40.0. Create ISOLATED envs only
(`python -m venv /tmp/opencode/<env>` or `conda create -n <tmp>`), with a fresh
`TORCHINDUCTOR_CACHE_DIR` per experiment. Never change the default interpreter the other
sessions use. cu13x wheel availability for candidate torch versions is an open risk —
smoke sm120 + the opt-in fp8 path on any candidate stack before investing.

## [2026-10-09] GPU state
Earlier today all 6 GPUs were busy with another job; as of this setup they are ALL IDLE
(0-3 MiB, 0%). State can change — run `nvidia-smi` before EACH run; if other jobs return,
pin ONE GPU via CUDA_VISIBLE_DEVICES and never kill others' pids; leave the MPS server.
Parity on GPU 0 (RTX PRO 6000 Server Edition = golden provenance); timing on GPU 4.

## [2026-10-09] Goldens are sacred
Never silently replace `tests/fixtures/goldens/fl2va_goldens.pt`. A re-baseline is a
Phase-2 conditional step and is its OWN reviewed change with provenance recorded.

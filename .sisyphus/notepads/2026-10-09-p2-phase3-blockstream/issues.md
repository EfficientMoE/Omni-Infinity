# Issues / infra state — P2 Phase 3

## [2026-10-09] elm/gpt-5.6-sol BUDGET EXCEEDED
The model that drove ALL of P2 is dead: `AI_APICallError: API key budget exceeded`
(retries fail). openai/* keys were already dead (from P2). **anthropic/claude-fable-5
WORKS** (verified 2026-10-09 via `opencode run`). Use it for per-commit reviews and any
`opencode run` delegation.

## [2026-10-09] Task-tool subagent delegation BROKEN
The running opencode server lacks the elm provider and mis-resolves category remaps.
Do NOT use the Task tool for delegation. Delegate GPU-heavy runs ONLY via:
  cd <worktree> && nohup opencode run -m anthropic/claude-fable-5 "$(cat PROMPTFILE)" \
    > /tmp/opencode/LOG 2>&1 &
then poll the log + `git`. Short probes may be run directly.

## [2026-10-09] ALL 6 GPUs BUSY (timing deferred)
nvidia-smi: all 6 RTX PRO 6000 at 17-20 GiB used, 69-86% util, ~30 compute pids from
another job (incl. pids 459100-459127). 
- Memory headroom ~78 GiB/GPU -> correctness/capture probes OK on a SHARED GPU.
- TIMING is INVALID on a loaded GPU. DEFER the Phase-4 timing ablation until an idle GPU
  is available; check nvidia-smi before any timing run. Report timing as PENDING, never
  report contaminated numbers.
- NEVER kill others' pids; leave the system nvidia-cuda-mps-server. Pin ONE GPU per run
  via CUDA_VISIBLE_DEVICES. Parity on GPU 0 (golden provenance); timing on GPU 4 (idle).

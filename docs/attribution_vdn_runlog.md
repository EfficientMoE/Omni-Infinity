
### 2026-09-23T09:24:37.959973+00:00 — workspace and git context

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && GIT_MASTER=1 git status && GIT_MASTER=1 git diff --staged --stat && GIT_MASTER=1 git diff --stat && GIT_MASTER=1 git log -30 --oneline && GIT_MASTER=1 git log -30 --pretty=format:"%s" && GIT_MASTER=1 git branch --show-current && GIT_MASTER=1 git rev-parse --git-dir && GIT_MASTER=1 git rev-parse --git-common-dir && GIT_MASTER=1 git rev-parse --show-superproject-working-tree && (GIT_MASTER=1 git merge-base HEAD main 2>/dev/null || GIT_MASTER=1 git merge-base HEAD master 2>/dev/null) && (GIT_MASTER=1 git rev-parse --abbrev-ref @{upstream} 2>/dev/null || true) && nvidia-smi`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `19cbd258b4dadd424b4c51f7d3fa1079881ab773`
- **Wall time:** 0.318 s
- **Artifacts:** docs/attribution_vdn_runlog.md
- **Outcome:** workspace and git context: exit 0

<details><summary>Verbatim output</summary>

```text
On branch feat/vdn-minimax-h3
Your branch is up to date with 'origin/feat/vdn-minimax-h3'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
  (commit or discard the untracked or modified content in submodules)
	modified:   third_party/vdn-minimax-h3 (untracked content)

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	.sisyphus/
	generated_latents.pt

no changes added to commit (use "git add" and/or "git commit -a")
19cbd25 feat(docker): reproducible VDN environment image (#10)
f58fc27 docs: VDN-H3 16-run ablation study results (#10)
43a3f97 fix(bench): drop frame-mismatched goldens gate from omni grid rows (#10)
b0efffb docs: model-arch/optimization category section in README (#10)
8914d6c docs: VDN repro results (sm120) + fix omni ablation rows for cross-repo guard (#10)
fd7fda5 test: record vdn-hybrid golden latents (#10)
db01ba2 feat(bench): VDN ablation harness - 16-run arch/optimization grid (#10)
e5b450b fix(test): vdn parity gate uses streamed-encoder recipe for 96GB cards (#10)
a5d14e0 feat(smoke): vdn_smoke CLI + vdn-hybrid parity gate (#10)
9e0a922 feat: VdnRunner - vdn-hybrid arch over the published diffusers component (#10)
021ca05 feat: arch/ and optim/ category namespace packages (#10)
2afe14a feat: model-arch/optimization category registry (#10)
a7e5982 chore: pin OpenVDN/vdn-minimax-h3 @ e9204ce as third_party submodule (#10)
46d2adb fix(ci): ruff format inc-8 files + add fp8_scale to smoke test Namespace
b87220a docs: fused fp8 kernel (Task 2 inc 8) + deferred moe-kernels extraction plan
9b546c6 feat(smoke): --fp8-scale per_row|block A/B end-to-end (runner->store->fp8/adaln)
72701e9 Merge branch 'feat/inc8-task5-adaln' into feat/task2-inc8-fused-fp8-kernel
ccd620f Merge branch 'feat/inc8-task4-fp8-linear' into feat/task2-inc8-fused-fp8-kernel
a7549d0 Merge branch 'feat/inc8-task7-bench' into feat/task2-inc8-fused-fp8-kernel
2bcd6b8 feat(adaln): fused fp8 weight-only AdaLN projection (block scale, half-byte H2D); bf16 path untouched
0a20b17 bench(fp8): fused w8a16 vs bf16-materialize baseline vs _scaled_mm (memory/bandwidth/latency)
4b59e12 feat(fp8): ScaledFp8Linear uses block-scaled fused fp8 weight-only GEMM (no bf16 materialization)
ac2e77e feat(kernels): op-centric facade with reference fallback + OMO_KERNELS_REFERENCE seam
88900e9 feat(kernels): vendor+harden BatchGen fused fp8 weight-only Triton GEMM (autotune, bias, return fix)
ea779a6 feat(kernels): block-wise fp8 quantizer + pure-torch reference GEMM (Task 2 inc 8)
388f8a3 Merge pull request #6 from EfficientMoE/feat/task2-inc7-whole-pipeline-gate
feef136 fix(ci): load smoke script by repository path (Task 2 inc. 7)
abc1c0b docs: record whole-pipeline VRAM gate result (Task 2 inc. 7)
504b749 feat(smoke): whole-pipeline <22 GiB max_memory_allocated gate (Task 2 inc. 7)
1d4a9a9 Merge pull request #5 from EfficientMoE/feat/task2-inc6-text-encoder-streaming
feat(docker): reproducible VDN environment image (#10)
docs: VDN-H3 16-run ablation study results (#10)
fix(bench): drop frame-mismatched goldens gate from omni grid rows (#10)
docs: model-arch/optimization category section in README (#10)
docs: VDN repro results (sm120) + fix omni ablation rows for cross-repo guard (#10)
test: record vdn-hybrid golden latents (#10)
feat(bench): VDN ablation harness - 16-run arch/optimization grid (#10)
fix(test): vdn parity gate uses streamed-encoder recipe for 96GB cards (#10)
feat(smoke): vdn_smoke CLI + vdn-hybrid parity gate (#10)
feat: VdnRunner - vdn-hybrid arch over the published diffusers component (#10)
feat: arch/ and optim/ category namespace packages (#10)
feat: model-arch/optimization category registry (#10)
chore: pin OpenVDN/vdn-minimax-h3 @ e9204ce as third_party submodule (#10)
fix(ci): ruff format inc-8 files + add fp8_scale to smoke test Namespace
docs: fused fp8 kernel (Task 2 inc 8) + deferred moe-kernels extraction plan
feat(smoke): --fp8-scale per_row|block A/B end-to-end (runner->store->fp8/adaln)
Merge branch 'feat/inc8-task5-adaln' into feat/task2-inc8-fused-fp8-kernel
Merge branch 'feat/inc8-task4-fp8-linear' into feat/task2-inc8-fused-fp8-kernel
Merge branch 'feat/inc8-task7-bench' into feat/task2-inc8-fused-fp8-kernel
feat(adaln): fused fp8 weight-only AdaLN projection (block scale, half-byte H2D); bf16 path untouched
bench(fp8): fused w8a16 vs bf16-materialize baseline vs _scaled_mm (memory/bandwidth/latency)
feat(fp8): ScaledFp8Linear uses block-scaled fused fp8 weight-only GEMM (no bf16 materialization)
feat(kernels): op-centric facade with reference fallback + OMO_KERNELS_REFERENCE seam
feat(kernels): vendor+harden BatchGen fused fp8 weight-only Triton GEMM (autotune, bias, return fix)
feat(kernels): block-wise fp8 quantizer + pure-torch reference GEMM (Task 2 inc 8)
Merge pull request #6 from EfficientMoE/feat/task2-inc7-whole-pipeline-gate
fix(ci): load smoke script by repository path (Task 2 inc. 7)
docs: record whole-pipeline VRAM gate result (Task 2 inc. 7)
feat(smoke): whole-pipeline <22 GiB max_memory_allocated gate (Task 2 inc. 7)
Merge pull request #5 from EfficientMoE/feat/task2-inc6-text-encoder-streamingfeat/vdn-minimax-h3
.git
.git
388f8a3ae7f254035f7d41d10d21e2a621e785c6
origin/feat/vdn-minimax-h3
Wed Sep 23 09:24:38 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.48.01              Driver Version: 590.48.01      CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX PRO 6000 Blac...    On  |   00000000:06:00.0 Off |                    0 |
| N/A   27C    P0             65W /  600W |       3MiB /  97887MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA RTX PRO 6000 Blac...    On  |   00000000:07:00.0 Off |                    0 |
| N/A   24C    P8             34W /  600W |       3MiB /  97887MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA RTX PRO 6000 Blac...    On  |   00000000:75:00.0 Off |                    0 |
| N/A   29C    P8             34W /  600W |       3MiB /  97887MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA RTX PRO 6000 Blac...    On  |   00000000:76:00.0 Off |                    0 |
| N/A   28C    P8             34W /  600W |       3MiB /  97887MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA RTX PRO 6000 Blac...    On  |   00000000:F4:00.0 Off |                  Off |
| 30%   53C    P1            300W /  300W |   88292MiB /  97887MiB |     99%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA RTX PRO 6000 Blac...    On  |   00000000:F5:00.0 Off |                  Off |
| 30%   60C    P1            300W /  300W |   88292MiB /  97887MiB |     99%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    4   N/A  N/A         2292996      C   VLLM::EngineCore                      88282MiB |
|    5   N/A  N/A         2292840      C   VLLM::EngineCore                      88282MiB |
+-----------------------------------------------------------------------------------------+
```
</details>

### 2026-09-23 — BLOCKED-at-Task-1-ruff-QA

- **Gate:** `ruff check benchmarks/`
- **Attempts:** 2 after implementation, with an honest import-order/line-length
  correction between attempts.
- **Evidence:** Both complete outputs are preserved above under “Task 1 ruff QA
  pre-GPU” and “Task 1 ruff QA retry”. The retry still reports I001 in both new
  modules and E501 at three lines in `benchmarks/attribution_vdn.py`.
- **Impact:** Tasks 2–5 depend on the Task 1 shim/driver and therefore cannot be
  executed without violating the approved plan's dependency order.
- **Process state:** No GPU run was started and no child process remains running.
- **Outcome:** PARTIAL; stopped per the mandatory two-failure QA rule.

### 2026-09-23T09:25:48.683559+00:00 — Task 1 RED pytest

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && python -m pytest tests/test_attribution_shim.py -v`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `19cbd258b4dadd424b4c51f7d3fa1079881ab773`
- **Wall time:** 0.183 s
- **Artifacts:** docs/attribution_vdn_runlog.md
- **Outcome:** Task 1 RED pytest: exit 1

<details><summary>Verbatim output</summary>

```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /mnt/raid0nvme0/leyang/envs/vdn/bin/python
cachedir: .pytest_cache
rootdir: /mnt/raid0nvme0/leyang/Omni-Infinity
configfile: pyproject.toml
plugins: timeout-2.4.0, anyio-4.15.1
collecting ... collected 2 items

tests/test_attribution_shim.py::test_shim_imports_without_upstream_package FAILED [ 50%]
tests/test_attribution_shim.py::test_range_accounting_sums_mock_cuda_events FAILED [100%]

=================================== FAILURES ===================================
__________________ test_shim_imports_without_upstream_package __________________

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x79d74b41ed20>

    def test_shim_imports_without_upstream_package(monkeypatch):
        monkeypatch.delitem(sys.modules, "src", raising=False)
    
>       module = _load_shim()
                 ^^^^^^^^^^^^

tests/test_attribution_shim.py:32: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_attribution_shim.py:25: in _load_shim
    spec.loader.exec_module(module)
<frozen importlib._bootstrap_external>:991: in exec_module
    ???
<frozen importlib._bootstrap_external>:1128: in get_code
    ???
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <_frozen_importlib_external.SourceFileLoader object at 0x79d74b41ecf0>
path = '/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim/sitecustomize.py'

>   ???
E   FileNotFoundError: [Errno 2] No such file or directory: '/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim/sitecustomize.py'

<frozen importlib._bootstrap_external>:1186: FileNotFoundError
_________________ test_range_accounting_sums_mock_cuda_events __________________

    def test_range_accounting_sums_mock_cuda_events():
>       module = _load_shim()
                 ^^^^^^^^^^^^

tests/test_attribution_shim.py:39: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_attribution_shim.py:25: in _load_shim
    spec.loader.exec_module(module)
<frozen importlib._bootstrap_external>:991: in exec_module
    ???
<frozen importlib._bootstrap_external>:1128: in get_code
    ???
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <_frozen_importlib_external.SourceFileLoader object at 0x79d74b41fad0>
path = '/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim/sitecustomize.py'

>   ???
E   FileNotFoundError: [Errno 2] No such file or directory: '/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim/sitecustomize.py'

<frozen importlib._bootstrap_external>:1186: FileNotFoundError
=========================== short test summary info ============================
FAILED tests/test_attribution_shim.py::test_shim_imports_without_upstream_package
FAILED tests/test_attribution_shim.py::test_range_accounting_sums_mock_cuda_events
============================== 2 failed in 0.02s ===============================
```
</details>

### 2026-09-23T09:27:30.127244+00:00 — Task 1 GREEN pytest

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && python -m pytest tests/test_attribution_shim.py -v`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `19cbd258b4dadd424b4c51f7d3fa1079881ab773`
- **Wall time:** 0.173 s
- **Artifacts:** docs/attribution_vdn_runlog.md
- **Outcome:** Task 1 GREEN pytest: exit 1

<details><summary>Verbatim output</summary>

```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /mnt/raid0nvme0/leyang/envs/vdn/bin/python
cachedir: .pytest_cache
rootdir: /mnt/raid0nvme0/leyang/Omni-Infinity
configfile: pyproject.toml
plugins: timeout-2.4.0, anyio-4.15.1
collecting ... collected 2 items

tests/test_attribution_shim.py::test_shim_imports_without_upstream_package PASSED [ 50%]
tests/test_attribution_shim.py::test_range_accounting_sums_mock_cuda_events FAILED [100%]

=================================== FAILURES ===================================
_________________ test_range_accounting_sums_mock_cuda_events __________________

    def test_range_accounting_sums_mock_cuda_events():
        module = _load_shim()
        timestamps = iter((0.0, 2.0, 8.0, 10.0))
        synchronizations = []
    
        class FakeEvent:
            def record(self):
                self.timestamp = next(timestamps)
    
            def elapsed_time(self, other):
                return other.timestamp - self.timestamp
    
        accounting = module.RangeAccounting(
            event_factory=FakeEvent,
            synchronize=lambda: synchronizations.append(True),
        )
    
        with accounting.measure("step_total"):
            with accounting.measure("dense_attn"):
                pass
    
        assert synchronizations == [True]
>       assert accounting.steps == [
            {
                "dense_attn": {"total_ms": 6.0, "calls": 1},
                "step_total": {"total_ms": 10.0, "calls": 1},
            }
        ]
E       AssertionError: assert [{'dense_attn... 'calls': 0}}] == [{'dense_attn... 'calls': 1}}]
E         
E         At index 0 diff: {'dense_attn': {'total_ms': 6.0, 'calls': 1}, 'step_total': {'total_ms': 10.0, 'calls': 1}, 'linear_calls': {'total_ms': 0.0, 'calls': 0}} != {'dense_attn': {'total_ms': 6.0, 'calls': 1}, 'step_total': {'total_ms': 10.0, 'calls': 1}}
E         
E         Full diff:
E           [
E               {
E                   'dense_attn': {...
E         
E         ...Full output truncated (13 lines hidden), use '-vv' to show

tests/test_attribution_shim.py:60: AssertionError
=========================== short test summary info ============================
FAILED tests/test_attribution_shim.py::test_range_accounting_sums_mock_cuda_events
========================= 1 failed, 1 passed in 0.02s ==========================
```
</details>

### 2026-09-23T09:27:39.516652+00:00 — Task 1 GREEN pytest retry

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && python -m pytest tests/test_attribution_shim.py -v`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `19cbd258b4dadd424b4c51f7d3fa1079881ab773`
- **Wall time:** 0.163 s
- **Artifacts:** docs/attribution_vdn_runlog.md
- **Outcome:** Task 1 GREEN pytest retry: exit 0

<details><summary>Verbatim output</summary>

```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /mnt/raid0nvme0/leyang/envs/vdn/bin/python
cachedir: .pytest_cache
rootdir: /mnt/raid0nvme0/leyang/Omni-Infinity
configfile: pyproject.toml
plugins: timeout-2.4.0, anyio-4.15.1
collecting ... collected 2 items

tests/test_attribution_shim.py::test_shim_imports_without_upstream_package PASSED [ 50%]
tests/test_attribution_shim.py::test_range_accounting_sums_mock_cuda_events PASSED [100%]

============================== 2 passed in 0.01s ===============================
```
</details>

### 2026-09-23T09:27:45.931789+00:00 — Task 1 dry-run QA

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && python benchmarks/attribution_vdn.py --dry-run`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `19cbd258b4dadd424b4c51f7d3fa1079881ab773`
- **Wall time:** 0.042 s
- **Artifacts:** docs/attribution_vdn_runlog.md
- **Outcome:** Task 1 dry-run QA: exit 0

<details><summary>Verbatim output</summary>

```text
CUDA_VISIBLE_DEVICES=0 python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=/mnt/raid0nvme0/leyang/Omni-Infinity/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-off.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false # cwd=/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-on.ranges.json python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=/mnt/raid0nvme0/leyang/Omni-Infinity/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-on.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false # cwd=/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=null render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-prof.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false # cwd=/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=/mnt/raid0nvme0/leyang/Omni-Infinity/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false # cwd=/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V1-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe_tuned.yaml checkpoint=/mnt/raid0nvme0/leyang/Omni-Infinity/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V1-prof.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false # cwd=/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V2-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe_tuned_fp8.yaml checkpoint=/mnt/raid0nvme0/leyang/Omni-Infinity/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V2-prof.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false # cwd=/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-120-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=null render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-120-prof.mp4 render.num_frames=120 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false # cwd=/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-345-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=null render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-345-prof.mp4 render.num_frames=345 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false # cwd=/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-120-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=/mnt/raid0nvme0/leyang/Omni-Infinity/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-120-prof.mp4 render.num_frames=120 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false # cwd=/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-311-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=/mnt/raid0nvme0/leyang/Omni-Infinity/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-311-prof.mp4 render.num_frames=311 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false # cwd=/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3
```
</details>

### 2026-09-23T09:27:45.931886+00:00 — Task 1 ruff QA pre-GPU

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && ruff check benchmarks/`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `19cbd258b4dadd424b4c51f7d3fa1079881ab773`
- **Wall time:** 0.052 s
- **Artifacts:** docs/attribution_vdn_runlog.md
- **Outcome:** Task 1 ruff QA pre-GPU: exit 1

<details><summary>Verbatim output</summary>

```text
benchmarks/attribution_vdn.py:6:1: I001 [*] Import block is un-sorted or un-formatted
   |
 4 |   """Run the VDN-H3 speedup-attribution experiment matrix."""
 5 |   
 6 | / from __future__ import annotations
 7 | | 
 8 | | import argparse
 9 | | import json
10 | | import os
11 | | from pathlib import Path
12 | | import shlex
13 | | import statistics
14 | | import subprocess
15 | | from dataclasses import dataclass
16 | | from typing import Any
17 | | 
18 | | 
19 | | REPO = Path(__file__).resolve().parents[1]
   | |_^ I001
20 |   DEFAULT_RESULTS = REPO / "results" / "vdn" / "attribution"
21 |   DEFAULT_VDN = REPO / "third_party" / "vdn-minimax-h3"
   |
   = help: Organize imports

benchmarks/attribution_vdn.py:16:20: F401 [*] `typing.Any` imported but unused
   |
14 | import subprocess
15 | from dataclasses import dataclass
16 | from typing import Any
   |                    ^^^ F401
   |
   = help: Remove unused import: `typing.Any`

benchmarks/attribution_vdn.py:36:81: E501 Line too long (84 > 80)
   |
35 | RUNS = (
36 |     Run("V0-prof-overheadcheck-off", "8nfe.yaml", "stage-dmd-step-250", 222, False),
   |                                                                                 ^^^^ E501
37 |     Run("V0-prof-overheadcheck-on", "8nfe.yaml", "stage-dmd-step-250", 222),
38 |     Run("D-prof", "8nfe.yaml", None, 222),
   |

benchmarks/attribution_vdn.py:93:81: E501 Line too long (83 > 80)
   |
91 |     if run.shim:
92 |         old_path = env.get("PYTHONPATH")
93 |         env["PYTHONPATH"] = str(SHIM) + (os.pathsep + old_path if old_path else "")
   |                                                                                 ^^^ E501
94 |         env["VDN_PROF_OUT"] = str(ranges)
95 |     else:
   |

benchmarks/attribution_vdn.py:137:81: E501 Line too long (97 > 80)
    |
135 |             env = f"CUDA_VISIBLE_DEVICES={args.gpu}"
136 |             if run.shim:
137 |                 env += f" PYTHONPATH={SHIM} VDN_PROF_OUT={results / (run.name + '.ranges.json')}"
    |                                                                                 ^^^^^^^^^^^^^^^^^ E501
138 |             print(
139 |                 f"{env} {shlex.join(build_command(run, results, ckpts))} "
    |

benchmarks/vdn_prof_shim/sitecustomize.py:11:1: I001 [*] Import block is un-sorted or un-formatted
   |
 9 |   """
10 |   
11 | / from __future__ import annotations
12 | | 
13 | | import atexit
14 | | import builtins
15 | | from contextlib import contextmanager
16 | | import json
17 | | import os
18 | | from pathlib import Path
19 | | import sys
20 | | from typing import Any, Callable, Iterator
21 | | 
22 | | 
23 | | UPSTREAM_PATCH_TARGETS = (
   | |_^ I001
24 |       "src.models.hybrid_attention.HybridAttention._window_softmax",
25 |       "src.models.softmax_attention.dense_processor.FlexFA4Processor.__call__",
   |
   = help: Organize imports

Found 6 errors.
[*] 3 fixable with the `--fix` option.
```
</details>

### 2026-09-23T09:28:14.180198+00:00 — Task 1 GPU discipline RED

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && python -m pytest tests/test_attribution_shim.py::test_driver_rejects_busy_gpu -v`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `19cbd258b4dadd424b4c51f7d3fa1079881ab773`
- **Wall time:** 0.175 s
- **Artifacts:** docs/attribution_vdn_runlog.md
- **Outcome:** Task 1 GPU discipline RED: exit 1

<details><summary>Verbatim output</summary>

```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /mnt/raid0nvme0/leyang/envs/vdn/bin/python
cachedir: .pytest_cache
rootdir: /mnt/raid0nvme0/leyang/Omni-Infinity
configfile: pyproject.toml
plugins: timeout-2.4.0, anyio-4.15.1
collecting ... collected 1 item

tests/test_attribution_shim.py::test_driver_rejects_busy_gpu FAILED      [100%]

=================================== FAILURES ===================================
_________________________ test_driver_rejects_busy_gpu _________________________

    def test_driver_rejects_busy_gpu():
        driver_path = Path(__file__).parents[1] / "benchmarks" / "attribution_vdn.py"
        spec = importlib.util.spec_from_file_location("attribution_vdn", driver_path)
        assert spec is not None and spec.loader is not None
        driver = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = driver
        spec.loader.exec_module(driver)
    
>       assert driver._gpu_is_idle("3, 0")
               ^^^^^^^^^^^^^^^^^^^
E       AttributeError: module 'attribution_vdn' has no attribute '_gpu_is_idle'

tests/test_attribution_shim.py:76: AttributeError
=========================== short test summary info ============================
FAILED tests/test_attribution_shim.py::test_driver_rejects_busy_gpu - Attribu...
============================== 1 failed in 0.02s ===============================
```
</details>

### 2026-09-23T09:29:03.807758+00:00 — Task 1 ruff QA retry

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && ruff check benchmarks/`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `19cbd258b4dadd424b4c51f7d3fa1079881ab773`
- **Wall time:** 0.024 s
- **Artifacts:** docs/attribution_vdn_runlog.md
- **Outcome:** Task 1 ruff QA retry: exit 1

<details><summary>Verbatim output</summary>

```text
benchmarks/attribution_vdn.py:6:1: I001 [*] Import block is un-sorted or un-formatted
   |
 4 |   """Run the VDN-H3 speedup-attribution experiment matrix."""
 5 |   
 6 | / from __future__ import annotations
 7 | | 
 8 | | import argparse
 9 | | import json
10 | | import os
11 | | import shlex
12 | | import statistics
13 | | import subprocess
14 | | import time
15 | | from dataclasses import dataclass
16 | | from datetime import datetime, timezone
17 | | from pathlib import Path
18 | | 
19 | | 
20 | | REPO = Path(__file__).resolve().parents[1]
   | |_^ I001
21 |   DEFAULT_RESULTS = REPO / "results" / "vdn" / "attribution"
22 |   DEFAULT_VDN = REPO / "third_party" / "vdn-minimax-h3"
   |
   = help: Organize imports

benchmarks/attribution_vdn.py:86:81: E501 Line too long (81 > 80)
   |
85 | def _gpu_is_idle(sample: str) -> bool:
86 |     memory_mib, utilization = (int(value.strip()) for value in sample.split(","))
   |                                                                                 ^ E501
87 |     return memory_mib < 512 and utilization < 10
   |

benchmarks/attribution_vdn.py:104:81: E501 Line too long (82 > 80)
    |
102 |     ).stdout.strip()
103 |     if not _gpu_is_idle(query):
104 |         raise RuntimeError(f"refusing busy GPU {gpu}: memory/utilization={query}")
    |                                                                                 ^^ E501
105 |     model = subprocess.run(
106 |         [
    |

benchmarks/attribution_vdn.py:149:81: E501 Line too long (82 > 80)
    |
147 |     with path.open("a", encoding="utf-8") as stream:
148 |         stream.write(
149 |             f"\n### {datetime.now(timezone.utc).isoformat()} — GPU {run.name}\n\n"
    |                                                                                 ^^ E501
150 |             f"- **Command:** `{command_line}`\n"
151 |             f"- **CWD:** `{cwd}`\n"
    |

benchmarks/vdn_prof_shim/sitecustomize.py:11:1: I001 [*] Import block is un-sorted or un-formatted
   |
 9 |   """
10 |   
11 | / from __future__ import annotations
12 | | 
13 | | import atexit
14 | | import builtins
15 | | import json
16 | | import os
17 | | import sys
18 | | from contextlib import contextmanager
19 | | from pathlib import Path
20 | | from typing import Any, Callable, Iterator
21 | | 
22 | | 
23 | | UPSTREAM_PATCH_TARGETS = (
   | |_^ I001
24 |       "src.models.hybrid_attention.HybridAttention._window_softmax",
25 |       "src.models.softmax_attention.dense_processor.FlexFA4Processor.__call__",
   |
   = help: Organize imports

Found 5 errors.
[*] 2 fixable with the `--fix` option.
```
</details>

### 2026-09-23T09:29:03.741967+00:00 — Task 1 complete unit QA

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && python -m pytest tests/test_attribution_shim.py -v`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `19cbd258b4dadd424b4c51f7d3fa1079881ab773`
- **Wall time:** 0.162 s
- **Artifacts:** docs/attribution_vdn_runlog.md
- **Outcome:** Task 1 complete unit QA: exit 0

<details><summary>Verbatim output</summary>

```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /mnt/raid0nvme0/leyang/envs/vdn/bin/python
cachedir: .pytest_cache
rootdir: /mnt/raid0nvme0/leyang/Omni-Infinity
configfile: pyproject.toml
plugins: timeout-2.4.0, anyio-4.15.1
collecting ... collected 3 items

tests/test_attribution_shim.py::test_shim_imports_without_upstream_package PASSED [ 33%]
tests/test_attribution_shim.py::test_range_accounting_sums_mock_cuda_events PASSED [ 66%]
tests/test_attribution_shim.py::test_driver_rejects_busy_gpu PASSED      [100%]

============================== 3 passed in 0.01s ===============================
```
</details>

### 2026-09-23T09:29:52.634115+00:00 — Task 1 ruff root-cause diagnostic

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && ruff check benchmarks/attribution_vdn.py benchmarks/vdn_prof_shim/sitecustomize.py --fix --diff`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `19cbd258b4dadd424b4c51f7d3fa1079881ab773`
- **Wall time:** 0.025 s
- **Artifacts:** docs/attribution_vdn_runlog.md
- **Outcome:** Task 1 ruff root-cause diagnostic: exit 1

<details><summary>Verbatim output</summary>

```text
--- benchmarks/vdn_prof_shim/sitecustomize.py
+++ benchmarks/vdn_prof_shim/sitecustomize.py
@@ -19,7 +19,6 @@
 from pathlib import Path
 from typing import Any, Callable, Iterator
 
-
 UPSTREAM_PATCH_TARGETS = (
     "src.models.hybrid_attention.HybridAttention._window_softmax",
     "src.models.softmax_attention.dense_processor.FlexFA4Processor.__call__",

--- benchmarks/attribution_vdn.py
+++ benchmarks/attribution_vdn.py
@@ -16,7 +16,6 @@
 from datetime import datetime, timezone
 from pathlib import Path
 
-
 REPO = Path(__file__).resolve().parents[1]
 DEFAULT_RESULTS = REPO / "results" / "vdn" / "attribution"
 DEFAULT_VDN = REPO / "third_party" / "vdn-minimax-h3"

Would fix 2 errors.
```
</details>

### 2026-09-23T09:30:05.925501+00:00 — Task 1 resumed ruff QA

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && ruff check benchmarks/`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `19cbd258b4dadd424b4c51f7d3fa1079881ab773`
- **Wall time:** 0.027 s
- **Artifacts:** docs/attribution_vdn_runlog.md
- **Outcome:** Task 1 resumed ruff QA: exit 0

<details><summary>Verbatim output</summary>

```text
All checks passed!
```
</details>

### 2026-09-23T09:31:05.520927+00:00 — GPU V0-prof-overheadcheck-off

- **Command:** `CUDA_VISIBLE_DEVICES=0 PYTHONPATH='' HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=/mnt/raid0nvme0/leyang/Omni-Infinity/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-off.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3`
- **GPU index + model:** 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
- **Git SHA:** `19cbd258b4dadd424b4c51f7d3fa1079881ab773`
- **Wall time:** 30.001 s
- **Artifacts:** /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-off.mp4.inference.json, /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-off.log, /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-off.mp4
- **Outcome:** V0-prof-overheadcheck-off: exit 1

### 2026-09-23T09:30:35.311160+00:00 — Task 1 overhead gate GPU runs

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface VDN_RUNLOG=/mnt/raid0nvme0/leyang/Omni-Infinity/docs/attribution_vdn_runlog.md && python benchmarks/attribution_vdn.py --only V0-prof-overheadcheck --gpu 0`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** 0; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
- **Git SHA:** `19cbd258b4dadd424b4c51f7d3fa1079881ab773`
- **Wall time:** 30.168 s
- **Artifacts:** results/vdn/attribution/V0-prof-overheadcheck-{off,on}.{log,mp4,inference.json}, results/vdn/attribution/V0-prof-overheadcheck-on.ranges.json, docs/attribution_vdn_runlog.md
- **Outcome:** Task 1 overhead gate GPU runs: exit 1

<details><summary>Verbatim output</summary>

```text
GPU idle check: 0, NVIDIA RTX PRO 6000 Blackwell Server Edition; memory/utilization=3, 0
run V0-prof-overheadcheck-off: CUDA_VISIBLE_DEVICES=0 PYTHONPATH='' HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=/mnt/raid0nvme0/leyang/Omni-Infinity/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-off.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false
poll V0-prof-overheadcheck-off: pid=2353501 running
Traceback (most recent call last):
  File "/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/attribution_vdn.py", line 279, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/attribution_vdn.py", line 272, in main
    _execute(run, args)
  File "/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/attribution_vdn.py", line 225, in _execute
    raise subprocess.CalledProcessError(process.returncode, command)
subprocess.CalledProcessError: Command '['python', 'src/inference/infer.py', '--config', 'configs/inference/8nfe.yaml', 'checkpoint=/mnt/raid0nvme0/leyang/Omni-Infinity/ckpts/vdn/stage-dmd-step-250', 'render.prompt_file=prompts/example_0.pt', 'render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-off.mp4', 'render.num_frames=222', 'render.seed=0', 'render.warmup_steps=2', 'render.record=true', 'render.save_latents=false']' returned non-zero exit status 1.
```
</details>

### 2026-09-23T09:31:25.899120+00:00 — Task 1 checkpoint path RED

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && python -m pytest tests/test_attribution_shim.py::test_driver_rejects_busy_gpu -v`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `19cbd258b4dadd424b4c51f7d3fa1079881ab773`
- **Wall time:** 0.170 s
- **Artifacts:** docs/attribution_vdn_runlog.md
- **Outcome:** Task 1 checkpoint path RED: exit 1

<details><summary>Verbatim output</summary>

```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /mnt/raid0nvme0/leyang/envs/vdn/bin/python
cachedir: .pytest_cache
rootdir: /mnt/raid0nvme0/leyang/Omni-Infinity
configfile: pyproject.toml
plugins: timeout-2.4.0, anyio-4.15.1
collecting ... collected 1 item

tests/test_attribution_shim.py::test_driver_rejects_busy_gpu FAILED      [100%]

=================================== FAILURES ===================================
_________________________ test_driver_rejects_busy_gpu _________________________

    def test_driver_rejects_busy_gpu():
        driver_path = Path(__file__).parents[1] / "benchmarks" / "attribution_vdn.py"
        spec = importlib.util.spec_from_file_location("attribution_vdn", driver_path)
        assert spec is not None and spec.loader is not None
        driver = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = driver
        spec.loader.exec_module(driver)
    
        assert driver._gpu_is_idle("3, 0")
        assert not driver._gpu_is_idle("90000, 99")
>       assert driver.DEFAULT_CKPTS == driver.DEFAULT_VDN / "ckpts"
E       AssertionError: assert PosixPath('/mnt/raid0nvme0/leyang/Omni-Infinity/ckpts/vdn') == (PosixPath('/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3') / 'ckpts')
E        +  where PosixPath('/mnt/raid0nvme0/leyang/Omni-Infinity/ckpts/vdn') = <module 'attribution_vdn' from '/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/attribution_vdn.py'>.DEFAULT_CKPTS
E        +  and   PosixPath('/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3') = <module 'attribution_vdn' from '/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/attribution_vdn.py'>.DEFAULT_VDN

tests/test_attribution_shim.py:78: AssertionError
=========================== short test summary info ============================
FAILED tests/test_attribution_shim.py::test_driver_rejects_busy_gpu - Asserti...
============================== 1 failed in 0.02s ===============================
```
</details>

### 2026-09-23T09:31:33.860096+00:00 — Task 1 checkpoint path GREEN

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && python -m pytest tests/test_attribution_shim.py -v`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `19cbd258b4dadd424b4c51f7d3fa1079881ab773`
- **Wall time:** 0.167 s
- **Artifacts:** docs/attribution_vdn_runlog.md
- **Outcome:** Task 1 checkpoint path GREEN: exit 0

<details><summary>Verbatim output</summary>

```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /mnt/raid0nvme0/leyang/envs/vdn/bin/python
cachedir: .pytest_cache
rootdir: /mnt/raid0nvme0/leyang/Omni-Infinity
configfile: pyproject.toml
plugins: timeout-2.4.0, anyio-4.15.1
collecting ... collected 3 items

tests/test_attribution_shim.py::test_shim_imports_without_upstream_package PASSED [ 33%]
tests/test_attribution_shim.py::test_range_accounting_sums_mock_cuda_events PASSED [ 66%]
tests/test_attribution_shim.py::test_driver_rejects_busy_gpu PASSED      [100%]

============================== 3 passed in 0.01s ===============================
```
</details>

### 2026-09-23T09:37:41.404286+00:00 — GPU V0-prof-overheadcheck-off

- **Command:** `CUDA_VISIBLE_DEVICES=0 PYTHONPATH='' HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=/mnt/raid0nvme0/leyang/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-off.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3`
- **GPU index + model:** 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
- **Git SHA:** `19cbd258b4dadd424b4c51f7d3fa1079881ab773`
- **Wall time:** 360.003 s
- **Artifacts:** /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-off.mp4.inference.json, /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-off.log, /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-off.mp4
- **Outcome:** V0-prof-overheadcheck-off: exit 0

### 2026-09-23T09:43:41.527180+00:00 — GPU V0-prof-overheadcheck-on

- **Command:** `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-on.ranges.json python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=/mnt/raid0nvme0/leyang/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-on.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3`
- **GPU index + model:** 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
- **Git SHA:** `19cbd258b4dadd424b4c51f7d3fa1079881ab773`
- **Wall time:** 360.003 s
- **Artifacts:** /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-on.mp4.inference.json, /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-on.log, /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-on.mp4, /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-on.ranges.json
- **Outcome:** V0-prof-overheadcheck-on: exit 0

### 2026-09-23T09:31:41.184843+00:00 — Task 1 overhead gate GPU retry

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface VDN_RUNLOG=/mnt/raid0nvme0/leyang/Omni-Infinity/docs/attribution_vdn_runlog.md && python benchmarks/attribution_vdn.py --only V0-prof-overheadcheck --gpu 0 --force`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** 0; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
- **Git SHA:** `19cbd258b4dadd424b4c51f7d3fa1079881ab773`
- **Wall time:** 720.298 s
- **Artifacts:** results/vdn/attribution/V0-prof-overheadcheck-{off,on}.{log,mp4,inference.json}, results/vdn/attribution/V0-prof-overheadcheck-on.ranges.json, docs/attribution_vdn_runlog.md
- **Outcome:** Task 1 overhead gate GPU retry: exit 0

<details><summary>Verbatim output</summary>

```text
GPU idle check: 0, NVIDIA RTX PRO 6000 Blackwell Server Edition; memory/utilization=3, 0
run V0-prof-overheadcheck-off: CUDA_VISIBLE_DEVICES=0 PYTHONPATH='' HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=/mnt/raid0nvme0/leyang/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-off.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false
poll V0-prof-overheadcheck-off: pid=2354589 running
poll V0-prof-overheadcheck-off: pid=2354589 running
poll V0-prof-overheadcheck-off: pid=2354589 running
poll V0-prof-overheadcheck-off: pid=2354589 running
poll V0-prof-overheadcheck-off: pid=2354589 running
poll V0-prof-overheadcheck-off: pid=2354589 running
poll V0-prof-overheadcheck-off: pid=2354589 running
poll V0-prof-overheadcheck-off: pid=2354589 running
poll V0-prof-overheadcheck-off: pid=2354589 running
poll V0-prof-overheadcheck-off: pid=2354589 running
poll V0-prof-overheadcheck-off: pid=2354589 running
poll V0-prof-overheadcheck-off: pid=2354589 running
completed V0-prof-overheadcheck-off in 360.0s; log=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-off.log
GPU idle check: 0, NVIDIA RTX PRO 6000 Blackwell Server Edition; memory/utilization=3, 0
run V0-prof-overheadcheck-on: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-on.ranges.json python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=/mnt/raid0nvme0/leyang/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-on.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false
poll V0-prof-overheadcheck-on: pid=2359192 running
poll V0-prof-overheadcheck-on: pid=2359192 running
poll V0-prof-overheadcheck-on: pid=2359192 running
poll V0-prof-overheadcheck-on: pid=2359192 running
poll V0-prof-overheadcheck-on: pid=2359192 running
poll V0-prof-overheadcheck-on: pid=2359192 running
poll V0-prof-overheadcheck-on: pid=2359192 running
poll V0-prof-overheadcheck-on: pid=2359192 running
poll V0-prof-overheadcheck-on: pid=2359192 running
poll V0-prof-overheadcheck-on: pid=2359192 running
poll V0-prof-overheadcheck-on: pid=2359192 running
poll V0-prof-overheadcheck-on: pid=2359192 running
completed V0-prof-overheadcheck-on in 360.0s; log=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-on.log
profiler overhead: off=23.1959s/NFE on=23.1903s/NFE delta=-0.02% (gate <3%)
```
</details>

### 2026-09-23T09:44:04.870425+00:00 — Task 1 exclusive range RED

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && python -m pytest tests/test_attribution_shim.py::test_linear_branch_accounting_subtracts_only_its_own_gate -v`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `19cbd258b4dadd424b4c51f7d3fa1079881ab773`
- **Wall time:** 0.166 s
- **Artifacts:** docs/attribution_vdn_runlog.md
- **Outcome:** Task 1 exclusive range RED: exit 1

<details><summary>Verbatim output</summary>

```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /mnt/raid0nvme0/leyang/envs/vdn/bin/python
cachedir: .pytest_cache
rootdir: /mnt/raid0nvme0/leyang/Omni-Infinity
configfile: pyproject.toml
plugins: timeout-2.4.0, anyio-4.15.1
collecting ... collected 1 item

tests/test_attribution_shim.py::test_linear_branch_accounting_subtracts_only_its_own_gate FAILED [100%]

=================================== FAILURES ===================================
__________ test_linear_branch_accounting_subtracts_only_its_own_gate ___________

    def test_linear_branch_accounting_subtracts_only_its_own_gate():
        module = _load_shim()
        timestamps = iter((0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0))
    
        class FakeEvent:
            def record(self):
                self.timestamp = next(timestamps)
    
            def elapsed_time(self, other):
                return other.timestamp - self.timestamp
    
        accounting = module.RangeAccounting(FakeEvent, lambda: None)
        with accounting.measure("step_total"):
            with accounting.measure("linear_branch"):
                with accounting.measure("gates"):
                    pass
            with accounting.measure("linear_branch"):
                with accounting.measure("gates"):
                    pass
    
>       assert accounting.steps[0]["linear_branch"] == {
            "total_ms": 6.0,
            "calls": 2,
        }
E       AssertionError: assert {'total_ms': 4.0, 'calls': 2} == {'total_ms': 6.0, 'calls': 2}
E         
E         Omitting 1 identical items, use -vv to show
E         Differing items:
E         {'total_ms': 4.0} != {'total_ms': 6.0}
E         
E         Full diff:
E           {...
E         
E         ...Full output truncated (6 lines hidden), use '-vv' to show

tests/test_attribution_shim.py:88: AssertionError
=========================== short test summary info ============================
FAILED tests/test_attribution_shim.py::test_linear_branch_accounting_subtracts_only_its_own_gate
============================== 1 failed in 0.02s ===============================
```
</details>

### 2026-09-23T09:44:19.169023+00:00 — Task 1 final pytest QA

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && python -m pytest tests/test_attribution_shim.py -v`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `19cbd258b4dadd424b4c51f7d3fa1079881ab773`
- **Wall time:** 0.158 s
- **Artifacts:** docs/attribution_vdn_runlog.md
- **Outcome:** Task 1 final pytest QA: exit 0

<details><summary>Verbatim output</summary>

```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /mnt/raid0nvme0/leyang/envs/vdn/bin/python
cachedir: .pytest_cache
rootdir: /mnt/raid0nvme0/leyang/Omni-Infinity
configfile: pyproject.toml
plugins: timeout-2.4.0, anyio-4.15.1
collecting ... collected 4 items

tests/test_attribution_shim.py::test_shim_imports_without_upstream_package PASSED [ 25%]
tests/test_attribution_shim.py::test_range_accounting_sums_mock_cuda_events PASSED [ 50%]
tests/test_attribution_shim.py::test_linear_branch_accounting_subtracts_only_its_own_gate PASSED [ 75%]
tests/test_attribution_shim.py::test_driver_rejects_busy_gpu PASSED      [100%]

============================== 4 passed in 0.01s ===============================
```
</details>

### 2026-09-23T09:44:24.356306+00:00 — Task 1 final dry-run QA

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && python benchmarks/attribution_vdn.py --dry-run`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `19cbd258b4dadd424b4c51f7d3fa1079881ab773`
- **Wall time:** 0.040 s
- **Artifacts:** docs/attribution_vdn_runlog.md
- **Outcome:** Task 1 final dry-run QA: exit 0

<details><summary>Verbatim output</summary>

```text
CUDA_VISIBLE_DEVICES=0 python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=/mnt/raid0nvme0/leyang/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-off.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false # cwd=/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-on.ranges.json python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=/mnt/raid0nvme0/leyang/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof-overheadcheck-on.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false # cwd=/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=null render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-prof.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false # cwd=/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=/mnt/raid0nvme0/leyang/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false # cwd=/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V1-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe_tuned.yaml checkpoint=/mnt/raid0nvme0/leyang/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V1-prof.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false # cwd=/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V2-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe_tuned_fp8.yaml checkpoint=/mnt/raid0nvme0/leyang/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V2-prof.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false # cwd=/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-120-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=null render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-120-prof.mp4 render.num_frames=120 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false # cwd=/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-345-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=null render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-345-prof.mp4 render.num_frames=345 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false # cwd=/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-120-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=/mnt/raid0nvme0/leyang/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-120-prof.mp4 render.num_frames=120 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false # cwd=/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-311-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=/mnt/raid0nvme0/leyang/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-311-prof.mp4 render.num_frames=311 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false # cwd=/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3
```
</details>

### 2026-09-23T09:44:30.354555+00:00 — Task 1 final ruff QA

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && ruff check benchmarks/`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `19cbd258b4dadd424b4c51f7d3fa1079881ab773`
- **Wall time:** 0.024 s
- **Artifacts:** docs/attribution_vdn_runlog.md
- **Outcome:** Task 1 final ruff QA: exit 0

<details><summary>Verbatim output</summary>

```text
All checks passed!
```
</details>

### 2026-09-23T09:45:11.724415+00:00 — Task 1 pre-commit review

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && GIT_MASTER=1 git status --short && GIT_MASTER=1 git diff -- benchmarks/attribution_vdn.py benchmarks/vdn_prof_shim/sitecustomize.py tests/test_attribution_shim.py docs/attribution_vdn_runlog.md && GIT_MASTER=1 git log --oneline -10`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `19cbd258b4dadd424b4c51f7d3fa1079881ab773`
- **Wall time:** 0.020 s
- **Artifacts:** docs/attribution_vdn_runlog.md
- **Outcome:** Task 1 pre-commit review: exit 0

<details><summary>Verbatim output</summary>

```text
 ? third_party/vdn-minimax-h3
?? .sisyphus/
?? benchmarks/attribution_vdn.py
?? benchmarks/vdn_prof_shim/
?? docs/attribution_vdn_runlog.md
?? generated_latents.pt
?? tests/test_attribution_shim.py
19cbd25 feat(docker): reproducible VDN environment image (#10)
f58fc27 docs: VDN-H3 16-run ablation study results (#10)
43a3f97 fix(bench): drop frame-mismatched goldens gate from omni grid rows (#10)
b0efffb docs: model-arch/optimization category section in README (#10)
8914d6c docs: VDN repro results (sm120) + fix omni ablation rows for cross-repo guard (#10)
fd7fda5 test: record vdn-hybrid golden latents (#10)
db01ba2 feat(bench): VDN ablation harness - 16-run arch/optimization grid (#10)
e5b450b fix(test): vdn parity gate uses streamed-encoder recipe for 96GB cards (#10)
a5d14e0 feat(smoke): vdn_smoke CLI + vdn-hybrid parity gate (#10)
9e0a922 feat: VdnRunner - vdn-hybrid arch over the published diffusers component (#10)
```
</details>

### 2026-09-23T09:45:19.299227+00:00 — Task 1 commit

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && GIT_MASTER=1 git add benchmarks/attribution_vdn.py benchmarks/vdn_prof_shim/sitecustomize.py tests/test_attribution_shim.py docs/attribution_vdn_runlog.md && GIT_MASTER=1 git diff --staged --stat && GIT_MASTER=1 git commit -m "feat(bench): profiler shim + attribution driver (#10)" -m "Ultraworked with [Sisyphus](https://github.com/code-yeongyu/oh-my-openagent)" -m "Co-authored-by: Sisyphus <clio-agent@sisyphuslabs.ai>" && GIT_MASTER=1 git log -1 --oneline`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `19cbd258b4dadd424b4c51f7d3fa1079881ab773`
- **Wall time:** 0.021 s
- **Artifacts:** git commit, docs/attribution_vdn_runlog.md
- **Outcome:** Task 1 commit: exit 0

<details><summary>Verbatim output</summary>

```text
 benchmarks/attribution_vdn.py             |  279 +++++++
 benchmarks/vdn_prof_shim/sitecustomize.py |  200 +++++
 docs/attribution_vdn_runlog.md            | 1131 +++++++++++++++++++++++++++++
 tests/test_attribution_shim.py            |  111 +++
 4 files changed, 1721 insertions(+)
[feat/vdn-minimax-h3 52c3365] feat(bench): profiler shim + attribution driver (#10)
 4 files changed, 1721 insertions(+)
 create mode 100644 benchmarks/attribution_vdn.py
 create mode 100644 benchmarks/vdn_prof_shim/sitecustomize.py
 create mode 100644 docs/attribution_vdn_runlog.md
 create mode 100644 tests/test_attribution_shim.py
52c3365 feat(bench): profiler shim + attribution driver (#10)
```
</details>

### 2026-09-23T09:52:39.283903+00:00 — GPU D-prof

- **Command:** `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=null render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-prof.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3`
- **GPU index + model:** 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
- **Git SHA:** `52c3365914720a4eaacb7c5099f17d9437464966`
- **Wall time:** 420.003 s
- **Artifacts:** /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-prof.mp4.inference.json, /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-prof.log, /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-prof.mp4, /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-prof.ranges.json
- **Outcome:** D-prof: exit 0

### 2026-09-23T09:45:39.072213+00:00 — Task 2 D-prof GPU run

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface VDN_RUNLOG=/mnt/raid0nvme0/leyang/Omni-Infinity/docs/attribution_vdn_runlog.md && python benchmarks/attribution_vdn.py --only D-prof --gpu 0`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** 0; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
- **Git SHA:** `52c3365914720a4eaacb7c5099f17d9437464966`
- **Wall time:** 420.167 s
- **Artifacts:** results/vdn/attribution/D-prof.{log,mp4,inference.json,ranges.json}, docs/attribution_vdn_runlog.md
- **Outcome:** Task 2 D-prof GPU run: exit 0

<details><summary>Verbatim output</summary>

```text
GPU idle check: 0, NVIDIA RTX PRO 6000 Blackwell Server Edition; memory/utilization=3, 0
run D-prof: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=null render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-prof.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false
poll D-prof: pid=2366104 running
poll D-prof: pid=2366104 running
poll D-prof: pid=2366104 running
poll D-prof: pid=2366104 running
poll D-prof: pid=2366104 running
poll D-prof: pid=2366104 running
poll D-prof: pid=2366104 running
poll D-prof: pid=2366104 running
poll D-prof: pid=2366104 running
poll D-prof: pid=2366104 running
poll D-prof: pid=2366104 running
poll D-prof: pid=2366104 running
poll D-prof: pid=2366104 running
poll D-prof: pid=2366104 running
completed D-prof in 420.0s; log=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-prof.log
```
</details>

### 2026-09-23T09:58:45.012264+00:00 — GPU V0-prof

- **Command:** `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=/mnt/raid0nvme0/leyang/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3`
- **GPU index + model:** 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
- **Git SHA:** `52c3365914720a4eaacb7c5099f17d9437464966`
- **Wall time:** 360.003 s
- **Artifacts:** /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof.mp4.inference.json, /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof.log, /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof.mp4, /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof.ranges.json
- **Outcome:** V0-prof: exit 0

### 2026-09-23T09:52:44.797638+00:00 — Task 2 V0-prof GPU run

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface VDN_RUNLOG=/mnt/raid0nvme0/leyang/Omni-Infinity/docs/attribution_vdn_runlog.md && python benchmarks/attribution_vdn.py --only V0-prof --gpu 0`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** 0; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
- **Git SHA:** `52c3365914720a4eaacb7c5099f17d9437464966`
- **Wall time:** 360.172 s
- **Artifacts:** results/vdn/attribution/V0-prof.{log,mp4,inference.json,ranges.json}, docs/attribution_vdn_runlog.md
- **Outcome:** Task 2 V0-prof GPU run: exit 0

<details><summary>Verbatim output</summary>

```text
GPU idle check: 0, NVIDIA RTX PRO 6000 Blackwell Server Edition; memory/utilization=3, 0
run V0-prof: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=/mnt/raid0nvme0/leyang/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false
poll V0-prof: pid=2371240 running
poll V0-prof: pid=2371240 running
poll V0-prof: pid=2371240 running
poll V0-prof: pid=2371240 running
poll V0-prof: pid=2371240 running
poll V0-prof: pid=2371240 running
poll V0-prof: pid=2371240 running
poll V0-prof: pid=2371240 running
poll V0-prof: pid=2371240 running
poll V0-prof: pid=2371240 running
poll V0-prof: pid=2371240 running
poll V0-prof: pid=2371240 running
completed V0-prof in 360.0s; log=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V0-prof.log
```
</details>

### 2026-09-23T10:04:19.941521+00:00 — GPU V1-prof

- **Command:** `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V1-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe_tuned.yaml checkpoint=/mnt/raid0nvme0/leyang/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V1-prof.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3`
- **GPU index + model:** 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
- **Git SHA:** `52c3365914720a4eaacb7c5099f17d9437464966`
- **Wall time:** 330.003 s
- **Artifacts:** /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V1-prof.mp4.inference.json, /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V1-prof.log, /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V1-prof.mp4, /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V1-prof.ranges.json
- **Outcome:** V1-prof: exit 0

### 2026-09-23T09:58:49.716519+00:00 — Task 2 V1-prof GPU run

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface VDN_RUNLOG=/mnt/raid0nvme0/leyang/Omni-Infinity/docs/attribution_vdn_runlog.md && python benchmarks/attribution_vdn.py --only V1-prof --gpu 0`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** 0; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
- **Git SHA:** `52c3365914720a4eaacb7c5099f17d9437464966`
- **Wall time:** 330.180 s
- **Artifacts:** results/vdn/attribution/V1-prof.{log,mp4,inference.json,ranges.json}, docs/attribution_vdn_runlog.md
- **Outcome:** Task 2 V1-prof GPU run: exit 0

<details><summary>Verbatim output</summary>

```text
GPU idle check: 0, NVIDIA RTX PRO 6000 Blackwell Server Edition; memory/utilization=3, 0
run V1-prof: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V1-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe_tuned.yaml checkpoint=/mnt/raid0nvme0/leyang/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V1-prof.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false
poll V1-prof: pid=2376095 running
poll V1-prof: pid=2376095 running
poll V1-prof: pid=2376095 running
poll V1-prof: pid=2376095 running
poll V1-prof: pid=2376095 running
poll V1-prof: pid=2376095 running
poll V1-prof: pid=2376095 running
poll V1-prof: pid=2376095 running
poll V1-prof: pid=2376095 running
poll V1-prof: pid=2376095 running
poll V1-prof: pid=2376095 running
completed V1-prof in 330.0s; log=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V1-prof.log
```
</details>

### 2026-09-23T10:08:57.003290+00:00 — GPU V2-prof

- **Command:** `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V2-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe_tuned_fp8.yaml checkpoint=/mnt/raid0nvme0/leyang/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V2-prof.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3`
- **GPU index + model:** 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
- **Git SHA:** `52c3365914720a4eaacb7c5099f17d9437464966`
- **Wall time:** 270.002 s
- **Artifacts:** /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V2-prof.mp4.inference.json, /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V2-prof.log, /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V2-prof.mp4, /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V2-prof.ranges.json
- **Outcome:** V2-prof: exit 0

### 2026-09-23T10:04:26.783533+00:00 — Task 2 V2-prof GPU run

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface VDN_RUNLOG=/mnt/raid0nvme0/leyang/Omni-Infinity/docs/attribution_vdn_runlog.md && python benchmarks/attribution_vdn.py --only V2-prof --gpu 0`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** 0; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
- **Git SHA:** `52c3365914720a4eaacb7c5099f17d9437464966`
- **Wall time:** 270.176 s
- **Artifacts:** results/vdn/attribution/V2-prof.{log,mp4,inference.json,ranges.json}, docs/attribution_vdn_runlog.md
- **Outcome:** Task 2 V2-prof GPU run: exit 0

<details><summary>Verbatim output</summary>

```text
GPU idle check: 0, NVIDIA RTX PRO 6000 Blackwell Server Edition; memory/utilization=3, 0
run V2-prof: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V2-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe_tuned_fp8.yaml checkpoint=/mnt/raid0nvme0/leyang/ckpts/vdn/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V2-prof.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false
poll V2-prof: pid=2385914 running
poll V2-prof: pid=2385914 running
poll V2-prof: pid=2385914 running
poll V2-prof: pid=2385914 running
poll V2-prof: pid=2385914 running
poll V2-prof: pid=2385914 running
poll V2-prof: pid=2385914 running
poll V2-prof: pid=2385914 running
poll V2-prof: pid=2385914 running
completed V2-prof in 270.0s; log=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/V2-prof.log
```
</details>

### 2026-09-23T10:09:31.159897+00:00 — Task 2 dense module fallback RED

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && python -m pytest tests/test_attribution_shim.py::test_shim_imports_without_upstream_package -v`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `52c3365914720a4eaacb7c5099f17d9437464966`
- **Wall time:** 0.166 s
- **Artifacts:** docs/attribution_vdn_runlog.md
- **Outcome:** Task 2 dense module fallback RED: exit 1

<details><summary>Verbatim output</summary>

```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /mnt/raid0nvme0/leyang/envs/vdn/bin/python
cachedir: .pytest_cache
rootdir: /mnt/raid0nvme0/leyang/Omni-Infinity
configfile: pyproject.toml
plugins: timeout-2.4.0, anyio-4.15.1
collecting ... collected 1 item

tests/test_attribution_shim.py::test_shim_imports_without_upstream_package FAILED [100%]

=================================== FAILURES ===================================
__________________ test_shim_imports_without_upstream_package __________________

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x710552e9a4b0>

    def test_shim_imports_without_upstream_package(monkeypatch):
        monkeypatch.delitem(sys.modules, "src", raising=False)
    
        module = _load_shim()
    
        assert module.UPSTREAM_PATCH_TARGETS
        assert "src" not in sys.modules
>       assert module._module_range_name("MiniMaxH3Attention") == "dense_attn"
               ^^^^^^^^^^^^^^^^^^^^^^^^^
E       AttributeError: module 'vdn_prof_sitecustomize' has no attribute '_module_range_name'

tests/test_attribution_shim.py:38: AttributeError
=========================== short test summary info ============================
FAILED tests/test_attribution_shim.py::test_shim_imports_without_upstream_package
============================== 1 failed in 0.02s ===============================
```
</details>

### 2026-09-23T10:10:01.562984+00:00 — Task 2 dense module fallback GREEN

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && python -m pytest tests/test_attribution_shim.py -v`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `52c3365914720a4eaacb7c5099f17d9437464966`
- **Wall time:** 0.159 s
- **Artifacts:** docs/attribution_vdn_runlog.md
- **Outcome:** Task 2 dense module fallback GREEN: exit 0

<details><summary>Verbatim output</summary>

```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /mnt/raid0nvme0/leyang/envs/vdn/bin/python
cachedir: .pytest_cache
rootdir: /mnt/raid0nvme0/leyang/Omni-Infinity
configfile: pyproject.toml
plugins: timeout-2.4.0, anyio-4.15.1
collecting ... collected 4 items

tests/test_attribution_shim.py::test_shim_imports_without_upstream_package PASSED [ 25%]
tests/test_attribution_shim.py::test_range_accounting_sums_mock_cuda_events PASSED [ 50%]
tests/test_attribution_shim.py::test_linear_branch_accounting_subtracts_only_its_own_gate PASSED [ 75%]
tests/test_attribution_shim.py::test_driver_rejects_busy_gpu PASSED      [100%]

============================== 4 passed in 0.01s ===============================
```
</details>

### 2026-09-23T10:17:06.657118+00:00 — GPU D-prof

- **Command:** `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=null render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-prof.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity/third_party/vdn-minimax-h3`
- **GPU index + model:** 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
- **Git SHA:** `52c3365914720a4eaacb7c5099f17d9437464966`
- **Wall time:** 420.003 s
- **Artifacts:** /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-prof.mp4.inference.json, /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-prof.log, /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-prof.mp4, /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-prof.ranges.json
- **Outcome:** D-prof: exit 0

### 2026-09-23T10:10:06.430071+00:00 — Task 2 D-prof module-boundary fallback rerun

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface VDN_RUNLOG=/mnt/raid0nvme0/leyang/Omni-Infinity/docs/attribution_vdn_runlog.md && python benchmarks/attribution_vdn.py --only D-prof --gpu 0 --force`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** 0; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
- **Git SHA:** `52c3365914720a4eaacb7c5099f17d9437464966`
- **Wall time:** 420.182 s
- **Artifacts:** results/vdn/attribution/D-prof.{log,mp4,inference.json,ranges.json}, docs/attribution_vdn_runlog.md
- **Outcome:** Task 2 D-prof module-boundary fallback rerun: exit 0

<details><summary>Verbatim output</summary>

```text
GPU idle check: 0, NVIDIA RTX PRO 6000 Blackwell Server Edition; memory/utilization=3, 0
run D-prof: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/raid0nvme0/leyang/Omni-Infinity/benchmarks/vdn_prof_shim HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface VDN_PROF_OUT=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-prof.ranges.json python src/inference/infer.py --config configs/inference/8nfe.yaml checkpoint=null render.prompt_file=prompts/example_0.pt render.out=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-prof.mp4 render.num_frames=222 render.seed=0 render.warmup_steps=2 render.record=true render.save_latents=false
poll D-prof: pid=2390652 running
poll D-prof: pid=2390652 running
poll D-prof: pid=2390652 running
poll D-prof: pid=2390652 running
poll D-prof: pid=2390652 running
poll D-prof: pid=2390652 running
poll D-prof: pid=2390652 running
poll D-prof: pid=2390652 running
poll D-prof: pid=2390652 running
poll D-prof: pid=2390652 running
poll D-prof: pid=2390652 running
poll D-prof: pid=2390652 running
poll D-prof: pid=2390652 running
poll D-prof: pid=2390652 running
completed D-prof in 420.0s; log=/mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/D-prof.log
```
</details>

### 2026-09-23T10:17:26.234607+00:00 — Task 2 report RED

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && python -m pytest tests/test_attribution_shim.py::test_profile_summary_drops_warmups_and_derives_residual -v`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `52c3365914720a4eaacb7c5099f17d9437464966`
- **Wall time:** 0.173 s
- **Artifacts:** docs/attribution_vdn_runlog.md
- **Outcome:** Task 2 report RED: exit 1

<details><summary>Verbatim output</summary>

```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /mnt/raid0nvme0/leyang/envs/vdn/bin/python
cachedir: .pytest_cache
rootdir: /mnt/raid0nvme0/leyang/Omni-Infinity
configfile: pyproject.toml
plugins: timeout-2.4.0, anyio-4.15.1
collecting ... collected 1 item

tests/test_attribution_shim.py::test_profile_summary_drops_warmups_and_derives_residual FAILED [100%]

=================================== FAILURES ===================================
___________ test_profile_summary_drops_warmups_and_derives_residual ____________

    def test_profile_summary_drops_warmups_and_derives_residual():
        driver_path = (
            Path(__file__).parents[1] / "benchmarks" / "attribution_vdn.py"
        )
        spec = importlib.util.spec_from_file_location(
            "attribution_vdn_summary",
            driver_path,
        )
        assert spec is not None and spec.loader is not None
        driver = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = driver
        spec.loader.exec_module(driver)
        step = {
            "step_total": {"total_ms": 1000.0, "calls": 1},
            "dense_attn": {"total_ms": 800.0, "calls": 52},
            "linear_calls": {"total_ms": 0.0, "calls": 370},
        }
    
>       summary = driver._summarize_steps([step, step, step], [1.02], warmups=2)
                  ^^^^^^^^^^^^^^^^^^^^^^^
E       AttributeError: module 'attribution_vdn_summary' has no attribute '_summarize_steps'

tests/test_attribution_shim.py:133: AttributeError
=========================== short test summary info ============================
FAILED tests/test_attribution_shim.py::test_profile_summary_drops_warmups_and_derives_residual
============================== 1 failed in 0.02s ===============================
```
</details>

### 2026-09-23T10:18:06.388236+00:00 — Task 2 report GREEN

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && python -m pytest tests/test_attribution_shim.py::test_profile_summary_drops_warmups_and_derives_residual -v`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `52c3365914720a4eaacb7c5099f17d9437464966`
- **Wall time:** 0.159 s
- **Artifacts:** docs/attribution_vdn_runlog.md
- **Outcome:** Task 2 report GREEN: exit 0

<details><summary>Verbatim output</summary>

```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /mnt/raid0nvme0/leyang/envs/vdn/bin/python
cachedir: .pytest_cache
rootdir: /mnt/raid0nvme0/leyang/Omni-Infinity
configfile: pyproject.toml
plugins: timeout-2.4.0, anyio-4.15.1
collecting ... collected 1 item

tests/test_attribution_shim.py::test_profile_summary_drops_warmups_and_derives_residual PASSED [100%]

============================== 1 passed in 0.01s ===============================
```
</details>

### 2026-09-23T10:18:11.839504+00:00 — Task 2 component QA

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && python benchmarks/attribution_vdn.py --report-components`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `52c3365914720a4eaacb7c5099f17d9437464966`
- **Wall time:** 0.044 s
- **Artifacts:** results/vdn/attribution/components.csv, docs/attribution_vdn_runlog.md
- **Outcome:** Task 2 component QA: exit 0

<details><summary>Verbatim output</summary>

```text
D-prof: step_total/step_seconds [0.9999, 0.9999] components_ms dense=29812.8 window=0.0 linear=0.0 gates=0.0 other=6055.4
V0-prof: step_total/step_seconds [0.9999, 0.9999] components_ms dense=0.0 window=9037.1 linear=2701.1 gates=174.1 other=11286.6
V1-prof: step_total/step_seconds [0.9999, 0.9999] components_ms dense=0.0 window=8374.2 linear=1726.2 gates=174.2 other=9595.6
V2-prof: step_total/step_seconds [0.9999, 0.9999] components_ms dense=0.0 window=8358.7 linear=1723.9 gates=174.7 other=5672.9
dense attention share = 83.12%
per-layer-equivalent ms (50 main blocks): D-prof=717.36, V0-prof=463.98, V1-prof=397.40, V2-prof=318.60
shape ratios: dense/hybrid=1.55x, hybrid/tuned=1.17x, tuned/fp8=1.25x; paper B200 dense/hybrid=1.73x, hybrid/(kernels+fp8)=1.53x
wrote /mnt/raid0nvme0/leyang/Omni-Infinity/results/vdn/attribution/components.csv
```
</details>

### 2026-09-23T10:18:21.811797+00:00 — Task 2 regression QA

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && python -m pytest tests/test_attribution_shim.py -q && ruff check benchmarks/ tests/test_attribution_shim.py`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `52c3365914720a4eaacb7c5099f17d9437464966`
- **Wall time:** 0.177 s
- **Artifacts:** docs/attribution_vdn_runlog.md
- **Outcome:** Task 2 regression QA: exit 1

<details><summary>Verbatim output</summary>

```text
.....                                                                    [100%]
5 passed in 0.01s
benchmarks/attribution_vdn.py:95:81: E501 Line too long (82 > 80)
   |
93 |     if len(measured) != len(step_seconds):
94 |         raise ValueError(
95 |             f"{len(measured)} measured ranges != {len(step_seconds)} step timings"
   |                                                                                 ^^ E501
96 |         )
97 |     range_names = set().union(*(step.keys() for step in measured))
   |

Found 1 error.
```
</details>

### 2026-09-23T10:18:34.433408+00:00 — Task 2 regression QA retry

- **Command:** `source /mnt/raid0nvme0/leyang/envs/vdn/bin/activate && export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface && python -m pytest tests/test_attribution_shim.py -q && ruff check benchmarks/ tests/test_attribution_shim.py`
- **CWD:** `/mnt/raid0nvme0/leyang/Omni-Infinity`
- **GPU index + model:** ""; 0, NVIDIA RTX PRO 6000 Blackwell Server Edition
1, NVIDIA RTX PRO 6000 Blackwell Server Edition
2, NVIDIA RTX PRO 6000 Blackwell Server Edition
3, NVIDIA RTX PRO 6000 Blackwell Server Edition
4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
5, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Git SHA:** `52c3365914720a4eaacb7c5099f17d9437464966`
- **Wall time:** 0.186 s
- **Artifacts:** docs/attribution_vdn_runlog.md
- **Outcome:** Task 2 regression QA retry: exit 0

<details><summary>Verbatim output</summary>

```text
.....                                                                    [100%]
5 passed in 0.01s
All checks passed!
```
</details>

#!/usr/bin/env python3
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""P2 Phase 3 / Phase 0 feasibility probe: CUDA graphs under block streaming.

Answers the make-or-break question from
``docs/superpowers/plans/2026-10-09-p2-phase3-cudagraph-block-streaming.md``:
can diffusers ``group_offload``-style H2D be redirected into pre-allocated,
pointer-stable arena buffers WITHOUT a CPU synchronization inside the
``torch.cuda.graph`` capture region?

Three probes, each isolated in its own subprocess (a failed capture can poison
the CUDA context):

- ``stock``:    diffusers ``apply_group_offloading`` (block_level, streams) on
                a toy model, then attempt whole-forward capture. Expected to
                FAIL: ``ModuleGroup._onload_from_memory`` calls
                ``stream.synchronize()`` (a CPU sync) on every onload.
- ``arena``:    the plan's design — double-buffered, pointer-stable device
                arenas; per-param H2D from fixed pinned host buffers issued on
                a copy stream inside the capture region, fork/join via events
                only. Whole-forward capture + N replays while HOST weights
                mutate between steps; bitwise compare vs eager.
- ``subgraph``: fallback granularity — per-block graphs capture compute only;
                H2D runs OUTSIDE the graphs on a copy stream, event-synced
                (still no CPU sync). Same mutation + bitwise bar.

Toy: 4 sequential Linear-pair blocks (bf16), 2 arenas, so arenas are REUSED
within one forward (groups 2,3 overwrite the slots of groups 0,1) — the same
hazard schedule the real streamed H3 transformer needs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "p2_phase3" / "phase0_probe.json"

N_BLOCKS = 4
N_ARENAS = 2
HIDDEN = 1024
N_STEPS = 5
WARMUP = 3


class ToyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(HIDDEN, HIDDEN, bias=True)
        self.fc2 = torch.nn.Linear(HIDDEN, HIDDEN, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.silu(self.fc1(x)))


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(ToyBlock() for _ in range(N_BLOCKS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def _seed_model() -> ToyModel:
    torch.manual_seed(0)
    return ToyModel().to(torch.bfloat16)


def _mutate_host(flat: torch.Tensor, step: int, group: int) -> None:
    generator = torch.Generator().manual_seed(1000 * step + group)
    flat.copy_(
        torch.randn(flat.shape, generator=generator, dtype=torch.float32).to(
            torch.bfloat16
        )
        * 0.02
    )


def probe_stock() -> dict:
    """Stock diffusers group offloading under capture (expected infeasible)."""
    from diffusers.hooks import apply_group_offloading

    model = _seed_model()
    apply_group_offloading(
        model,
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=1,
        use_stream=True,
        record_stream=False,
        low_cpu_mem_usage=False,
    )
    x = torch.randn(8, HIDDEN, device="cuda", dtype=torch.bfloat16)
    for _ in range(WARMUP):
        model(x)
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(side):
            with torch.cuda.graph(graph, stream=side):
                model(x)
    except Exception as exc:  # noqa: BLE001 - verdict evidence
        return {
            "capture_ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
    return {"capture_ok": True}


class ArenaStreaming:
    """Pointer-stable double-buffered arenas with event-only stream sync.

    Group ``g`` owns arena slot ``g % N_ARENAS``. Every parameter's ``.data``
    is rebound ONCE to a view inside its slot, so pointers the captured graph
    reads never change. Per-step H2D re-fills the slots from fixed pinned host
    buffers on a dedicated copy stream; hazards are enforced with events:

    - compute(g) waits copy_done(g)
    - copy(g)    waits compute_done(g - N_ARENAS)   (slot reuse)
    """

    def __init__(self, model: ToyModel) -> None:
        self.copy_stream = torch.cuda.Stream()
        self.copy_done = [torch.cuda.Event() for _ in range(N_BLOCKS)]
        self.compute_done = [torch.cuda.Event() for _ in range(N_BLOCKS)]
        self.fork = torch.cuda.Event()
        self.host: list[torch.Tensor] = []
        self.host_views: list[list[torch.Tensor]] = []
        self.arena_views: list[list[torch.Tensor]] = []
        group_numels = []
        for block in model.blocks:
            group_numels.append(sum(p.numel() for p in block.parameters()))
        slot_numel = max(group_numels)
        self.arenas = [
            torch.empty(slot_numel, device="cuda", dtype=torch.bfloat16)
            for _ in range(N_ARENAS)
        ]
        for index, block in enumerate(model.blocks):
            flat = torch.empty(
                group_numels[index], dtype=torch.bfloat16
            ).pin_memory()
            arena = self.arenas[index % N_ARENAS]
            offset = 0
            h_views, a_views = [], []
            for param in block.parameters():
                numel = param.numel()
                h_view = flat[offset : offset + numel].view(param.shape)
                a_view = arena[offset : offset + numel].view(param.shape)
                h_view.copy_(param.data)
                param.data = a_view  # pointer-stable rebinding, done once
                h_views.append(h_view)
                a_views.append(a_view)
                offset += numel
            self.host.append(flat)
            self.host_views.append(h_views)
            self.arena_views.append(a_views)

    def copy_group(self, group: int) -> None:
        """Issue H2D for ``group`` on the copy stream (event-guarded)."""
        if group >= N_ARENAS:
            self.copy_stream.wait_event(self.compute_done[group - N_ARENAS])
        with torch.cuda.stream(self.copy_stream):
            for a_view, h_view in zip(
                self.arena_views[group], self.host_views[group], strict=True
            ):
                a_view.copy_(h_view, non_blocking=True)
        self.copy_done[group].record(self.copy_stream)

    def streamed_forward(self, model: ToyModel, x: torch.Tensor):
        """Whole forward with pipelined arena refill; no CPU sync."""
        stream = torch.cuda.current_stream()
        self.fork.record(stream)
        self.copy_stream.wait_event(self.fork)
        self.copy_group(0)
        for index, block in enumerate(model.blocks):
            if index + 1 < N_BLOCKS:
                self.copy_group(index + 1)
            stream.wait_event(self.copy_done[index])
            x = block(x)
            self.compute_done[index].record(stream)
        return x

    def write_step_weights(self, step: int) -> None:
        for group, flat in enumerate(self.host):
            _mutate_host(flat, step, group)

    def load_reference(self, reference: ToyModel, step: int) -> None:
        for group, block in enumerate(reference.blocks):
            offset = 0
            flat = torch.empty_like(self.host[group])
            _mutate_host(flat, step, group)
            for param in block.parameters():
                numel = param.numel()
                param.data = (
                    flat[offset : offset + numel].view(param.shape).to("cuda")
                )
                offset += numel


def probe_arena() -> dict:
    """Whole-forward capture against pointer-stable arenas (plan design)."""
    model = _seed_model()
    streaming = ArenaStreaming(model)
    x = torch.randn(8, HIDDEN, device="cuda", dtype=torch.bfloat16)
    static_x = x.clone()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(WARMUP):
            streaming.streamed_forward(model, static_x)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(side):
            with torch.cuda.graph(graph, stream=side):
                static_out = streaming.streamed_forward(model, static_x)
    except Exception as exc:  # noqa: BLE001 - verdict evidence
        return {
            "capture_ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }

    reference = _seed_model().to("cuda")
    steps = []
    for step in range(N_STEPS):
        torch.cuda.synchronize()  # outside capture: safe to mutate host
        streaming.write_step_weights(step)
        graph.replay()
        torch.cuda.synchronize()
        replay_out = static_out.clone()
        streaming.load_reference(reference, step)
        with torch.no_grad():
            eager_out = reference(static_x)
        steps.append(
            {
                "step": step,
                "bitwise": bool(torch.equal(replay_out, eager_out)),
                "max_abs_diff": float(
                    (replay_out.float() - eager_out.float()).abs().max()
                ),
            }
        )
    return {
        "capture_ok": True,
        "replay_steps": steps,
        "all_bitwise": all(entry["bitwise"] for entry in steps),
    }


def probe_subgraph() -> dict:
    """Per-block sub-graphs; H2D outside the graphs, event-synced."""
    model = _seed_model()
    streaming = ArenaStreaming(model)
    x = torch.randn(8, HIDDEN, device="cuda", dtype=torch.bfloat16)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    statics: list[torch.Tensor] = [x.clone()]
    graphs: list[torch.cuda.CUDAGraph] = []
    with torch.cuda.stream(side):
        for _ in range(WARMUP):
            streaming.streamed_forward(model, statics[0])
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    try:
        for index, block in enumerate(model.blocks):
            # Weights must be resident in the slot while capture records.
            streaming.copy_group(index)
            side.wait_event(streaming.copy_done[index])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(side):
                with torch.cuda.graph(graph, stream=side):
                    statics.append(block(statics[index]))
            streaming.compute_done[index].record(side)
            graphs.append(graph)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - verdict evidence
        return {
            "capture_ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }

    reference = _seed_model().to("cuda")
    stream = torch.cuda.current_stream()
    steps = []
    for step in range(N_STEPS):
        torch.cuda.synchronize()
        streaming.write_step_weights(step)
        streaming.fork.record(stream)
        streaming.copy_stream.wait_event(streaming.fork)
        for index in range(N_BLOCKS):
            streaming.copy_group(index)
            stream.wait_event(streaming.copy_done[index])
            graphs[index].replay()
            streaming.compute_done[index].record(stream)
        torch.cuda.synchronize()
        replay_out = statics[-1].clone()
        streaming.load_reference(reference, step)
        with torch.no_grad():
            eager_out = reference(statics[0])
        steps.append(
            {
                "step": step,
                "bitwise": bool(torch.equal(replay_out, eager_out)),
                "max_abs_diff": float(
                    (replay_out.float() - eager_out.float()).abs().max()
                ),
            }
        )
    return {
        "capture_ok": True,
        "replay_steps": steps,
        "all_bitwise": all(entry["bitwise"] for entry in steps),
    }


PROBES = {
    "stock": probe_stock,
    "arena": probe_arena,
    "subgraph": probe_subgraph,
}


def run_single(name: str) -> int:
    torch.manual_seed(0)
    try:
        result = PROBES[name]()
    except Exception:  # noqa: BLE001 - subprocess boundary
        result = {"capture_ok": False, "error": traceback.format_exc()[-800:]}
    print(json.dumps({name: result}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=[*PROBES, "all"], default="all")
    parser.add_argument("--json-out", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.probe != "all":
        return run_single(args.probe)

    combined: dict[str, dict] = {}
    for name in PROBES:
        completed = subprocess.run(
            [sys.executable, __file__, "--probe", name],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        lines = [
            line
            for line in completed.stdout.strip().splitlines()
            if line.startswith("{")
        ]
        if completed.returncode == 0 and lines:
            combined.update(json.loads(lines[-1]))
        else:
            combined[name] = {
                "capture_ok": False,
                "error": (completed.stderr or completed.stdout)[-800:],
            }
        print(f"[{name}] {json.dumps(combined[name])[:300]}")

    arena = combined.get("arena", {})
    subgraph = combined.get("subgraph", {})
    whole_forward_ok = bool(
        arena.get("capture_ok") and arena.get("all_bitwise")
    )
    subgraph_ok = bool(
        subgraph.get("capture_ok") and subgraph.get("all_bitwise")
    )
    combined["verdict"] = {
        "stock_hooks_capturable": bool(
            combined.get("stock", {}).get("capture_ok")
        ),
        "whole_forward_arena_feasible": whole_forward_ok,
        "subgraph_arena_feasible": subgraph_ok,
        "feasible": whole_forward_ok or subgraph_ok,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(combined, indent=2) + "\n")
    print(json.dumps(combined["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

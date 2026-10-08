# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import torch

from benchmarks import bench_graph_denoise


def _stats(*, captures=0, replays=0, failures=0, fallbacks=None):
    return {
        "captures": captures,
        "replays": replays,
        "capture_failures": failures,
        "fallback_reasons": fallbacks or {},
    }


def test_classify_graph_step_distinguishes_warmup_capture_replay_and_failure():
    empty = _stats()
    assert (
        bench_graph_denoise.classify_graph_step(
            empty, _stats(fallbacks={"warmup_not_done": 1})
        )
        == "warmup"
    )
    assert (
        bench_graph_denoise.classify_graph_step(empty, _stats(captures=1))
        == "capture"
    )
    assert (
        bench_graph_denoise.classify_graph_step(empty, _stats(replays=1))
        == "replay"
    )
    assert (
        bench_graph_denoise.classify_graph_step(
            empty,
            _stats(failures=1, fallbacks={"capture_failed": 1}),
        )
        == "capture-failed"
    )


def test_replay_summary_excludes_warmup_and_capture_rows():
    rows = [
        {
            "phase": "warmup",
            "step_wall_ms": 100.0,
            "compute_busy_ms": 50.0,
            "copy_busy_ms": 40.0,
            "launch_gap_ms": 10.0,
            "kernel_count": 100,
        },
        {
            "phase": "capture",
            "step_wall_ms": 200.0,
            "compute_busy_ms": 90.0,
            "copy_busy_ms": 80.0,
            "launch_gap_ms": 30.0,
            "kernel_count": 100,
        },
        {
            "phase": "replay",
            "step_wall_ms": 80.0,
            "compute_busy_ms": 45.0,
            "copy_busy_ms": 30.0,
            "launch_gap_ms": 5.0,
            "kernel_count": 3,
        },
        {
            "phase": "replay",
            "step_wall_ms": 82.0,
            "compute_busy_ms": 47.0,
            "copy_busy_ms": 29.0,
            "launch_gap_ms": 6.0,
            "kernel_count": 3,
        },
    ]

    summary = bench_graph_denoise.replay_summary(rows)

    assert summary == {
        "steps": 2,
        "median_step_wall_ms": 81.0,
        "median_compute_busy_ms": 46.0,
        "median_copy_busy_ms": 29.5,
        "median_launch_gap_ms": 5.5,
        "median_kernel_count": 3.0,
    }


def test_compare_output_trees_reports_each_tensor_rms_relative_error():
    graph = (torch.tensor([1.0, 2.0]), torch.tensor([3.0]))
    eager = (torch.tensor([1.0, 2.0]), torch.tensor([4.0]))

    comparison = bench_graph_denoise.compare_output_trees(graph, eager)

    assert comparison[0] == 0.0
    assert comparison[1] == 0.25


def test_parse_driver_version_handles_open_kernel_module_banner():
    banner = (
        "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  "
        "590.48.01  Release Build"
    )

    assert bench_graph_denoise.parse_driver_version(banner) == "590.48.01"


def test_graph_execution_gate_rejects_eager_fallback_and_capture_failure():
    assert bench_graph_denoise.graph_execution_gate_passes("resident", None)
    assert not bench_graph_denoise.graph_execution_gate_passes(
        "graph-resident", _stats()
    )
    assert not bench_graph_denoise.graph_execution_gate_passes(
        "graph-resident", _stats(captures=1, replays=3, failures=1)
    )
    assert bench_graph_denoise.graph_execution_gate_passes(
        "graph-resident", _stats(captures=1, replays=3)
    )

# Multi-GPU platform validation — 6× RTX PRO 6000 Blackwell (PCIe)

Phase 0 of the [P7 plan](superpowers/plans/2026-10-07-p7-multigpu.md)
(tracking [#42](https://github.com/EfficientMoE/Omni-Infinity/issues/42)).
All numbers below are **measured on this host** (`gala2`, 2026-10-08), not
survey ceilings. Host-level changes are **documented only** — nothing in
BIOS, kernel command line, or modprobe was modified by this validation.

## Host inventory

| Item | Value |
|---|---|
| Host | `gala2`, 2× AMD EPYC 9355 (dual-socket, NUMA0 + NUMA1) |
| Kernel | `6.8.0-134-generic`, cmdline: `ro` only (no iommu/acs overrides) |
| GPUs | 6× RTX PRO 6000 Blackwell 96 GB (0–3 Server Edition on NUMA0, 4–5 Max-Q Workstation on NUMA1), PCIe Gen5 ×16 |
| Driver | 590.48.01 |
| CUDA | nvcc 13.1 (12.9/13.0/13.1 toolkits installed) |
| NCCL | system `libnccl2 2.28.9-1+cuda13.1` (used by nccl-tests); torch 2.12.0+cu130 bundles NCCL 2.29.7 |

## Phase 0 checklist — audit result

| Plan item | Measured state | Verdict |
|---|---|---|
| BIOS: IOMMU **off**, ACS off | IOMMU is **ON** (AMD-Vi, 79 groups, type `DMA-FQ` = translated, not passthrough). ACS caps not readable without root. However, each PIX GPU pair shares one IOMMU group (0+1 → group 21, 2+3 → group 2, 4+5 → group 43), which indicates ACS is not isolating devices below the shared switches. | **Works as-is.** P2P measured at full PCIe Gen5 bandwidth with no hangs (below). Turning IOMMU off remains the recommended end state if instability appears; see "Required host changes". |
| `options nvidia_uvm uvm_disable_hmm=1` | **Not set** — `/sys/module/nvidia_uvm/parameters/uvm_disable_hmm = N` (HMM enabled). | **Document-only change pending.** No UVM-related instability observed during validation (CUDA P2P + NCCL paths do not exercise HMM); set it before production multi-GPU serving. |
| `ForceP2P` registry override | **Already set**: `/etc/modprobe.d/nvidia.conf` carries `NVreg_RegistryDwords="PeerMappingOverride=1;"` and `/proc/driver/nvidia/params` confirms `RegistryDwords: "PeerMappingOverride=1;"`. | ✅ Active. `torch.cuda.can_device_access_peer` is `True` for **all 30 ordered pairs**, including cross-socket. |
| `nvidia-smi topo -m` + `p2pBandwidthLatencyTest` | Captured below. | ✅ |
| `nccl-tests` all_gather/all_to_all tuned vs untuned | Captured below. **Platform deviation:** `NCCL_P2P_LEVEL=SYS` is *harmful* on this dual-socket host; `NCCL_MIN_NCHANNELS=8` is the knob that pays. | ✅ (with revised tuning) |
| Fallback `NCCL_P2P_DISABLE=1` | Captured below: −43 % (2 GPU) to −48 % (6 GPU, vs tuned). | ✅ |

### Required host changes (not applied; need root/BIOS — ask before changing)

1. `echo 'options nvidia_uvm uvm_disable_hmm=1' | sudo tee /etc/modprobe.d/nvidia-uvm-p2p.conf` + initramfs refresh + reboot (or module reload).
2. Optional hardening per NVIDIA guidance for PCIe P2P: BIOS IOMMU **disabled**
   (current: enabled, `DMA-FQ`) and ACS disabled on the GPU switch ports.
   Current measurements show full-speed, hang-free P2P without these, so they
   are listed as contingency, to be applied only if P2P instability or
   bandwidth collapse is observed under sustained multi-GPU load.
3. ACS verification itself requires root (`lspci -vvv` extended caps); record
   `ACSCtl` for the bridges above the GPUs when root access is available.

## Topology

`nvidia-smi topo -m`:

```text
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    CPU Affinity    NUMA
GPU0     X      PIX     NODE    NODE    SYS     SYS     0-31,64-95      0
GPU1    PIX      X      NODE    NODE    SYS     SYS     0-31,64-95      0
GPU2    NODE    NODE     X      PIX     SYS     SYS     0-31,64-95      0
GPU3    NODE    NODE    PIX      X      SYS     SYS     0-31,64-95      0
GPU4    SYS     SYS     SYS     SYS      X      PIX     32-63,96-127    1
GPU5    SYS     SYS     SYS     SYS     PIX      X      32-63,96-127    1
```

Three pair classes: **PIX** (0↔1, 2↔3, 4↔5 — shared PCIe switch),
**NODE** (0/1↔2/3 — same socket, across host bridges), **SYS**
(0–3 ↔ 4–5 — across the socket interconnect).

## p2pBandwidthLatencyTest (cuda-samples v12.9, sm120 build)

Unidirectional, P2P **enabled** (GB/s) — ~52–55 intra-socket, ~46–50 cross-socket:

```text
   D\D     0      1      2      3      4      5
     0 1449.4  52.2   53.5   52.6   46.7   48.5
     1  54.5 1488.1   53.8   54.1   48.5   48.4
     2  55.4   55.0 1493.8   55.1   47.0   50.1
     3  54.2   54.1   53.8 1485.3   49.4   49.9
     4  49.3   47.4   46.3   48.2 1511.1   54.0
     5  50.2   50.3   47.6   51.7   53.9 1514.1
```

Unidirectional, P2P **disabled**: flat ~40–42 GB/s (staged through host).
P2P gain: **+30 %** bandwidth.

Bidirectional, P2P **enabled** (GB/s):

```text
   D\D     0      1      2      3      4      5
     0 1467.1  52.1  104.3  104.6   93.9   89.0
     1  52.0 1490.2  102.9  102.9   86.5   93.5
     2 104.0  102.8 1482.4   52.0   92.6   94.1
     3 102.6  105.6   52.0 1486.6   94.2   86.4
     4  94.5   93.3   91.5   94.0 1500.9   51.3
     5  93.5   86.0   94.8   94.2   51.1 1498.0
```

**Placement-relevant anomaly:** PIX pairs (0↔1, 2↔3, 4↔5) are capped at
~52 GB/s *bidirectional* — both directions share the switch's single
upstream Gen5 ×16 link — while NODE pairs reach ~104 GB/s and SYS pairs
~86–94 GB/s. For bidirectionally-chatty role pairs, prefer NODE pairs over
PIX pairs.

Latency (µs): P2P enabled **0.44–0.59** everywhere (incl. cross-socket);
P2P disabled 14.3–18.0. **≈30× better latency with P2P.** No hangs in any
cell of the full 6×6 matrix (dual-socket hang risk from the plan did not
materialize on this platform/driver).

## nccl-tests (2.28.9, single process, `-b 1M -e 512M -f 2`)

Avg bus bandwidth in GB/s (per-size peak at 512 MB in parentheses where it
differs materially). "untuned" = no NCCL env; "chan8" =
`NCCL_MIN_NCHANNELS=8`; "SYS+chan8" = `NCCL_P2P_LEVEL=SYS
NCCL_MIN_NCHANNELS=8` (the plan's suggested tuning); "p2p-off" =
`NCCL_P2P_DISABLE=1`.

| topology | collective | untuned | chan8 | SYS+chan8 | p2p-off |
|---|---|---:|---:|---:|---:|
| 2 GPU PIX (0,1) | all_gather | 17.7 (18.9) | — | 18.4 | 10.1 |
| 2 GPU PIX (0,1) | alltoall | 19.8 (20.3) | — | 17.0 | 9.7 |
| 2 GPU NODE (0,2) | alltoall | 24.6 | — | — | — |
| 2 GPU SYS (0,4) | all_gather | 23.6 | — | — | — |
| 2 GPU SYS (0,4) | alltoall | 21.8 | — | — | — |
| 4 GPU NUMA0 (0–3) | all_gather | 17.6 (19.1) | — | 17.4 | 15.6 |
| 4 GPU NUMA0 (0–3) | alltoall | 16.5 (17.1) | 16.7 | **19.4** | 14.7 |
| 6 GPU (0–5) | all_gather | 7.0 (8.9) | **17.3 (18.7)** | 7.3 | 9.0 |
| 6 GPU (0–5) | alltoall | 7.3 | **16.6 (19.0)** | **0.53** ⚠ | 7.2 |

2-GPU algorithmic bandwidth (what a latent handoff actually sees) is 2× the
all_gather/alltoall bus bandwidth above: **~40 GB/s algbw** on a PIX pair,
~44–49 GB/s on NODE/SYS pairs.

### Platform deviations from the plan's expectations

1. **`NCCL_P2P_LEVEL=SYS` is pathological across sockets** — 6-GPU alltoall
   collapses to **0.53 GB/s** (vs 7.3 untuned, 16.6 with chan8). Forcing
   SM-initiated P2P stores over the inter-socket link is the failure mode;
   CE-based copies (what `p2pBandwidthLatencyTest` and plain
   `cudaMemcpyPeer`/`Tensor.copy_` use) cross the socket at 46–50 GB/s just
   fine. `NCCL_P2P_LEVEL=SYS` is only safe *within* a socket (4-GPU alltoall
   16.5 → 19.4).
2. **`NCCL_MIN_NCHANNELS=8` is the high-value knob** for any group spanning
   both sockets: 6-GPU collectives improve ~2.4× (7.0 → 17.3 all_gather,
   7.3 → 16.6 alltoall).
3. The plan's "~40 GB/s class vs ~22 untuned" expectation does not transfer
   to this dual-socket EPYC host; the realistic envelope is **~19 GB/s bus
   bandwidth** for ≥4-GPU collectives and ~40–49 GB/s algbw for pairwise
   transfers.

### Recommended NCCL environment

```bash
# Any communicator confined to one socket (e.g. denoiser group on GPUs 0-3):
NCCL_P2P_LEVEL=SYS NCCL_MIN_NCHANNELS=8

# Any communicator spanning both sockets (e.g. all 6 GPUs):
NCCL_MIN_NCHANNELS=8          # do NOT set NCCL_P2P_LEVEL=SYS

# Fallback if P2P instability appears (measured cost: −43 % on 2-GPU,
# −48 % on 6-GPU vs the tuned rows above):
NCCL_P2P_DISABLE=1
```

## Implications for P7 design

- **Phase 0 gate: PASS.** P2P works at full PCIe Gen5 bandwidth with the
  current host config; no dual-socket hangs. Code work may proceed.
- **Stage handoffs (Phase 1)** move one latent/embed tensor at a time
  between fixed device pairs → use direct CE copies (`Tensor.to(device)` /
  `cudaMemcpyPeer`), which are fast in *every* pair class (46–55 GB/s uni),
  rather than NCCL collectives. Cross-socket handoff is acceptable.
- **Collective groups (Phase 2 USP, CFG)** must be pinned **within one
  socket** (degree ≤4 on GPUs 0–3, or a PIX/NODE subset). Never span
  sockets with `NCCL_P2P_LEVEL=SYS`.
- **Role placement:** put the bidirectionally-chatty pair (denoiser group
  internals) on NODE pairs (0↔2, 1↔3: 104 GB/s bidi) in preference to PIX
  pairs (52 GB/s bidi cap). Encoder→denoiser→decoder handoffs are
  unidirectional and placement-insensitive.
- **Sizing check:** a 256p/120f latent (≈ `1×C×120×16×16` bf16) is a few
  tens of MB — a sub-millisecond copy at 50 GB/s. Handoff bandwidth is not
  a bottleneck for stage pipelining; latency discipline (streams, events)
  is what matters.

## Reproduction

```bash
# p2pBandwidthLatencyTest (cuda-samples v12.9 tag; sample was removed from master)
git clone --depth 1 --filter=blob:none --sparse https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples && git fetch --depth 1 origin tag v12.9
git show v12.9:Samples/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest.cu > /tmp/p2p.cu
# extract Common/*.h the same way, then:
nvcc -O2 -I common -o p2pBandwidthLatencyTest /tmp/p2p.cu -gencode arch=compute_120,code=sm_120
./p2pBandwidthLatencyTest

# nccl-tests
git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests && make -j CUDA_HOME=/usr/local/cuda-13.1 NCCL_HOME=/usr
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NCCL_MIN_NCHANNELS=8 \
  ./build/all_gather_perf -b 1M -e 512M -f 2 -g 6
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_LEVEL=SYS NCCL_MIN_NCHANNELS=8 \
  ./build/alltoall_perf -b 1M -e 512M -f 2 -g 4
```

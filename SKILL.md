---
name: gpu-programming
description: Use for standalone GPU programming work across CUDA, HIP/ROCm, OpenCL, Vulkan compute, Triton, CUTLASS, cuBLAS/cuDNN, TensorRT, NCCL, Nsight, GPU memory, debugging, benchmarking, parallel patterns, warp primitives, stencil/convolution, CUDA Graphs, and NVENC/NVDEC. Trigger for GPU kernels, profiling, optimization, portability, multi-GPU collectives, and hardware-accelerated video. Do not use for GPU cluster provisioning, scheduler orchestration, or general ML modeling without GPU implementation work.
---

# GPU Programming

Use the focused guides under `references/guides/` to implement, debug, profile, and optimize GPU software. The repository is self-contained and has no Babysitter runtime dependency.

## Workflow

1. Inspect the user's code, target hardware, toolchain, data types, shapes, and correctness requirements.
2. Select the smallest relevant guide from the routing table below. Read a second guide only when the task crosses concerns.
3. Check that required vendor tools are installed before invoking them. Do not assume that a GPU is available.
4. Establish a correct baseline before optimization.
5. Profile or benchmark with warmups, synchronization at measurement boundaries, repeated samples, and explicit hardware/software versions.
6. Make the smallest change that addresses the measured bottleneck.
7. Re-run correctness checks and compare performance against the baseline.
8. Report commands, environment, measurements, limitations, and any unverified assumptions.

## Guide Routing

| Topic | Guide |
|---|---|
| CUDA compilation, launch configuration, PTX/SASS, runtime APIs | `references/guides/cuda-toolkit/GUIDE.md` |
| Compute Sanitizer, CUDA-GDB, races, memory errors | `references/guides/cuda-debugging/GUIDE.md` |
| CUDA Graph capture, updates, and launch overhead | `references/guides/cuda-graphs/GUIDE.md` |
| CUTLASS and Triton kernel implementation/tuning | `references/guides/cutlass-triton/GUIDE.md` |
| GPU benchmark design and regression detection | `references/guides/gpu-benchmarking/GUIDE.md` |
| Coalescing, caches, shared memory, bank conflicts | `references/guides/gpu-memory-analysis/GUIDE.md` |
| CUDA Unified Memory, prefetch, advice, oversubscription | `references/guides/unified-memory/GUIDE.md` |
| Warp shuffles, voting, cooperative groups, divergence | `references/guides/warp-primitives/GUIDE.md` |
| Reduction, scan, histogram, sorting, compaction | `references/guides/parallel-patterns/GUIDE.md` |
| Stencil and convolution tiling, halos, boundaries | `references/guides/stencil-convolution/GUIDE.md` |
| Nsight Systems and Nsight Compute profiling | `references/guides/nsight-profiler/GUIDE.md` |
| cuBLAS, cuDNN, mixed precision, tensor cores | `references/guides/cublas-cudnn/GUIDE.md` |
| TensorRT conversion, precision, calibration, plugins | `references/guides/tensorrt-optimization/GUIDE.md` |
| NCCL/RCCL collectives and multi-GPU communication | `references/guides/nccl-communication/GUIDE.md` |
| HIP, ROCm, hipify, and CUDA portability | `references/guides/hip-rocm/GUIDE.md` |
| Portable OpenCL runtime and kernel development | `references/guides/opencl-runtime/GUIDE.md` |
| Vulkan compute shaders and pipelines | `references/guides/vulkan-compute/GUIDE.md` |
| NVENC/NVDEC video pipelines with CUDA interop | `references/guides/nvenc-nvdec/GUIDE.md` |

## Cross-Guide Combinations

- CUDA kernel tuning: `cuda-toolkit` + `nsight-profiler` + the measured bottleneck guide.
- Correct performance testing: `gpu-benchmarking` + `cuda-debugging`.
- Custom matrix kernels: `cutlass-triton` + `cublas-cudnn` as the reference baseline.
- Multi-GPU optimization: `nccl-communication` + `gpu-benchmarking`; add `unified-memory` only if managed memory is involved.
- CUDA-to-AMD migration: `hip-rocm` + the original CUDA topic guide.

## Operating Rules

- Never claim a speedup without a comparable before/after measurement.
- Distinguish kernel time from end-to-end time and synchronization overhead.
- Treat sample code as a starting point; adapt architecture targets and API versions to the user's environment.
- Check every GPU API call at system boundaries and check asynchronous kernel errors.
- Preserve numerical accuracy requirements when changing precision or enabling fast math.
- Prefer vendor documentation and profiler evidence when version-sensitive behavior conflicts with a guide.

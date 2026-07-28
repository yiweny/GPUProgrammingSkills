# GPU Programming Skills for Grok Build

A standalone Grok Build skill for GPU software development. It packages 18 focused guides for CUDA, HIP/ROCm, OpenCL, Vulkan compute, Triton, CUTLASS, GPU libraries, profiling, debugging, memory optimization, benchmarking, multi-GPU communication, and hardware video processing.

The repository root is the skill directory, following the same clone-and-use pattern as [KernelWiki](https://github.com/mit-han-lab/KernelWiki). It does not require Babysitter, an MCP server, a package install, or a workflow runtime.

## Install

```bash
git clone https://github.com/yiweny/GPUProgrammingSkills.git ~/.grok/skills/gpu-programming
```

Grok Build will load `SKILL.md` from the clone root. Invoke it with `/gpu-programming` or ask a GPU programming question that matches its description.

## Included guides

| Area | Guides |
|---|---|
| CUDA development | `cuda-toolkit`, `cuda-debugging`, `cuda-graphs` |
| Performance | `nsight-profiler`, `gpu-benchmarking`, `gpu-memory-analysis`, `unified-memory`, `warp-primitives` |
| Algorithms and kernels | `parallel-patterns`, `stencil-convolution`, `cutlass-triton` |
| NVIDIA libraries | `cublas-cudnn`, `tensorrt-optimization`, `nccl-communication`, `nvenc-nvdec` |
| Cross-platform GPU | `hip-rocm`, `opencl-runtime`, `vulkan-compute` |

All reference material lives in `references/guides/<topic>/GUIDE.md`. The root `SKILL.md` routes Grok to the smallest relevant guide and defines a correctness-first profiling workflow.

## Scope

Included content must work as standalone GPU programming guidance. This repository intentionally excludes:

- Babysitter JavaScript processes and workflow orchestration
- Babysitter agent definitions and role metadata
- Backlogs, process catalogs, and specialization planning documents
- Babysitter configuration examples, SDK imports, and process-integration sections
- General agent tooling and GPU cluster provisioning workflows

## Verification

The repository is checked for:

- Valid Grok skill frontmatter at the root
- Exactly 18 curated guide files
- No Babysitter SDK, configuration, process, or runtime references
- No JavaScript workflow or agent-definition files
- Preserved upstream MIT attribution

## Provenance and license

The guides were adapted from the GPU programming specialization in [`a5c-ai/babysitter`](https://github.com/a5c-ai/babysitter/tree/main/library/specializations/gpu-programming/skills), pinned to commit `44a5d58b47b93962f143914e0652d9e3ae5dcf2c`. Babysitter-specific integration material was removed and the remaining technical references were repackaged for standalone Grok Build use.

The adapted material remains available under the upstream MIT License. See `LICENSE`.

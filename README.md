# GPU Compute Runtime

A GPU compute runtime built from scratch. Compiles shaders, dispatches them on the GPU, and profiles performance — all from Rust with a C++ compiler frontend.

Built to understand what GPU driver stacks (NVIDIA CUDA, Vulkan compute, Qualcomm Adreno).

---

## Why this exists

Running computation on a GPU requires either CUDA (locked to NVIDIA) or a high-level framework like PyTorch that hides everything from you. This project sits right in the middle. It gives a developer direct, vendor-agnostic access to GPU compute through Vulkan, with a lightweight toolchain they control entirely. No framework overhead, no vendor lock-in, works on AMD, NVIDIA, and Intel GPUs equally.

## Who would use this

**Game devs** writing custom compute shaders for particle systems, cloth simulation, or GPU-driven culling, who would need to know if their shader is hitting peak bandwidth without pulling in a full profiler (NSight).

**HPC and scientific computing developers** running simulations (fluid dynamics, molecular dynamics, finite element analysis) who want to prototype a GPU-accelerated kernel quickly before committing to a full CUDA or OpenCL implementation.

**Graphics/compute engineers at hardware companies** who need to test how a shader or algorithm behaves on a specific GPU, measure bandwidth utilization, and iterate. This is a lightweight version of internal tools these teams build and use daily.

**Embedded and cross-platform developers** who can't use CUDA because their target hardware is AMD or ARM Mali, and need a portable compute solution that runs the same code everywhere Vulkan runs.

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│                   Rust Runtime                    │
│                                                   │
│  GpuContext → GpuBuffer → ComputePipeline        │
│                                    ↓              │
│                           Dispatcher → dispatch() │
│                                    ↓              │
│  GpuProfiler ← timestamp queries ←┘              │
│                                                   │
│  FFI bridge (extern "C")                          │
└──────────────────────┬───────────────────────────┘
                       │ SPIR-V bytes
┌──────────────────────▼───────────────────────────┐
│                C++ Compiler Frontend               │
│                                                   │
│  .glsl / .ks → Lexer → Parser → AST → Opt passes  │
│                                       ↓            │
│                               CodeGen → SPIR-V     │
└──────────────────────────────────────────────────┘
```

- **Rust runtime** (`runtime/`) — library crate that talks to Vulkan via `ash`. Handles device setup, buffer management, shader pipeline creation, dispatch, and profiling.
- **C++ compiler** (`compiler/`) — compiles GLSL (via `shaderc`) or a custom KernelScript language to SPIR-V. Includes hand-written optimization passes (constant folding, loop unrolling).
- **Kernels** (`kernels/`) — benchmark compute shaders: prefix sum, histogram, sparse matrix-vector multiply.

---

## Build

### Prerequisites

- Rust 1.75+ (`rustup`)
- Vulkan SDK 1.3+ (from [LunarG](https://vulkan.lunarg.com/))
- A GPU with Vulkan support

Verify your setup:

```bash
vulkaninfo   # should show ur GPU
cargo build --manifest-path runtime/Cargo.toml
```

### Build & test

```bash
cargo build --manifest-path runtime/Cargo.toml
cargo test --manifest-path runtime/Cargo.toml
```

---

## Design Decisions

- **Vulkan over CUDA** — vendor-agnostic. Mirrors how Qualcomm/Intel/AMD compute teams target multiple GPU architectures.
- **Ash over Vulkano** — `ash` is a thin wrapper over the raw Vulkan API. More boilerplate, but more control and closer to the C API that job postings ask about.
- **Explicit destroy over Drop** — Vulkan requires reverse-order teardown (Allocator → Device → Instance). Using `ManuallyDrop` + explicit `destroy_*` methods avoids lifetime complexity while learning.
- **Host-visible memory first** — Phase 1 uses `CpuToGpu`/`GpuToCpu` memory for simplicity. Staging buffers for `DeviceLocal` memory come later for performance.
- **`extern "C"` FFI bridge** — C++ compiler compiled via `cc`/`cmake-rs` in the build script, exposed to Rust via a simple C ABI. Avoids C++/Rust binding complexity.

---

## Benchmarks

Check docs/Benchmarks/

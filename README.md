# GPU Compute Runtime

A minimal GPU compute runtime built from scratch. Compiles shaders, dispatches them on the GPU, and profiles performance — all from Rust with a C++ compiler frontend.

Built to understand what GPU driver stacks (NVIDIA CUDA, Vulkan compute, Qualcomm Adreno) do under the hood.

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

**Rust runtime** (`runtime/`) — library crate that talks to Vulkan via `ash`. Handles device setup, buffer management, shader pipeline creation, dispatch, and profiling.

**C++ compiler** (`compiler/`) — compiles GLSL (via `shaderc`) or a custom KernelScript language to SPIR-V. Includes hand-written optimization passes (constant folding, loop unrolling).

**Kernels** (`kernels/`) — benchmark compute shaders: prefix sum, histogram, sparse matrix-vector multiply.

## Build

### Prerequisites

- Rust 1.75+ (`rustup`)
- Vulkan SDK 1.3+ (from [LunarG](https://vulkan.lunarg.com/))
- A GPU with Vulkan support

Verify your setup:
```sh
vulkaninfo   # should show your GPU
cargo build --manifest-path runtime/Cargo.toml
```

### Build & test

```sh
# Build the runtime library
cargo build --manifest-path runtime/Cargo.toml

# Run integration tests (once implemented)
cargo test --manifest-path runtime/Cargo.toml
```

## Design Decisions

- **Vulkan over CUDA** — vendor-agnostic. Mirrors how Qualcomm/Intel/AMD compute teams target multiple GPU architectures.
- **Ash over Vulkano** — ash is a thin wrapper over the raw Vulkan API. More boilerplate but more control and closer to the C API that job postings ask about.
- **Explicit destroy over Drop** — Vulkan requires reverse-order teardown (Allocator → Device → Instance). Using `ManuallyDrop` + explicit `destroy_*` methods avoids lifetime complexity while learning.
- **Host-visible memory first** — Phase 1 uses `CpuToGpu`/`GpuToCpu` memory for simplicity. Staging buffers for `DeviceLocal` memory come later for performance.
- **`extern "C"` FFI bridge** — C++ compiler compiled via `cc`/`cmake-rs` in the build script, exposed to Rust via a simple C ABI. Avoids C++/Rust binding complexity.

## License

MIT

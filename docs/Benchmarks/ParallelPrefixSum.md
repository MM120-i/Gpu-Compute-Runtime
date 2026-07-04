### Parallel Prefix Sum (Scan)

**Algorithm**: 3 pass parallel prefix sum (Blelloch/Harris). Each pass is a separate compute shader dispatch.

**Pass 1**: Workgroup local scan: each 256 thread workgroup loads a chunk of the input, performs a local prefix sum, and writes both the per element result and a per workgroup partial sum.

**Pass 2**: Scan the partial sums: a single workgroup scans the array of partial sums produced by pass 1. For 1M elements this is ~4,096 partials, small enough for one workgroup.

**Pass 3**: Add carry: each workgroup takes the scanned partial from pass 2 and adds it to every element computed in pass 1, producing the final global prefix sum.

**Shader variants**:

- **Warp-shuffle** (`scan_pass1_warp.comp`, `scan_pass2_warp.comp`): Uses `subgroupInclusiveAdd()` instead of shared-memory stride trees. Two passes instead of three (the carry add pass is unchanged). Activated automatically when `subgroup_arithmetic` is detected.
- **Shared-memory** (`scan_pass1.comp`, `scan_pass2.comp`): Original 3 pass implementation with explicit shared memory stride tree and barriers. Fallback for GPUs without subgroup arithmetic.

**Results** (NVIDIA GeForce RTX 2060 SUPER):

| Version                       | GPU wall (ms) | GPU pure (ms) | Invocations | CPU (ms) | Speedup |
| ----------------------------- | ------------- | ------------- | ----------- | -------- | ------- |
| v1 (naive, recreate per call) | 182.80        | —             | —           | 7.26     | 0.04×   |
| v2 (pipeline cached)          | 1.21          | 0.84          | —           | 7.88     | 6.51×   |
| v2 (warp-shuffle + profiled)  | 1.30          | 0.87          | 2,101,248   | 7.69     | 5.89×   |

**Why the v1→v2 jump**: v1 created 3 pipelines + 3 dispatchers + 3 buffers inside the timed loop on every call. Actual compute time is sub millisecond; the 182 ms was entirely API overhead. v2 pre creates everything the timed loop just dispatches.

**Reference**: [GPU Gems 3, Chapter 39 — Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)

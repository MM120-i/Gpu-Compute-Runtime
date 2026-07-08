### Histogram

**Algorithm**: Single pass, two level PRSG (per workgroup shared memory reduction to global) histogram. Each value maps to bucket `val % 256`.

**Shader** (`histogram.glsl`): Each thread picks one element from the input, computes its bucket, and atomically increments `shared uint smem[bucket]` in workgroup local shared memory. After a workgroup wide barrier, the first 256 threads each write their shared memory slot to the global histogram buffer via another atomic add. Shared memory atomics are fast because they stay on chip; only the final flush hits global memory.

**Why this works**: With 256 buckets and `local_size_x=256`, each workgroup naturally covers all buckets. Every thread zeroes its own slot at the start of the dispatch so stale data from a prior dispatch is overwritten before the atomic phase begins.

**Results** (NVIDIA GeForce RTX 2060 SUPER):

| Version                       | GPU wall (ms) | GPU pure (ms) | Invocations | CPU (ms) | Speedup |
| ----------------------------- | ------------- | ------------- | ----------- | -------- | ------- |
| v1 (naive, recreate per call) | 59.06         | —             | —           | 11.86    | 0.20×   |
| v2 (pipeline cached)          | 0.65          | 0.47          | —           | 12.43    | 19.12×  |
| v2 (profiled)                 | 0.70          | 0.56          | 1,048,576   | 12.15    | 17.27×  |

**Bandwidth**: 5.97 GB/s measured, 1.77% of 448 GB/s peak. Histogram is latency-bound (atomic contention), not bandwidth bound.

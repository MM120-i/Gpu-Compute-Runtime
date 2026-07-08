### SpMV (Sparse Matrix-Vector Multiply)

**Algorithm**: CSR-format sparse matrix-vector multiply. Given a sparse matrix A (CSR: row pointers, column indices, values) and dense vector x, compute y = A · x where `y[row] = sum(A[row][col] * x[col])`.

**Matrix**: 262,144 × 262,144 with ~4.2M non-zeros. Randomly generated with 8–24 non-zeros per row (uniform distribution). Average 16 non-zeros per row.

**Shader variants**:

- **Naive** (`spmv.glsl`) — 1 thread per CSR row. Each thread iterates over its row's non-zero range (`row_ptrs[row]..row_ptrs[row+1]`), loads the column index and value, fetches `x[col]`, multiplies, and accumulates. Simple, but poor memory coalescing when non-zeros in a warp map to scattered columns.

- **Tiled / warp-per-row** (`spmv_tiled.glsl`) — 1 subgroup (warp, 32 threads on RTX 2060 SUPER) per row. Threads within the warp cooperatively load non-zeros and the corresponding x values, then `subgroupAdd` reduces the partial products. Workgroup count adapts: `ceil(rows / (WG_SIZE/subgroup_size))`. Activated automatically when `subgroup_arithmetic` is detected.

**Results** (NVIDIA GeForce RTX 2060 SUPER):

| Version | GPU wall (ms) | GPU pure (ms) | Invocations | CPU (ms) | Speedup |
|---------|-------------|-------------|-----------|----------|---------|
| v1 (naive, recreate per call) | 65.77 | — | — | 94.05 | 1.43× |
| v2 (pipeline cached, naive shader) | 0.34 | — | — | 97.19 | 285× |
| v2 (tiled/shuffle + profiled) | 3.19 | 3.03 | 8,388,608 | 94.78 | 29.67× |

**Why tiled is slower than naive**: The tiled variant dispatches 32× more workgroups (32,768 vs 1,024) because each workgroup handles only `WG_SIZE/subgroup_size = 8` rows instead of 256. With the sparse matrix having only ~16 non-zeros per row on average, the warp-level reduction overhead dominates the benefit of better memory coalescing. For denser matrices (> 64 non-zeros per row), the tiled variant would pull ahead.

**Bandwidth**: 11.15 GB/s measured, 2.34% of 448 GB/s peak. SpMV is memory-bound — the bottleneck is fetching the column vector x at scattered indices, not the arithmetic.

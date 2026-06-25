### SpMV (Sparse Matrix-Vector Multiply)

**CSR format, one thread per row**: each thread reads its row's non-zero range from `row_ptrs`, multiplies each value by `x[col]`, and writes the sum to `y[row]`.

```
{
  "spmv": {
    "bandwidth_gbps": 0.54,
    "cols": 262144,
    "correct": true,
    "cpu_ms": 94.05,
    "device": "NVIDIA GeForce RTX 2060 SUPER",
    "gpu_ms": 65.77,
    "nnz": 4188912,
    "rows": 262144,
    "speedup": 1.43,
    "workgroup_size": 256
  }
}
```

**First benchmark where GPU wins** (1.43× speedup). SpMV is memory bound. The GPU's 256 bit memory bus and high bandwidth dominate the CPU's DDR5, even with pipeline recreation overhead. 262K × 262K matrix with ~4M non-zeros (avg 16 per row, randomly distributed).

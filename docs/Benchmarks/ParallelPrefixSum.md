### Parallel Prefix Sum (Scan)

**3-pass naive scan**: workgroup local scan → scan partial sums → add carry.
Each pass recompiles the shader pipelines are not cached yet :(

```
{
  "scan": {
    "bandwidth_gbps": 0.05,
    "correct": true,
    "cpu_ms": 7.26,
    "device": "NVIDIA GeForce RTX 2060 SUPER",
    "elements": 1048576,
    "gpu_ms": 182.8,
    "speedup": 0.04,
    "workgroup_size": 256
  }
}
```

GPU is 25× slower than CPU because `run_scan_gpu` creates 3 pipelines + 3 dispatchers + 3 buffers on every call (inside the timed loop). Actual compute time is sub-millisecond.

<img width="876" height="449" alt="image" src="https://github.com/user-attachments/assets/1921428a-fef9-42a7-b6ee-0003967c6868" />

https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

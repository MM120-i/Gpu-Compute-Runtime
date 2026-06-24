### Parallel Prefix Sum (Scan)

Current results (naive implementations), no shared memory optimizations, no Vulkan timestamp queries (host-timed). We'll optimize each kernel later at v2 :(

```
{
  "scan": {
    "bandwidth_gbps": 0.05,
    "correct": true,
    "cpu_ms": 7.67,
    "device": "NVIDIA GeForce RTX 2060 SUPER",
    "elements": 1048576,
    "gpu_ms": 182.57,
    "speedup": 0.04,
    "workgroup_size": 256
  }
}
```

**3-pass naive scan**: workgroup-local scan → scan partial sums → add carry

<img width="876" height="449" alt="image" src="https://github.com/user-attachments/assets/1921428a-fef9-42a7-b6ee-0003967c6868" />

https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

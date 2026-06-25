### Histogram

**Single-pass, two-level atomics**: each workgroup builds a local histogram in shared memory (`atomicAdd`), then flattens to global memory.

```
{
  "histogram": {
    "bandwidth_gbps": 0.07,
    "buckets": 256,
    "correct": true,
    "cpu_ms": 11.86,
    "device": "NVIDIA GeForce RTX 2060 SUPER",
    "elements": 1048576,
    "gpu_ms": 59.06,
    "range": 1024,
    "speedup": 0.2,
    "workgroup_size": 256
  }
}
```

Uses `shared uint smem[256]`, each thread zeros its slot, atomically increments the right bucket, then writes its slot to global. Shared memory atomics are fast; the bottleneck is pipeline recreation.

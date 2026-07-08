## Mandelbrot Set Renderer

A GPU accelerated Mandelbrot fractal renderer built as a demo for the GPU Compute Runtime. Renders the Mandelbrot set at arbitrary resolution using a Vulkan compute shader, then writes the output as a binary PPM image.

### How It Works

The GPU dispatches a 2D grid of workgroups (16×16 threads each). Every thread computes one pixel mapping its (x, y) position to a complex plane coordinate, then running the escape-time algorithm:

```
z₀ = 0
zₙ₊₁ = zₙ² + c
```

If `|z|` exceeds 2.0 before `max_iters` iterations, the pixel escapes and is colored by escape speed. If it never escapes, the pixel is black (in the Mandelbrot set).

### Architecture

```
CPU (Rust)                              GPU (GLSL)
───────────                             ─────────
render_gpu()                              mandelbrot.glsl
  │                                       16×16 workgroups
  ├─ upload MandelbrotParams ──────────→  binding 1 (readonly SSBO)
  │    width, height, max_iters
  │    cx, cy, scale
  │
  ├─ create output buffer ←──────────── binding 0 (SSBO)
  │    width × height u32 pixels
  │
  ├─ create pipeline + descriptor set
  ├─ dispatch 2D workgroups
  ├─ download results
  │
  └─ write_ppm() → binary P6 PPM file
```

### Running

```bash
# Render 1080p full set view (1000 iterations)
cargo test mandelbrot_render_full_1080p -- --nocapture

# Or u can do it with make
make mandelbrot
```

Output: `mandelbrot.ppm` 1920×1080 binary PPM. Open with any image viewer (Windows Photos, GIMP, etc.).

### Parameters

| Parameter   | Controls                                                                                 |
| ----------- | ---------------------------------------------------------------------------------------- |
| `cx`, `cy`  | Center of the viewport on the complex plane                                              |
| `scale`     | Zoom level smaller = deeper zoom                                                         |
| `max_iters` | Maximum iterations before declaring a pixel "in set" — higher = finer detail at boundary |

### Viewpoint Presets _(Phase 2)_

| Name       | cx           | cy          | scale     | What You See                        |
| ---------- | ------------ | ----------- | --------- | ----------------------------------- |
| `full`     | -0.5         | 0.0         | 3.5       | The entire Mandelbrot set           |
| `seahorse` | -0.743643887 | 0.131825904 | 0.0000001 | Deep zoom — seahorse valley spirals |
| `elephant` | 0.275        | 0.006       | 0.00001   | Elephant valley curlicues           |
| `spiral`   | -0.7269      | 0.1889      | 0.0002    | Spiral galaxy-like patterns         |

### Full Set Render (1920×1080, 1000 iterations)

![Mandelbrot Full Set](/docs/WebPage/chart.js/public/mandelbrot.jpg)

### Performance

On an NVIDIA GeForce RTX 2060 SUPER, a 1024×1024 render at 200 iterations completes in ~2ms on GPU vs ~80ms on CPU approximately 40× speedup. The shader is purely compute-bound (minimal memory traffic just one u32 write per pixel).

### Files

| File                               | Purpose                                      |
| ---------------------------------- | -------------------------------------------- |
| `kernels/demos/mandelbrot.glsl`    | 2D compute shader with escape-time algorithm |
| `runtime/src/demos/mandelbrot.rs`  | Dispatch, buffer management, PPM output      |
| `runtime/tests/mandelbrot_test.rs` | Correctness tests + render-to-disk test      |

### References

- **Wikipedia**: [Mandelbrot Set](https://en.wikipedia.org/wiki/Mandelbrot_set) — Comprehensive reference on the set's properties, coloring algorithms, and zoom sequences. Includes the coordinate locations of famous deep-zoom regions (Seahorse Valley, Elephant Valley).
- **Paul Bourke**: [Mandelbrot Set Rendering](https://paulbourke.net/fractals/mandelbrot/) — Practical rendering techniques, smooth coloring algorithms, and coordinate reference points for scenic viewpoints.

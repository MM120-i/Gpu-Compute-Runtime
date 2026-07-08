#version 450

#extension GL_ARB_gpu_shader_fp64 : enable

layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) buffer Output {
    uint pixels[];
};

layout(std430, binding = 1) readonly buffer Params {
    uint width;
    uint height;
    uint max_iters;
    uint _pad;
    double cx;
    double cy;
    double scale;
};

void main() {
    uvec2 id = gl_GlobalInvocationID.xy;

    if (id.x >= width || id.y >= height)
        return;

    double aspect = double(width) / double(height);
    double x0 = (double(id.x) / double(width) - 0.5) * scale * aspect + cx;
    double y0 = (double(id.y) / double(height) - 0.5) * scale + cy;

    double x = 0.0;
    double y = 0.0;
    uint iter = 0u;

    while (iter < max_iters && (x * x + y * y) <= 4.0) {
        double xtemp = x * x - y * y + x0;
        y = 2.0 * x * y + y0;
        x = xtemp;
        iter++;
    }

    pixels[id.y * width + id.x] = iter;
}

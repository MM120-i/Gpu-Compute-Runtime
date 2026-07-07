#version 450

layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) buffer Output {
    uint pixels[];
};

layout(std430, binding = 1) readonly buffer Params {
    uint width;
    uint height;
    uint max_iters;
    float cx;
    float cy;
    float scale;
};

void main() {
    uvec2 id = gl_GlobalInvocationID.xy;

    if (id.x >= width || id.y >= height) 
        return;

    float aspect = float(width) / float(height);
    float x0 = (float(id.x) / float(width) - 0.5) * scale * aspect + cx;
    float y0 = (float(id.y) / float(height) - 0.5) * scale + cy;

    float x = 0.0;
    float y = 0.0;
    uint iter = 0u;

    while (iter < max_iters && (x * x + y * y) <= 4.0) {
        float xtemp = x * x - y * y + x0;
        y = 2.0 * x * y + y0;
        x = xtemp;
        iter++;
    }

    pixels[id.y * width + id.x] = iter;
}
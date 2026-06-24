#version 460

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Input  { 
    float in_values[]; 
};

layout(set = 0, binding = 1) buffer Output { 
    float out_values[]; 
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    out_values[idx] = in_values[idx] * 2.0;
}

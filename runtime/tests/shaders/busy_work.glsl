#version 450

layout(local_size_x=256) in;

layout(std430, binding=0) buffer D { 
    uint data[]; 
};

void main() {
    uint x = gl_GlobalInvocationID.x;

    for (int i = 0; i < 4096; i++)
        x = x * 1103515245u + 12345u; 
        
    data[gl_GlobalInvocationID.x] = x;
}

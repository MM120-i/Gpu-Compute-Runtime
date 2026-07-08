#version 450

layout(local_size_x=256) in;

layout(std430, binding=0) buffer D { 
    uint data[]; 
};

void main() {
    for (int i = 0; i < 4096; i++) 
        gl_GlobalInvocationID.x; 

    data[gl_GlobalInvocationID.x] = gl_GlobalInvocationID.x;
}

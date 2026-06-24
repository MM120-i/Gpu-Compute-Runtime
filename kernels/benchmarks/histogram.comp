#version 460

layout(local_size_x = 256) in;

layout(binding = 0) buffer Input {
    uint data[];
};

layout(binding = 1) buffer Output {
    uint hist[];
};

shared uint smem[256];

void main(){
    uint tid = gl_LocalInvocationIndex;
    uint gid = gl_GlobalInvocationID.x;

    smem[tid] = 0u;
    barrier();

    if(gid < data.length()){
        uint bucket = data[gid] % 256u;
        atomicAdd(smem[bucket], 1u);
    }

    barrier();
    atomicAdd(hist[tid], smem[tid]);
}

#version 460

layout(local_size_x = 256) in;

layout(binding = 0) buffer Input  { 
    uint in_data[]; 
};

layout(binding = 1) buffer Output { 
    uint out_data[]; 
};

layout(binding = 2) buffer Partial { 
    uint partial_sums[]; 
};

shared uint temp[256];

void main(){
    uint tid = gl_LocalInvocationIndex;
    uint gid = gl_GlobalInvocationID.x;

    temp[tid] = in_data[gid];
    barrier();

    for(uint stride = 1u; stride < 256u; stride <<= 1u){
        uint val = 0u;

        if(tid >= stride)
            val = temp[tid - stride];
        
        barrier();
        temp[tid] += val;
        barrier();
    }

    out_data[gid] = temp[tid];

    if(tid == 255u)
        partial_sums[gl_WorkGroupID.x] = temp[255];
    
}
#version 460
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256) in;
layout(binding = 0) buffer Partial { uint partial_sums[]; };

#define ITEMS_PER_THREAD 16

shared uint warp_scan[32];

void main() {
    uint tid = gl_LocalInvocationIndex;
    uint n = partial_sums.length();
    uint start = tid * ITEMS_PER_THREAD;
    uint sg_size = gl_SubgroupSize;
    uint sg_idx  = tid / sg_size;
    uint lane    = gl_SubgroupInvocationID;
    uint num_wgs = gl_WorkGroupSize.x / sg_size;
    uint local[ITEMS_PER_THREAD];
    uint running = 0u;

    for (int j = 0; j < ITEMS_PER_THREAD; j++) {
        uint index = start + j;

        if (index < n) {
            running += partial_sums[index];
            local[j] = running;
        }
    }

    uint scanned = subgroupInclusiveAdd(running);

    if (lane == sg_size - 1u)
        warp_scan[sg_idx] = scanned;

    barrier();

    if (sg_idx == 0u) {
        uint w = (lane < num_wgs) ? warp_scan[lane] : 0u;
        uint ws = subgroupInclusiveAdd(w);
        if (lane < num_wgs) warp_scan[lane] = ws;
    }

    barrier();

    uint carry_warps = (sg_idx > 0u) ? warp_scan[sg_idx - 1u] : 0u;
    uint carry_lanes = scanned - running;
    uint carry = carry_warps + carry_lanes;

    for (int j = 0; j < ITEMS_PER_THREAD; j++) {
        uint index = start + j;
        
        if (index < n)
            partial_sums[index] = local[j] + carry;
    }
}

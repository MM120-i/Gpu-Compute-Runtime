#version 460
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256) in;

layout(binding = 0) buffer Input  { uint in_data[]; };
layout(binding = 1) buffer Output { uint out_data[]; };
layout(binding = 2) buffer Partial { uint partial_sums[]; };

shared uint warp_scan[32];

void main() {
    uint tid = gl_LocalInvocationIndex;
    uint gid = gl_GlobalInvocationID.x;

    uint sg_size = gl_SubgroupSize;
    uint sg_idx  = tid / sg_size;
    uint lane    = gl_SubgroupInvocationID;
    uint num_wgs = gl_WorkGroupSize.x / sg_size;
    uint val = in_data[gid];
    uint scanned = subgroupInclusiveAdd(val);

    if (lane == sg_size - 1u)
        warp_scan[sg_idx] = scanned;

    barrier();

    if (sg_idx == 0u) {
        uint w = (lane < num_wgs) ? warp_scan[lane] : 0u;
        uint ws = subgroupInclusiveAdd(w);
        if (lane < num_wgs) warp_scan[lane] = ws;
    }

    barrier();

    uint offset = (sg_idx > 0u) ? warp_scan[sg_idx - 1u] : 0u;
    out_data[gid] = scanned + offset;

    if (tid == 255u)
        partial_sums[gl_WorkGroupID.x] = out_data[gid];
}

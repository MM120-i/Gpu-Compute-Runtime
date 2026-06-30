#version 460
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256) in;

layout(binding = 0) buffer RowPtrs { 
    uint row_ptrs[]; 
};

layout(binding = 1) buffer ColIndices { 
    uint col_indices[]; 
};

layout(binding = 2) buffer Values { 
    float values[]; 
};

layout(binding = 3) buffer X { 
    float x[]; 
};

layout(binding = 4) buffer Y { 
    float y[]; 
};

void main() {
    uint sg_idx = gl_LocalInvocationIndex / gl_SubgroupSize;
    uint num_sgs = gl_WorkGroupSize.x / gl_SubgroupSize;
    uint row = gl_WorkGroupID.x * num_sgs + sg_idx;
    uint n_rows = row_ptrs.length() - 1u;

    if (row >= n_rows) 
        return;

    uint start = row_ptrs[row];
    uint end = row_ptrs[row + 1];
    uint nnz = end - start;
    float sum = 0.0;

    for (uint i = gl_SubgroupInvocationID; i < nnz; i += gl_SubgroupSize)
        sum += values[start + i] * x[col_indices[start + i]];

    float total = subgroupAdd(sum);

    if (gl_SubgroupInvocationID == 0u)
        y[row] = total;
}
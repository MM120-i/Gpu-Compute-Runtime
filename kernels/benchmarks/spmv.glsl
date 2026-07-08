#version 460

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

void main(){
    uint row = gl_GlobalInvocationID.x;   
    uint n_rows = row_ptrs.length() - 1u;

    if(row >= n_rows)
        return;

    uint start = row_ptrs[row];
    uint end = row_ptrs[row + 1];
    float sum = 0.0;

    for(uint i = start; i < end; i++)
        sum += values[i] * x[col_indices[i]];

    y[row] = sum;
}
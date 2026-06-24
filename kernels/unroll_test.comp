#version 460

layout(local_size_x = 64) in;

layout(binding = 0) buffer Out {
    float data[];
}

out_buf;

void main(){
    uint idx = gl_GlobalInvocationID.x;
    
    float sum = 0.0;
    for(int i = 0; i < 8; i++)
        sum += float(i) * 0.5;

    float prod = 1.0;
    for(int j = 1; j <= 4; j++)
        prod *= float(j);

    for(int k = 0; k < idx; k++)
        sum += 1.0;

    out_buf.data[idx] = sum + prod;
}
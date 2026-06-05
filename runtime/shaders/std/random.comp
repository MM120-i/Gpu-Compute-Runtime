#ifndef RANDOM_GLSL
#define RANDOM_GLSL

float rand_hash(uint seed) {
    // Wang hash
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);

    return float(seed) / 4294967296.0;
}

float rand_hash(float min_val, float max_val, uint seed){
    return min_val + rand_hash(seed) * (max_val - min_val);
}

#endif
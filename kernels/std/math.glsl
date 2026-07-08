#ifndef MATH_GLSL
#define MATH_GLSL

#define PI 3.14159265359

float saturate(float x){
    return clamp(x, 0.0, 1.0);
}

float lerp(float a, float b, float t){
    return a + (b - a) * t;
}

float smoothstep_val(float edge0, float edge1, float x){
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

#endif
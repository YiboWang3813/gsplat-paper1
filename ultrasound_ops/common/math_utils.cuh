#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>

__device__ __forceinline__ float dot_prod(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// =========================================================
// Global Operator Overloads (为了写代码爽)
// =========================================================

__device__ __forceinline__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

// =========================================================
// Pure Mathematical Primitives (Linear Algebra)
// =========================================================

__device__ __forceinline__ void vec3_copy(const float* A, float* B) {
    #pragma unroll
    for (int i = 0; i < 3; i ++) B[i] = A[i];  
}

__device__ __forceinline__ void vec4_copy(const float* A, float* B) {
    #pragma unroll
    for (int i = 0; i < 4; i ++) B[i] = A[i];  
}

__device__ __forceinline__ void mat3_copy(const float* A, float* B) {
    #pragma unroll
    for (int i = 0; i < 9; i ++) B[i] = A[i]; 
}

__device__ __forceinline__ void vec3_zero(float* A) {
    #pragma unroll
    for (int i = 0; i < 3; i ++) A[i] = 0.0f; 
}

__device__ __forceinline__ void mat3_zero(float* A) {
    #pragma unroll
    for (int i = 0; i < 9; i ++) A[i] = 0.0f; 
}

__device__ __forceinline__ void mat3_mul(const float* A, const float* B, float* C) {
    #pragma unroll
    for (int i = 0; i < 3; i ++) {
        float ai0 = A[i * 3 + 0], ai1 = A[i * 3 + 1], ai2 = A[i * 3 + 2]; 
        C[i * 3 + 0] = ai0 * B[0] + ai1 * B[3] + ai2 * B[6]; 
        C[i * 3 + 1] = ai0 * B[1] + ai1 * B[4] + ai2 * B[7]; 
        C[i * 3 + 2] = ai0 * B[2] + ai1 * B[5] + ai2 * B[8]; 
    }
}

__device__ __forceinline__ void mat3_scale(const float* A, const float s, float* B) {
    #pragma unroll
    for (int i = 0; i < 9; i ++) B[i] = s * A[i]; 
}

__device__ __forceinline__ void mat3_transpose(const float* A, float* AT) {
    AT[0] = A[0]; AT[1] = A[3]; AT[2] = A[6]; 
    AT[3] = A[1]; AT[4] = A[4]; AT[5] = A[7]; 
    AT[6] = A[2]; AT[7] = A[5]; AT[8] = A[8]; 
}

__device__ __forceinline__ void mat3_symmetrize(const float* A, float* S) {
    float AT[9]; 
    mat3_transpose(A, AT); 
    #pragma unroll
    for (int i = 0; i < 9; i ++) S[i] = 0.5f * (A[i] + AT[i]); 
}

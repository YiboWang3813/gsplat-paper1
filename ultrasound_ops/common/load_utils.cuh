#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>

__device__ __forceinline__ float3 unpack_float4(const float4 v) {
    return make_float3(v.x, v.y, v.z);
}

__device__ __forceinline__ float3 xyz(const float4& v) {
    return make_float3(v.x, v.y, v.z);
}
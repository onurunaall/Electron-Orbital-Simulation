#pragma once

#include <cuda_runtime.h>
#include <cmath>


constexpr float kPi{3.14159265358979323846f};


[[nodiscard]] __host__ __device__ inline float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}


[[nodiscard]] __host__ __device__ inline float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}


[[nodiscard]] __host__ __device__ inline float3 operator*(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}


[[nodiscard]] __host__ __device__ inline float3 operator*(float3 vector, float scalar) {
    return make_float3(vector.x * scalar, vector.y * scalar, vector.z * scalar);
}


[[nodiscard]] __host__ __device__ inline float3 operator*(float scalar, float3 vector) {
    return vector * scalar;
}


[[nodiscard]] __host__ __device__ inline float3 operator/(float3 vector, float scalar) {
    return make_float3(vector.x / scalar, vector.y / scalar, vector.z / scalar);
}


[[nodiscard]] __host__ __device__ inline float dot(float3 a, float3 b) {
    return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

[[nodiscard]] __host__ __device__ inline float3 cross(float3 a, float3 b) {
    const float x{(a.y * b.z) - (a.z * b.y)};
    const float y{(a.z * b.x) - (a.x * b.z)};
    const float z{(a.x * b.y) - (a.y * b.x)};
    return make_float3(x, y, z);
}

[[nodiscard]] __host__ __device__ inline float length(float3 vector) {
    return sqrtf(dot(vector, vector));
}


[[nodiscard]] __host__ __device__ inline float3 normalize(float3 vector) {
    return vector / length(vector);
}


[[nodiscard]] __host__ __device__ inline float3 vmin(float3 a, float3 b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}


// Component-wise maximum of two vectors
[[nodiscard]] __host__ __device__ inline float3 vmax(float3 a, float3 b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}


// Clamps a scalar float to the range [minimum, maximum]
[[nodiscard]] __host__ __device__ inline float clampf(float value, float minimum, float maximum) {
    return fminf(fmaxf(value, minimum), maximum);
}


// Converts linear RGB color channels in [0.0, 1.0] to an 8-bit uchar3 [0, 255]
[[nodiscard]] __host__ __device__ inline uchar3 toRgb8(float3 linearColor) {
    const float clampedRed{clampf(linearColor.x, 0.0f, 1.0f)};
    const float clampedGreen{clampf(linearColor.y, 0.0f, 1.0f)};
    const float clampedBlue{clampf(linearColor.z, 0.0f, 1.0f)};

    const auto byteRed{static_cast<unsigned char>(clampedRed * 255.0f + 0.5f)};
    const auto byteGreen{static_cast<unsigned char>(clampedGreen * 255.0f + 0.5f)};
    const auto byteBlue{static_cast<unsigned char>(clampedBlue * 255.0f + 0.5f)};

    return make_uchar3(byteRed, byteGreen, byteBlue);
}


// Converts spherical coordinates to Cartesian coordinates with the polar axis along +y
[[nodiscard]] __host__ __device__ inline float3 sphericalToCartesian(float radius, float theta, float phi) {
    const float sinTheta{sinf(theta)};
    const float cosTheta{cosf(theta)};
    const float sinPhi{sinf(phi)};
    const float cosPhi{cosf(phi)};

    const float x{radius * sinTheta * cosPhi};
    const float y{radius * cosTheta};
    const float z{radius * sinTheta * sinPhi};

    return make_float3(x, y, z);
}

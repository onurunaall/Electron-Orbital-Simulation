#pragma once

#include "vec_math.cuh"

// Computes 1.0f / component. Replaces zero with a large finite constant to avoid NaNs
[[nodiscard]] __host__ __device__ inline float safeReciprocal(float component) {
    constexpr float fallbackValue = 1e30f;
    if (component == 0.0f) {
        return fallbackValue;
    }
    return 1.0f / component;
}


// Inverts ray direction per axis for slab-based box intersection tests
[[nodiscard]] __host__ __device__ inline float3 safeInverse(float3 rayDirection) {
    float3 inverseDirection;
    inverseDirection.x = safeReciprocal(rayDirection.x);
    inverseDirection.y = safeReciprocal(rayDirection.y);
    inverseDirection.z = safeReciprocal(rayDirection.z);
    return inverseDirection;
}


// Computes the distance along a normalized ray to the nearest sphere hit.
// Returns -1.0f if the ray misses the sphere.
[[nodiscard]] __host__ __device__ inline float intersectSphere(
    float3 rayOrigin,
    float3 rayDirection,
    float3 sphereCenter,
    float sphereRadius) 
{
    const float3 rayToCenter = rayOrigin - sphereCenter;
    
    // Ray direction is unit length, so a = dot(rayDirection, rayDirection) = 1.0
    const float projectionLength = dot(rayToCenter, rayDirection);
    const float centerDistanceSq = dot(rayToCenter, rayToCenter) - (sphereRadius * sphereRadius);
    
    const float discriminant = (projectionLength * projectionLength) - centerDistanceSq;
    if (discriminant < 0.0f) {
        return -1.0f;
    }

    return -projectionLength - sqrtf(discriminant);
}


// Slab test: checks if the ray intersects an axis-aligned box within (0, maxDistance]
[[nodiscard]] __host__ __device__ inline bool rayHitsBox(
    float3 rayOrigin,
    float3 inverseDirection,
    float3 boxMin,
    float3 boxMax,
    float maxDistance) 
{
    const float3 rayToMinPlanes = (boxMin - rayOrigin) * inverseDirection;
    const float3 rayToMaxPlanes = (boxMax - rayOrigin) * inverseDirection;

    const float3 axisIntervalMin = vmin(rayToMinPlanes, rayToMaxPlanes);
    const float3 axisIntervalMax = vmax(rayToMinPlanes, rayToMaxPlanes);

    const float entryDistance = fmaxf(fmaxf(axisIntervalMin.x, axisIntervalMin.y), fmaxf(axisIntervalMin.z, 0.0f));
    const float exitDistance  = fminf(fminf(axisIntervalMax.x, axisIntervalMax.y), fminf(axisIntervalMax.z, maxDistance));

    return entryDistance <= exitDistance;
}

#pragma once

#include "vec_math.cuh"


// Pinhole camera orbiting the origin (0, 0, 0).
// Camera space is right-handed: +x points right, +y points up, and +z points forward (depth).
struct Camera {
    float3 eye{};
    float3 forward{};
    float3 right{};
    float3 up{};

    float tanHalfFov{0.0f};
    float aspect{1.0f};
    float nearPlane{0.1f};

    int width{0};
    int height{0};

    // Builds an orbit camera positioned on a sphere centered at the origin.
    // Elevation is the polar angle measured from the +y axis in radians.
    [[nodiscard]] static Camera orbit(
        float radius,
        float azimuth,
        float elevation,
        float fovDegrees,
        float nearPlaneDistance,
        int imageWidth,
        int imageHeight) 
    {
        // Clamp elevation away from poles to avoid gimbal lock with the world up vector
        constexpr float poleEpsilon{0.01f};
        const float clampedElevation{clampf(elevation, poleEpsilon, kPi - poleEpsilon)};

        const float sinElevation{sinf(clampedElevation)};
        const float cosElevation{cosf(clampedElevation)};

        Camera camera{};
        camera.eye = make_float3(
            radius * sinElevation * cosf(azimuth),
            radius * cosElevation,
            radius * sinElevation * sinf(azimuth)
        );

        // Target the origin
        const float3 worldOrigin{0.0f, 0.0f, 0.0f};
        camera.forward = normalize(worldOrigin - camera.eye);

        // Build the orthonormal basis using world +y as reference
        const float3 worldUpReference{0.0f, 1.0f, 0.0f};
        camera.right = normalize(cross(camera.forward, worldUpReference));
        camera.up = cross(camera.right, camera.forward);

        constexpr float degToRad{kPi / 180.0f};
        const float halfFovRadians{0.5f * fovDegrees * degToRad};
        camera.tanHalfFov = tanf(halfFovRadians);

        camera.aspect = static_cast<float>(imageWidth) / static_cast<float>(imageHeight);
        camera.nearPlane = nearPlaneDistance;
        camera.width = imageWidth;
        camera.height = imageHeight;

        return camera;
    }


    // Projects a world-space point into camera space
    [[nodiscard]] __host__ __device__ float3 toCameraSpace(float3 worldPoint) const {
        const float3 offsetFromEye{worldPoint - eye};

        const float cameraX{dot(offsetFromEye, right)};
        const float cameraY{dot(offsetFromEye, up)};
        const float cameraZ{dot(offsetFromEye, forward)};

        return make_float3(cameraX, cameraY, cameraZ);
    }


    // Projects a camera-space point (with z > 0) onto 2D viewport pixel coordinates
    [[nodiscard]] __host__ __device__ float2 toPixel(float3 cameraPoint) const {
        const float invDepth{1.0f / cameraPoint.z};

        const float ndcX{cameraPoint.x * invDepth / (tanHalfFov * aspect)};
        const float ndcY{cameraPoint.y * invDepth / tanHalfFov};

        // Map normalized device coordinates [-1, 1] to pixel coordinates [0, width] x [0, height]
        const float pixelX{(ndcX + 1.0f) * 0.5f * width};
        const float pixelY{(1.0f - ndcY) * 0.5f * height};

        return make_float2(pixelX, pixelY);
    }


    // Computes a normalized world-space ray direction through pixel coordinates (pixelX, pixelY)
    [[nodiscard]] __host__ __device__ float3 rayThroughPixel(float pixelX, float pixelY) const {
        const float ndcX{(2.0f * pixelX / width) - 1.0f};
        const float ndcY{1.0f - (2.0f * pixelY / height)};

        const float3 horizontalOffset{right * (ndcX * tanHalfFov * aspect)};
        const float3 verticalOffset{up * (ndcY * tanHalfFov)};

        return normalize(forward + horizontalOffset + verticalOffset);
    }
};

#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "common/camera.cuh"


// 3D line segment defined by two world-space endpoints
struct Segment {
    float3 startPoint{};
    float3 endPoint{};
};


// Projected 2D screen line segment with reciprocal depth endpoints for perspective-correct interpolation
struct ScreenSegment {
    float2 startPixel{};
    float2 endPixel{};
    float invDepthStart{0.0f};
    float invDepthEnd{0.0f};
};


// 2D inclusive pixel bounding box
struct PixelBox {
    int minX{0};
    int minY{0};
    int maxX{0};
    int maxY{0};
};


// Computes the screen-space bounding box enclosing a 3D sphere.
// Returns false if the sphere lies behind or crosses the camera near clip plane, or is completely off-screen.
[[nodiscard]] __host__ __device__ inline bool computeSphereScreenBounds(
    const Camera& camera,
    float3 center,
    float radius,
    PixelBox& outBoundingBox)
{
    float screenMinX{1e30f};
    float screenMinY{1e30f};
    float screenMaxX{-1e30f};
    float screenMaxY{-1e30f};

    // Project all 8 vertices of the sphere's bounding cube
    for (int cornerIndex{0}; cornerIndex < 8; ++cornerIndex) {
        float offsetX = (cornerIndex & 1) ? radius : -radius;
        float offsetY = (cornerIndex & 2) ? radius : -radius;
        float offsetZ = (cornerIndex & 4) ? radius : -radius;

        float3 worldCorner = center + make_float3(offsetX, offsetY, offsetZ);
        float3 cameraPoint = camera.toCameraSpace(worldCorner);

        // Near-plane clipping guard
        if (cameraPoint.z < camera.nearPlane) {
            return false;
        }

        float2 screenPixel = camera.toPixel(cameraPoint);

        screenMinX = fminf(screenMinX, screenPixel.x);
        screenMaxX = fmaxf(screenMaxX, screenPixel.x);
        screenMinY = fminf(screenMinY, screenPixel.y);
        screenMaxY = fmaxf(screenMaxY, screenPixel.y);
    }

    // Clamp bounding bounds to viewport dimensions
    outBoundingBox.minX = max(static_cast<int>(floorf(screenMinX)), 0);
    outBoundingBox.maxX = min(static_cast<int>(ceilf(screenMaxX)), camera.width - 1);
    outBoundingBox.minY = max(static_cast<int>(floorf(screenMinY)), 0);
    outBoundingBox.maxY = min(static_cast<int>(ceilf(screenMaxY)), camera.height - 1);

    return (outBoundingBox.minX <= outBoundingBox.maxX) && 
           (outBoundingBox.minY <= outBoundingBox.maxY);
}


class Rasterizer {
public:
    Rasterizer(int viewportWidth, int viewportHeight);
    ~Rasterizer();

    Rasterizer(const Rasterizer&) = delete;
    Rasterizer& operator=(const Rasterizer&) = delete;

    Rasterizer(Rasterizer&&) noexcept = default;
    Rasterizer& operator=(Rasterizer&&) noexcept = default;

    // Resets the depth/ID buffer
    void clear();

    // Draws projected reference lines
    void drawGrid(const std::vector<Segment>& segments, const Camera& camera);

    // Rasterizes spheres via ray tests inside screen-space bounding boxes
    void drawSpheres(
        const float3* deviceCenters,
        const int* deviceIndices,
        int activeCount,
        float radius,
        const Camera& camera
    );

    // Resolves the 64-bit depth/ID buffer into a final 24-bit RGB pixel buffer
    void resolve(
        const float3* deviceCenters,
        const float3* deviceColors,
        const Camera& camera,
        bool enableShading
    );

    // Copies the resolved RGB framebuffer from GPU to CPU memory
    void download(std::vector<uchar3>& outPixels) const;

    [[nodiscard]] int width() const { return width_; }
    [[nodiscard]] int height() const { return height_; }

private:
    int width_{0};
    int height_{0};

    // 64-bit atomic depth-ID buffer: high 32 bits = float depth bits, low 32 bits = primitive ID
    unsigned long long* deviceDepthId_{nullptr};
    uchar3* devicePixels_{nullptr};

    ScreenSegment* deviceSegments_{nullptr};
    int segmentCapacity_{0};
};

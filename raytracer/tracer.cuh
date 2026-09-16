#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "bvh.cuh"
#include "common/camera.cuh"

// Point light source with diffuse falloff and uniform ambient baseline
struct PointLight {
    float3 position{};
    float3 ambient{};
    float intensity{1.0f};
};

// Primary ray tracer evaluating ray-BVH intersections and point-light shading
class Tracer {
public:
    Tracer(int viewportWidth, int viewportHeight);
    ~Tracer();

    Tracer(const Tracer&) = delete;
    Tracer& operator=(const Tracer&) = delete;

    Tracer(Tracer&&) = delete;
    Tracer& operator=(Tracer&&) = delete;

    // Renders the scene into an internal RGB device framebuffer
    void render(const BvhView& bvh, const Camera& camera, const PointLight& light);

    // Downloads rendered pixels from device memory to the host
    void download(std::vector<uchar3>& outPixels) const;

    [[nodiscard]] int width() const { return width_; }
    [[nodiscard]] int height() const { return height_; }

private:
    int width_{0};
    int height_{0};
    uchar3* devicePixels_{nullptr};
};

#include "tracer.cuh"

#include "common/cuda_check.cuh"
#include "common/vec_math.cuh"

static const dim3 PIXEL_TILE_DIMENSIONS(16, 16);

static inline dim3 getPixelGrid(int width, int height) {
    return dim3(
        (width  + PIXEL_TILE_DIMENSIONS.x - 1) / PIXEL_TILE_DIMENSIONS.x,
        (height + PIXEL_TILE_DIMENSIONS.y - 1) / PIXEL_TILE_DIMENSIONS.y
    );
}

// Ray tracing kernel: casts one primary ray per pixel and a shadow ray toward the light source
__global__ void rayTraceKernel(
    BvhView bvh,
    Camera camera,
    PointLight light,
    uchar3* outPixels)
{
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelX >= camera.width || pixelY >= camera.height) {
        return;
    }

    // Generate normalized camera ray through pixel center
    float3 rayDirection = camera.rayThroughPixel(pixelX + 0.5f, pixelY + 0.5f);

    constexpr float maxTraceDistance = 1e20f;
    RayHit primaryHit = bvhClosestHit(bvh, camera.eye, rayDirection, maxTraceDistance);

    float3 pixelColor = make_float3(0.0f, 0.0f, 0.0f); // Default background

    if (primaryHit.leafIndex >= 0) {
        // Reconstruct surface hit point and normal
        float3 hitPoint = camera.eye + rayDirection * primaryHit.hitDistance;
        float3 sphereCenter = bvh.centers[primaryHit.leafIndex];
        float3 normal = normalize(hitPoint - sphereCenter);
        float3 surfaceAlbedo = bvh.colors[primaryHit.leafIndex];

        // Vector to point light
        float3 toLight = light.position - hitPoint;
        float distanceToLight = length(toLight);
        float3 lightDirection = toLight / distanceToLight;

        // Offset shadow ray origin along normal to prevent self-intersection
        constexpr float shadowOffsetEpsilon = 0.001f;
        float3 shadowRayOrigin = hitPoint + normal * shadowOffsetEpsilon;

        bool isInShadow = bvhAnyHit(bvh, shadowRayOrigin, lightDirection, distanceToLight);

        float diffuseFactor = 0.0f;
        if (!isInShadow) {
            diffuseFactor = fmaxf(dot(normal, lightDirection), 0.0f);
        }

        // Composite illumination: (ambient + diffuse) * albedo * intensity
        float3 totalIllumination = light.ambient + make_float3(diffuseFactor, diffuseFactor, diffuseFactor);
        pixelColor = surfaceAlbedo * totalIllumination * light.intensity;
    }

    int pixelIndex = pixelY * camera.width + pixelX;
    outPixels[pixelIndex] = toRgb8(pixelColor);
}



Tracer::Tracer(int viewportWidth, int viewportHeight)
    : width_{viewportWidth}, height_{viewportHeight}
{
    size_t totalBytes = static_cast<size_t>(width_) * height_ * sizeof(uchar3);
    CUDA_CHECK(cudaMalloc(&devicePixels_, totalBytes));
}

Tracer::~Tracer() {
    if (devicePixels_ != nullptr) {
        cudaFree(devicePixels_);
    }
}

void Tracer::render(const BvhView& bvh, const Camera& camera, const PointLight& light) {
    rayTraceKernel<<<getPixelGrid(width_, height_), PIXEL_TILE_DIMENSIONS>>>(
        bvh,
        camera,
        light,
        devicePixels_
    );
    cudaCheckKernel();
}

void Tracer::download(std::vector<uchar3>& outPixels) const {
    outPixels.resize(static_cast<size_t>(width_) * height_);
    size_t totalBytes = outPixels.size() * sizeof(uchar3);

    CUDA_CHECK(cudaMemcpy(
        outPixels.data(),
        devicePixels_,
        totalBytes,
        cudaMemcpyDeviceToHost
    ));
}

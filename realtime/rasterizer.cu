#include "rasterizer.cuh"

#include "common/cuda_check.cuh"
#include "common/geometry.cuh"
#include "common/vec_math.cuh"


static constexpr unsigned long long EMPTY_DEPTH_KEY = ~0ull;
static constexpr unsigned GRID_PRIMITIVE_ID  = 0xFFFFFFFFu;
static constexpr int THREADS_PER_BLOCK = 256;
static const dim3 PIXEL_TILE_DIMENSIONS(16, 16);


// Packs a positive floating-point depth and an object index into an unsigned 64-bit integer.
// Because IEEE 754 positive float bits preserve order, a standard integer comparison directly compares depth.
__device__ inline unsigned long long packDepthAndId(float depth, unsigned id) {
    unsigned long long depthBits = static_cast<unsigned long long>(__float_as_uint(depth));
    return (depthBits << 32) | static_cast<unsigned long long>(id);
}

__device__ inline float unpackDepth(unsigned long long key) {
    unsigned depthBits = static_cast<unsigned>(key >> 32);
    return __uint_as_float(depthBits);
}

__device__ inline unsigned unpackId(unsigned long long key) {
    return static_cast<unsigned>(key & 0xFFFFFFFFu);
}

static inline int getBlockCount(int totalElements, int threadsPerBlock) {
    return (totalElements + threadsPerBlock - 1) / threadsPerBlock;
}

static inline dim3 getPixelGrid(int width, int height) {
    return dim3(
        (width  + PIXEL_TILE_DIMENSIONS.x - 1) / PIXEL_TILE_DIMENSIONS.x,
        (height + PIXEL_TILE_DIMENSIONS.y - 1) / PIXEL_TILE_DIMENSIONS.y
    );
}


// One thread per active sphere: identifies covered pixels, solves analytic ray-sphere intersections,
// and submits the nearest surface hit to the depth buffer via atomicMin.
__global__ void splatSpheresKernel(
    const float3* centers,
    const int* indices,
    int totalSpheres,
    float radius,
    Camera camera,
    unsigned long long* depthIdBuffer)
{
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex >= totalSpheres) {
        return;
    }

    unsigned objectId = static_cast<unsigned>(indices[threadIndex]);
    float3 center = centers[objectId];

    PixelBox bounds;
    if (!computeSphereScreenBounds(camera, center, radius, bounds)) {
        return;
    }

    bool hitAnyPixelCenter = false;

    // Test rays through every pixel covered by the bounding box
    for (int y = bounds.minY; y <= bounds.maxY; ++y) {
        for (int x = bounds.minX; x <= bounds.maxX; ++x) {
            float3 rayDirection = camera.rayThroughPixel(x + 0.5f, y + 0.5f);
            float hitDistance = intersectSphere(camera.eye, rayDirection, center, radius);

            if (hitDistance <= 0.0f) {
                continue;
            }

            // Convert ray parametric distance to view-space depth along camera forward axis
            float viewSpaceDepth = hitDistance * dot(rayDirection, camera.forward);
            int pixelIndex = y * camera.width + x;

            atomicMin(&depthIdBuffer[pixelIndex], packDepthAndId(viewSpaceDepth, objectId));
            hitAnyPixelCenter = true;
        }
    }

    // Sub-pixel fallback: tiny spheres that miss continuous pixel centers are drawn as a single pixel
    if (!hitAnyPixelCenter) {
        float3 cameraPoint = camera.toCameraSpace(center);
        float2 pixel = camera.toPixel(cameraPoint);

        int pixelX = static_cast<int>(floorf(pixel.x));
        int pixelY = static_cast<int>(floorf(pixel.y));

        if (pixelX >= 0 && pixelX < camera.width && pixelY >= 0 && pixelY < camera.height) {
            int pixelIndex = pixelY * camera.width + pixelX;
            atomicMin(&depthIdBuffer[pixelIndex], packDepthAndId(cameraPoint.z, objectId));
        }
    }
}


// One thread per pixel: checks distance to all projected line segments
__global__ void drawGridKernel(
    const ScreenSegment* segments,
    int segmentCount,
    int width,
    int height,
    unsigned long long* depthIdBuffer)
{
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelX >= width || pixelY >= height) {
        return;
    }

    float sampleX = pixelX + 0.5f;
    float sampleY = pixelY + 0.5f;

    for (int i = 0; i < segmentCount; ++i) {
        ScreenSegment segment = segments[i];

        float deltaX = segment.endPixel.x - segment.startPixel.x;
        float deltaY = segment.endPixel.y - segment.startPixel.y;
        float segmentLengthSquared = (deltaX * deltaX) + (deltaY * deltaY);

        // Parametric projection of pixel center onto the 2D segment
        float projection = 0.0f;
        if (segmentLengthSquared > 0.0f) {
            float dotProduct = ((sampleX - segment.startPixel.x) * deltaX) + 
                               ((sampleY - segment.startPixel.y) * deltaY);
            projection = clampf(dotProduct / segmentLengthSquared, 0.0f, 1.0f);
        }

        // Distance from pixel center to closest point on segment
        float closestX = segment.startPixel.x + projection * deltaX;
        float closestY = segment.startPixel.y + projection * deltaY;
        float diffX = closestX - sampleX;
        float diffY = closestY - sampleY;
        float distanceSquared = (diffX * diffX) + (diffY * diffY);

        // 1-pixel line thickness: radius is 0.5 pixels, radius squared is 0.25
        if (distanceSquared > 0.25f) {
            continue;
        }

        // Perspective-correct depth interpolation via reciprocal depth
        float invDepth = segment.invDepthStart + projection * (segment.invDepthEnd - segment.invDepthStart);
        float depth = 1.0f / invDepth;

        int pixelIndex = pixelY * width + pixelX;
        atomicMin(&depthIdBuffer[pixelIndex], packDepthAndId(depth, GRID_PRIMITIVE_ID));
    }
}


// Resolves winning depth/ID buffer entries into the final color framebuffer
__global__ void resolveKernel(
    const unsigned long long* depthIdBuffer,
    const float3* centers,
    const float3* colors,
    Camera camera,
    bool enableShading,
    uchar3* outPixels)
{
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelX >= camera.width || pixelY >= camera.height) {
        return;
    }

    int pixelIndex = pixelY * camera.width + pixelX;
    unsigned long long packedValue = depthIdBuffer[pixelIndex];

    float3 finalColor = make_float3(0.0f, 0.0f, 0.0f); // Default background

    if (packedValue != EMPTY_DEPTH_KEY) {
        unsigned primitiveId = unpackId(packedValue);

        if (primitiveId == GRID_PRIMITIVE_ID) {
            finalColor = make_float3(1.0f, 1.0f, 1.0f); // White grid lines
        } else {
            finalColor = colors[primitiveId];

            if (enableShading) {
                // Reconstruct surface hit position from depth and pixel ray
                float3 rayDirection = camera.rayThroughPixel(pixelX + 0.5f, pixelY + 0.5f);
                float rayDistance = unpackDepth(packedValue) / dot(rayDirection, camera.forward);

                float3 surfaceHit = camera.eye + rayDirection * rayDistance;
                float3 normal = normalize(surfaceHit - centers[primitiveId]);

                // Directional light from (1, 1, 1) with 0.5 ambient floor
                float3 lightDirection = normalize(make_float3(1.0f, 1.0f, 1.0f));
                float diffuseIntensity = fmaxf(dot(normal, lightDirection), 0.5f);

                finalColor = finalColor * diffuseIntensity;
            }
        }
    }

    outPixels[pixelIndex] = toRgb8(finalColor);
}


Rasterizer::Rasterizer(int viewportWidth, int viewportHeight)
    : width_{viewportWidth}, height_{viewportHeight}
{
    size_t totalPixels = static_cast<size_t>(width_) * height_;
    CUDA_CHECK(cudaMalloc(&deviceDepthId_, totalPixels * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&devicePixels_, totalPixels * sizeof(uchar3)));
}


Rasterizer::~Rasterizer() {
    if (deviceDepthId_ != nullptr) cudaFree(deviceDepthId_);
    if (devicePixels_  != nullptr) cudaFree(devicePixels_);
    if (deviceSegments_!= nullptr) cudaFree(deviceSegments_);
}


void Rasterizer::clear() {
    // 0xFF pattern sets each unsigned long long to ~0ull (EMPTY_DEPTH_KEY)
    size_t bufferBytes = static_cast<size_t>(width_) * height_ * sizeof(unsigned long long);
    CUDA_CHECK(cudaMemset(deviceDepthId_, 0xFF, bufferBytes));
}


void Rasterizer::drawGrid(const std::vector<Segment>& segments, const Camera& camera) {
    std::vector<ScreenSegment> visibleSegments;

    // Project and clip segments against the camera near clip plane on host
    for (const Segment& segment : segments) {
        float3 cameraA = camera.toCameraSpace(segment.startPoint);
        float3 cameraB = camera.toCameraSpace(segment.endPoint);

        // Fully behind near clip plane
        if (cameraA.z < camera.nearPlane && cameraB.z < camera.nearPlane) {
            continue;
        }

        // Clip endpoint A against near plane
        if (cameraA.z < camera.nearPlane) {
            float clipT = (camera.nearPlane - cameraA.z) / (cameraB.z - cameraA.z);
            cameraA = cameraA + (cameraB - cameraA) * clipT;
        }

        // Clip endpoint B against near plane
        if (cameraB.z < camera.nearPlane) {
            float clipT = (camera.nearPlane - cameraB.z) / (cameraA.z - cameraB.z);
            cameraB = cameraB + (cameraA - cameraB) * clipT;
        }

        visibleSegments.push_back({
            camera.toPixel(cameraA),
            camera.toPixel(cameraB),
            1.0f / cameraA.z,
            1.0f / cameraB.z
        });
    }

    if (visibleSegments.empty()) {
        return;
    }

    // Reallocate segment buffer on device if capacity is exceeded
    int requiredCount = static_cast<int>(visibleSegments.size());
    if (requiredCount > segmentCapacity_) {
        if (deviceSegments_ != nullptr) {
            cudaFree(deviceSegments_);
        }
        segmentCapacity_ = requiredCount;
        CUDA_CHECK(cudaMalloc(&deviceSegments_, segmentCapacity_ * sizeof(ScreenSegment)));
    }

    CUDA_CHECK(cudaMemcpy(
        deviceSegments_,
        visibleSegments.data(),
        visibleSegments.size() * sizeof(ScreenSegment),
        cudaMemcpyHostToDevice
    ));

    drawGridKernel<<<getPixelGrid(width_, height_), PIXEL_TILE_DIMENSIONS>>>(
        deviceSegments_,
        requiredCount,
        width_,
        height_,
        deviceDepthId_
    );
    cudaCheckKernel();
}


void Rasterizer::drawSpheres(
    const float3* deviceCenters,
    const int* deviceIndices,
    int activeCount,
    float radius,
    const Camera& camera)
{
    if (activeCount == 0) {
        return;
    }

    int gridBlocks = getBlockCount(activeCount, THREADS_PER_BLOCK);
    splatSpheresKernel<<<gridBlocks, THREADS_PER_BLOCK>>>(
        deviceCenters,
        deviceIndices,
        activeCount,
        radius,
        camera,
        deviceDepthId_
    );
    cudaCheckKernel();
}


void Rasterizer::resolve(
    const float3* deviceCenters,
    const float3* deviceColors,
    const Camera& camera,
    bool enableShading)
{
    resolveKernel<<<getPixelGrid(width_, height_), PIXEL_TILE_DIMENSIONS>>>(
        deviceDepthId_,
        deviceCenters,
        deviceColors,
        camera,
        enableShading,
        devicePixels_
    );
    cudaCheckKernel();
}


void Rasterizer::download(std::vector<uchar3>& outPixels) const {
    outPixels.resize(static_cast<size_t>(width_) * height_);
    CUDA_CHECK(cudaMemcpy(
        outPixels.data(),
        devicePixels_,
        outPixels.size() * sizeof(uchar3),
        cudaMemcpyDeviceToHost
    ));
}

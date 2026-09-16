#include "bvh.cuh"

#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <cstdio>
#include <cstdlib>

#include "bvh_build.cuh"
#include "common/cuda_check.cuh"

static constexpr int THREADS_PER_BLOCK = 256;

static inline int getBlockCount(int totalElements, int threadsPerBlock) {
    return (totalElements + threadsPerBlock - 1) / threadsPerBlock;
}

// Calculates a 30-bit Morton code for each sphere center normalized to [0, 1]^3
__global__ void computeMortonCodesKernel(
    const float3* positions,
    const int* indices,
    int totalSpheres,
    float sceneRadius,
    unsigned* outMortonCodes,
    int* outSortedIndices)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= totalSpheres) {
        return;
    }

    int particleIndex = indices[index];
    float3 position = positions[particleIndex];

    // Normalize coordinates from [-sceneRadius, sceneRadius] to [0, 1]
    float invDomain = 1.0f / (2.0f * sceneRadius);
    float3 normalizedPos = (position + make_float3(sceneRadius, sceneRadius, sceneRadius)) * invDomain;

    outMortonCodes[index] = mortonCode(normalizedPos);
    outSortedIndices[index] = particleIndex;
}

// Copies sorted particle data to leaf positions and initializes leaf bounding boxes
__global__ void buildLeavesKernel(
    const float3* positions,
    const float3* colors,
    const int* sortedIndices,
    int totalSpheres,
    float sphereRadius,
    float3* outCenters,
    float3* outColors,
    BvhNode* outNodes)
{
    int leafIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (leafIndex >= totalSpheres) {
        return;
    }

    int particleIndex = sortedIndices[leafIndex];
    float3 center = positions[particleIndex];

    outCenters[leafIndex] = center;
    outColors[leafIndex] = colors[particleIndex];

    // Leaves reside in the upper partition: [totalSpheres - 1, 2 * totalSpheres - 1)
    int nodeIndex = (totalSpheres - 1) + leafIndex;
    BvhNode& leafNode = outNodes[nodeIndex];

    float3 extent = make_float3(sphereRadius, sphereRadius, sphereRadius);
    leafNode.boxMin = center - extent;
    leafNode.boxMax = center + extent;
    leafNode.left   = -1;
    leafNode.right  = -1;
    leafNode.parent = -1;
}

// Constructs internal radix tree nodes by determining partition boundaries
__global__ void buildRadixTreeKernel(
    const unsigned* sortedMortonCodes,
    int totalSpheres,
    BvhNode* outNodes)
{
    int internalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (internalIndex >= totalSpheres - 1) {
        return;
    }

    Children children = radixTreeChildren(sortedMortonCodes, totalSpheres, internalIndex);

    outNodes[internalIndex].left = children.left;
    outNodes[internalIndex].right = children.right;
    outNodes[children.left].parent = internalIndex;
    outNodes[children.right].parent = internalIndex;

    if (internalIndex == 0) {
        outNodes[0].parent = -1; // Root node has no parent
    }
}

// Propagates bounding boxes bottom-up from leaves to the root
__global__ void propagateBoundingBoxesKernel(
    int totalSpheres,
    BvhNode* nodes,
    int* visitCounts)
{
    int leafIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (leafIndex >= totalSpheres) {
        return;
    }

    propagateBoxesUpward(leafIndex, totalSpheres, nodes, visitCounts);
}

// --- Bvh Class Implementation ---

Bvh::Bvh(int maxSpheres) : capacity_{maxSpheres} {
    CUDA_CHECK(cudaMalloc(&deviceMortonCodes_, maxSpheres * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&deviceSortedIndices_, maxSpheres * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&deviceCenters_, maxSpheres * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&deviceColors_, maxSpheres * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&deviceNodes_, 2 * static_cast<size_t>(maxSpheres) * sizeof(BvhNode)));
    CUDA_CHECK(cudaMalloc(&deviceVisitCounts_, maxSpheres * sizeof(int)));
}

Bvh::~Bvh() {
    if (deviceMortonCodes_)   cudaFree(deviceMortonCodes_);
    if (deviceSortedIndices_) cudaFree(deviceSortedIndices_);
    if (deviceCenters_)       cudaFree(deviceCenters_);
    if (deviceColors_)        cudaFree(deviceColors_);
    if (deviceNodes_)         cudaFree(deviceNodes_);
    if (deviceVisitCounts_)   cudaFree(deviceVisitCounts_);
}

void Bvh::build(
    const float3* devicePositions,
    const float3* deviceColors,
    const int* deviceIndices,
    int count,
    float sphereRadius,
    float sceneRadius)
{
    if (count > capacity_) {
        std::fprintf(stderr, "Error: Sphere count (%d) exceeds allocated BVH capacity (%d)\n", count, capacity_);
        std::exit(EXIT_FAILURE);
    }

    leafCount_ = count;
    radius_ = sphereRadius;

    if (count == 0) {
        return;
    }

    int leafBlocks = getBlockCount(count, THREADS_PER_BLOCK);

    // 1. Compute Morton codes
    computeMortonCodesKernel<<<leafBlocks, THREADS_PER_BLOCK>>>(
        devicePositions,
        deviceIndices,
        count,
        sceneRadius,
        deviceMortonCodes_,
        deviceSortedIndices_
    );
    cudaCheckKernel();

    // 2. Sort primitives along the Z-order curve
    thrust::sort_by_key(
        thrust::device,
        deviceMortonCodes_,
        deviceMortonCodes_ + count,
        deviceSortedIndices_
    );

    // 3. Build leaf entries
    buildLeavesKernel<<<leafBlocks, THREADS_PER_BLOCK>>>(
        devicePositions,
        deviceColors,
        deviceSortedIndices_,
        count,
        sphereRadius,
        deviceCenters_,
        deviceColors_,
        deviceNodes_
    );
    cudaCheckKernel();

    // A single-sphere scene requires no internal radix hierarchy
    if (count == 1) {
        return;
    }

    // 4. Construct internal radix tree nodes
    int internalBlocks = getBlockCount(count - 1, THREADS_PER_BLOCK);
    buildRadixTreeKernel<<<internalBlocks, THREADS_PER_BLOCK>>>(
        deviceMortonCodes_,
        count,
        deviceNodes_
    );
    cudaCheckKernel();

    // 5. Fit bounding boxes bottom-up
    CUDA_CHECK(cudaMemset(deviceVisitCounts_, 0, (count - 1) * sizeof(int)));
    propagateBoundingBoxesKernel<<<leafBlocks, THREADS_PER_BLOCK>>>(
        count,
        deviceNodes_,
        deviceVisitCounts_
    );
    cudaCheckKernel();
}

BvhView Bvh::view() const {
    return BvhView{deviceNodes_, deviceCenters_, deviceColors_, leafCount_, radius_};
}

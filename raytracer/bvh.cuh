#pragma once

#include <cuda_runtime.h>
#include "common/geometry.cuh"
#include "common/vec_math.cuh"

// Node in the linear bounding volume hierarchy (LBVH)
struct BvhNode {
    float3 boxMin{};
    float3 boxMax{};
    int left{-1};    // Child index (internal node) or -1 (leaf)
    int right{-1};   // Child index (internal node) or -1 (leaf)
    int parent{-1};  // Parent index or -1 (root)
};

// Read-only tree descriptor passed by value to device kernels.
// Internal nodes span indices [0, leafCount - 1), while leaf nodes occupy [leafCount - 1, 2 * leafCount - 1).
struct BvhView {
    const BvhNode* nodes{nullptr};
    const float3* centers{nullptr};
    const float3* colors{nullptr};
    int leafCount{0};
    float radius{0.0f};

    [[nodiscard]] __host__ __device__ bool isLeaf(int nodeIndex) const {
        return nodeIndex >= (leafCount - 1);
    }

    [[nodiscard]] __host__ __device__ int leafIndex(int nodeIndex) const {
        return nodeIndex - (leafCount - 1);
    }
};

// Linear Bounding Volume Hierarchy constructed on the GPU via Morton-code radix trees
class Bvh {
public:
    explicit Bvh(int maxSpheres);
    ~Bvh();

    Bvh(const Bvh&) = delete;
    Bvh& operator=(const Bvh&) = delete;

    Bvh(Bvh&&) noexcept = default;
    Bvh& operator=(Bvh&&) noexcept = default;

    // Rebuilds the acceleration structure over the provided sphere buffers
    void build(
        const float3* devicePositions,
        const float3* deviceColors,
        const int* deviceIndices,
        int count,
        float sphereRadius,
        float sceneRadius
    );

    [[nodiscard]] BvhView view() const;

private:
    int capacity_{0};
    int leafCount_{0};
    float radius_{0.0f};

    // Device scratch buffers
    unsigned* deviceMortonCodes_{nullptr};
    int* deviceSortedIndices_{nullptr};
    float3* deviceCenters_{nullptr};
    float3* deviceColors_{nullptr};
    BvhNode* deviceNodes_{nullptr};
    int* deviceVisitCounts_{nullptr};
};

// Result of a ray-BVH intersection query
struct RayHit {
    int leafIndex{-1};       // Index of hit sphere, or -1 on miss
    float hitDistance{0.0f}; // Parametric distance along ray
};

// Maximum traversal stack depth (64 levels accommodates 30-bit Morton radix trees)
static constexpr int BVH_STACK_CAPACITY = 64;

// Traverses the BVH using an explicit local stack.
// If findAnyHit is true, terminates immediately upon encountering any valid intersection (shadow ray test).
[[nodiscard]] __host__ __device__ inline RayHit bvhTrace(
    const BvhView& bvh,
    float3 rayOrigin,
    float3 rayDirection,
    float maxDistance,
    bool findAnyHit)
{
    RayHit closestHit{-1, maxDistance};
    if (bvh.leafCount == 0) {
        return closestHit;
    }

    float3 inverseDirection = safeInverse(rayDirection);

    int traversalStack[BVH_STACK_CAPACITY];
    int stackPointer = 0;
    traversalStack[stackPointer++] = 0; // Push root node

    while (stackPointer > 0) {
        int currentNodeIndex = traversalStack[--stackPointer];
        const BvhNode node = bvh.nodes[currentNodeIndex];

        // Cull subtrees that do not intersect the ray closer than our current best hit
        if (!rayHitsBox(rayOrigin, inverseDirection, node.boxMin, node.boxMax, closestHit.hitDistance)) {
            continue;
        }

        if (bvh.isLeaf(currentNodeIndex)) {
            int leaf = bvh.leafIndex(currentNodeIndex);
            float distance = intersectSphere(rayOrigin, rayDirection, bvh.centers[leaf], bvh.radius);

            if (distance > 0.0f && distance < closestHit.hitDistance) {
                closestHit.leafIndex = leaf;
                closestHit.hitDistance = distance;

                if (findAnyHit) {
                    return closestHit;
                }
            }
        } else if (stackPointer + 2 <= BVH_STACK_CAPACITY) {
            traversalStack[stackPointer++] = node.left;
            traversalStack[stackPointer++] = node.right;
        }
    }

    return closestHit;
}

// Queries the closest sphere hit along the ray
[[nodiscard]] __host__ __device__ inline RayHit bvhClosestHit(
    const BvhView& bvh,
    float3 rayOrigin,
    float3 rayDirection,
    float maxDistance)
{
    return bvhTrace(bvh, rayOrigin, rayDirection, maxDistance, false);
}

// Evaluates boolean ray occlusion for shadow casting
[[nodiscard]] __host__ __device__ inline bool bvhAnyHit(
    const BvhView& bvh,
    float3 rayOrigin,
    float3 rayDirection,
    float maxDistance)
{
    return bvhTrace(bvh, rayOrigin, rayDirection, maxDistance, true).leafIndex >= 0;
}

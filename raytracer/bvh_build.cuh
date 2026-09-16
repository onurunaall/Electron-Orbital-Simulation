#pragma once

#include <cuda_runtime.h>
#include "bvh.cuh"
#include "common/vec_math.cuh"

// Dilates a 10-bit integer by inserting two zero bits between each bit.
// Prepares x, y, and z coordinates for 3D bit-interleaving into a 30-bit Morton code.
[[nodiscard]] __host__ __device__ inline unsigned spreadBits(unsigned value) {
    value = (value * 0x00010001u) & 0xFF0000FFu;
    value = (value * 0x00000101u) & 0x0F00F00Fu;
    value = (value * 0x00000011u) & 0xC30C30C3u;
    value = (value * 0x00000005u) & 0x49249249u;
    return value;
}

// Generates a 30-bit Morton code for a normalized coordinate in [0, 1]^3.
// Quantizes each axis to 10 bits (0-1023) and interleaves them along the Z-order space-filling curve.
[[nodiscard]] __host__ __device__ inline unsigned mortonCode(float3 point) {
    unsigned x = static_cast<unsigned>(clampf(point.x * 1024.0f, 0.0f, 1023.0f));
    unsigned y = static_cast<unsigned>(clampf(point.y * 1024.0f, 0.0f, 1023.0f));
    unsigned z = static_cast<unsigned>(clampf(point.z * 1024.0f, 0.0f, 1023.0f));

    return (spreadBits(x) << 2) | (spreadBits(y) << 1) | spreadBits(z);
}

// Counts leading zero bits in a 32-bit unsigned integer
[[nodiscard]] __host__ __device__ inline int countLeadingZeros(unsigned value) {
#ifdef __CUDA_ARCH__
    return __clz(value);
#else
    for (int bit = 0; bit < 32; ++bit) {
        if (value & (0x80000000u >> bit)) {
            return bit;
        }
    }
    return 32;
#endif
}

// Computes the number of matching leading bits between keys at index i and j.
// When Morton codes match, indices break ties to enforce a strict total ordering.
[[nodiscard]] __host__ __device__ inline int commonPrefix(
    const unsigned* sortedCodes,
    int totalCount,
    int indexA,
    int indexB) 
{
    if (indexB < 0 || indexB >= totalCount) {
        return -1;
    }

    unsigned codeA = sortedCodes[indexA];
    unsigned codeB = sortedCodes[indexB];

    if (codeA == codeB) {
        return 32 + countLeadingZeros(static_cast<unsigned>(indexA ^ indexB));
    }

    return countLeadingZeros(codeA ^ codeB);
}

struct Children {
    int left{-1};
    int right{-1};
};

// Determines the left and right child node indices for internal node i in the radix tree
[[nodiscard]] __host__ __device__ inline Children radixTreeChildren(
    const unsigned* sortedCodes,
    int leafCount,
    int internalNodeIndex) 
{
    const int leafNodeOffset = leafCount - 1;

    // Determine search direction (+1 or -1) toward the neighbor with the longer common prefix
    int prefixNext = commonPrefix(sortedCodes, leafCount, internalNodeIndex, internalNodeIndex + 1);
    int prefixPrev = commonPrefix(sortedCodes, leafCount, internalNodeIndex, internalNodeIndex - 1);
    int direction = (prefixNext > prefixPrev) ? 1 : -1;

    // Find upper bound for the range length
    int minPrefix = commonPrefix(sortedCodes, leafCount, internalNodeIndex, internalNodeIndex - direction);
    int maxLength = 2;
    while (commonPrefix(sortedCodes, leafCount, internalNodeIndex, internalNodeIndex + maxLength * direction) > minPrefix) {
        maxLength *= 2;
    }

    // Binary search for the exact range boundary
    int rangeLength = 0;
    for (int step = maxLength / 2; step >= 1; step /= 2) {
        if (commonPrefix(sortedCodes, leafCount, internalNodeIndex, internalNodeIndex + (rangeLength + step) * direction) > minPrefix) {
            rangeLength += step;
        }
    }
    int otherEnd = internalNodeIndex + rangeLength * direction;

    // Binary search for the split point within the range
    int nodePrefix = commonPrefix(sortedCodes, leafCount, internalNodeIndex, otherEnd);
    int splitOffset = 0;
    int step = rangeLength;

    do {
        step = (step + 1) / 2;
        if (commonPrefix(sortedCodes, leafCount, internalNodeIndex, internalNodeIndex + (splitOffset + step) * direction) > nodePrefix) {
            splitOffset += step;
        }
    } while (step > 1);

    int split = internalNodeIndex + splitOffset * direction + min(direction, 0);

    // Leaf nodes are offset by (leafCount - 1) in the flat node array
    int rangeStart = min(internalNodeIndex, otherEnd);
    int rangeEnd   = max(internalNodeIndex, otherEnd);

    Children children{};
    children.left  = (rangeStart == split)     ? (leafNodeOffset + split)     : split;
    children.right = (rangeEnd == split + 1)   ? (leafNodeOffset + split + 1) : (split + 1);

    return children;
}

// Atomic counter increment wrapper
[[nodiscard]] __host__ __device__ inline int incrementVisitCount(int* counter) {
#ifdef __CUDA_ARCH__
    return atomicAdd(counter, 1);
#else
    return (*counter)++;
#endif
}

// Memory fence to synchronize global memory writes across GPU thread blocks
__host__ __device__ inline void memoryFence() {
#ifdef __CUDA_ARCH__
    __threadfence();
#endif
}

// Propagates leaf bounding boxes upward toward the root using atomic flags
__host__ __device__ inline void propagateBoxesUpward(
    int leafIndex,
    int leafCount,
    BvhNode* nodes,
    int* visitCounts) 
{
    int currentNode = nodes[leafCount - 1 + leafIndex].parent;

    while (currentNode != -1) {
        // First thread to reach the parent exits; second thread computes the box union
        if (incrementVisitCount(&visitCounts[currentNode]) == 0) {
            return;
        }

        // Ensure the sibling thread's bounding box writes are visible
        memoryFence();

        BvhNode& parentNode = nodes[currentNode];
        const BvhNode& leftChild  = nodes[parentNode.left];
        const BvhNode& rightChild = nodes[parentNode.right];

        parentNode.boxMin = vmin(leftChild.boxMin, rightChild.boxMin);
        parentNode.boxMax = vmax(leftChild.boxMax, rightChild.boxMax);

        // Ensure updated bounding boxes are visible before moving to the next level
        memoryFence();

        currentNode = parentNode.parent;
    }
}

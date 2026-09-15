#pragma once

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>


// Quantum state definition: principal (n), azimuthal (l), and magnetic (m) numbers
struct Orbital {
    int n{2};
    int l{1};
    int m{0};
};


// Manages the GPU particle representation of a hydrogen orbital cloud
class ParticleCloud {
public:
    ParticleCloud(Orbital orbital, int particleCount, float colorScale, unsigned seed);
    ~ParticleCloud();

    // Non-copyable to prevent accidental double-frees of device memory
    ParticleCloud(const ParticleCloud&) = delete;
    ParticleCloud& operator=(const ParticleCloud&) = delete;

    // Default move operations
    ParticleCloud(ParticleCloud&&) noexcept = default;
    ParticleCloud& operator=(ParticleCloud&&) noexcept = default;

    // Advances the probability current flow for a time step dt
    void advance(float dt);

    [[nodiscard]] int count() const { return particleCount_; }
    [[nodiscard]] const float3* positions() const { return devicePositions_; }
    [[nodiscard]] const float3* colors() const { return deviceColors_; }
    [[nodiscard]] float maxRadius() const { return boundingRadius_; }

private:
    Orbital orbital_{};
    int particleCount_{0};
    float boundingRadius_{0.0f};

    // Device buffer pointers
    float3* deviceSpherical_{nullptr};   // Internal state: (r, theta, phi) per particle
    float3* devicePositions_{nullptr};   // Cartesian coordinates: (x, y, z)
    float3* deviceColors_{nullptr};      // RGB colors in [0, 1]
};


// Filters particles using a device predicate and stores matching indices into outIndicesBuffer.
// Returns the total number of particles that satisfied the predicate.
template <typename FilterPredicate>
[[nodiscard]] int selectParticles(
    const float3* devicePositions,
    int particleCount,
    int* outIndicesBuffer,
    FilterPredicate keepPredicate) 
{
    const thrust::counting_iterator<int> indexBegin{0};
    const thrust::counting_iterator<int> indexEnd{particleCount};

    int* copiedEnd{thrust::copy_if(
        thrust::device,
        indexBegin,
        indexEnd,
        devicePositions,
        outIndicesBuffer,
        keepPredicate
    )};

    return static_cast<int>(copiedEnd - outIndicesBuffer);
}

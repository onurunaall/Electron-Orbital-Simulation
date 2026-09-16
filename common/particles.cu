#include "particles.cuh"

#include <curand_kernel.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <cmath>
#include <cstdlib>
#include <vector>


#include "cdf.cuh"
#include "colormap.cuh"
#include "cuda_check.cuh"
#include "hydrogen.cuh"
#include "vec_math.cuh"


static constexpr int RADIAL_TABLE_SIZE = 4096;
static constexpr int POLAR_TABLE_SIZE  = 2048;
static constexpr int THREADS_PER_BLOCK = 256;

static inline int getBlockCount(int totalElements, int threadsPerBlock) {
    return (totalElements + threadsPerBlock - 1) / threadsPerBlock;
}

static float* uploadTableToDevice(const std::vector<float>& hostTable) {
    float* deviceBuffer = nullptr;
    size_t bytes = hostTable.size() * sizeof(float);

    CUDA_CHECK(cudaMalloc(&deviceBuffer, bytes));
    CUDA_CHECK(cudaMemcpy(deviceBuffer, hostTable.data(), bytes, cudaMemcpyHostToDevice));

    return deviceBuffer;
}


// Samples particle positions and colors according to the hydrogen wave function
__global__ void sampleParticlesKernel(
    int count,
    Orbital orbital,
    CdfTable radialTable,
    CdfTable polarTable,
    float colorScale,
    unsigned seed,
    float3* sphericalOut,
    float3* positionsOut,
    float3* colorsOut)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    // Initialize an independent Philox random number generator for this particle
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, index, 0, &rng);

    float uRadius = curand_uniform(&rng);
    float uTheta = curand_uniform(&rng);
    float uPhi = curand_uniform(&rng);

    // Sample coordinates using Inverse CDF lookup
    float r = sampleFromCdf(radialTable, uRadius);
    float theta = sampleFromCdf(polarTable, uTheta);
    float phi = 2.0f * kPi * uPhi;

    // Store coordinates, calculate Cartesian position and color
    sphericalOut[index] = make_float3(r, theta, phi);
    positionsOut[index] = sphericalToCartesian(r, theta, phi);
    colorsOut[index] = orbitalColor(orbital.n, orbital.l, orbital.m, r, theta, colorScale);
}

// Advances particles along the probability current around the y-axis
__global__ void advanceParticlesKernel(
    int count,
    int m,
    float dt,
    float3* sphericalInOut,
    float3* positionsOut)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    float3 coords = sphericalInOut[index];
    float r = coords.x;
    float theta = coords.y;
    float phi = coords.z;

    // Velocity is undefined at the nucleus; leave stationary
    if (r < 1e-6f) {
        return;
    }

    // Cylindrical distance to the y-axis, clamped near the poles to avoid division by zero
    float cylinderRadius = r * fmaxf(sinf(theta), 1e-4f);

    // Exact rotation over dt: deltaPhi = v*dt/rho = m*dt/rho^2
    float deltaPhi = (float)m * dt / (cylinderRadius * cylinderRadius);

    // Cap the per-step rotation. Near the y-axis the exact angular velocity
    // diverges; without this, fmodf below loses all precision.
    constexpr float maxStepRadians = 0.5f * kPi;
    deltaPhi = clampf(deltaPhi, -maxStepRadians, maxStepRadians);

    phi = fmodf(phi + deltaPhi, 2.0f * kPi);

    // Save updated coordinates
    sphericalInOut[index].z = phi;
    positionsOut[index] = sphericalToCartesian(r, theta, phi);
}

// Finds the maximum radial distance among all particles using Thrust
static float findMaxRadius(const float3* deviceSpherical, int count) {
    return thrust::transform_reduce(
        thrust::device,
        deviceSpherical,
        deviceSpherical + count,
        [] __device__ (float3 s) { return s.x; },
        0.0f,
        thrust::maximum<float>()
    );
}

ParticleCloud::ParticleCloud(Orbital orbital, int count, float colorScale, unsigned seed) : orbital_(orbital), particleCount_(count) {
    // Radial cutoff based on principal quantum number
    double rMax = 10.0 * orbital.n * orbital.n;
    int absM = std::abs(orbital.m);

    // Build CDF distribution tables on CPU
    std::vector<float> radialCdf = buildCdf(RADIAL_TABLE_SIZE, rMax, [&](double r) {
        double R = radialWavefunction(orbital.n, orbital.l, r);
        return r * r * R * R;
    });

    std::vector<float> polarCdf = buildCdf(POLAR_TABLE_SIZE, kPi, [&](double theta) {
        double P = associatedLegendre(orbital.l, absM, cos(theta));
        return sin(theta) * P * P;
    });

    // Upload CDF tables to GPU
    float* d_radial = uploadTableToDevice(radialCdf);
    float* d_polar  = uploadTableToDevice(polarCdf);

    float radialStep = (float)(rMax / (RADIAL_TABLE_SIZE - 1));
    float polarStep  = (float)(kPi / (POLAR_TABLE_SIZE - 1));

    CdfTable radialTable = { d_radial, RADIAL_TABLE_SIZE, radialStep };
    CdfTable polarTable  = { d_polar,  POLAR_TABLE_SIZE,  polarStep };

    // Allocate device buffers
    size_t bufferSize = count * sizeof(float3);
    CUDA_CHECK(cudaMalloc(&deviceSpherical_, bufferSize));
    CUDA_CHECK(cudaMalloc(&devicePositions_, bufferSize));
    CUDA_CHECK(cudaMalloc(&deviceColors_,    bufferSize));

    // Sample particles
    int blocks = getBlockCount(count, THREADS_PER_BLOCK);
    sampleParticlesKernel<<<blocks, THREADS_PER_BLOCK>>>(
        count,
        orbital_,
        radialTable,
        polarTable,
        colorScale,
        seed,
        deviceSpherical_,
        devicePositions_,
        deviceColors_
    );
    cudaCheckKernel();

    CUDA_CHECK(cudaFree(d_radial));
    CUDA_CHECK(cudaFree(d_polar));

    // Compute radius for camera/bounding volume checks
    boundingRadius_ = findMaxRadius(deviceSpherical_, count);
}


ParticleCloud::~ParticleCloud() {
    if (deviceSpherical_) cudaFree(deviceSpherical_);
    if (devicePositions_) cudaFree(devicePositions_);
    if (deviceColors_)    cudaFree(deviceColors_);
}

void ParticleCloud::advance(float dt) {
    int blocks = getBlockCount(particleCount_, THREADS_PER_BLOCK);
    advanceParticlesKernel<<<blocks, THREADS_PER_BLOCK>>>(
        particleCount_,
        orbital_.m,
        dt,
        deviceSpherical_,
        devicePositions_
    );
    cudaCheckKernel();
}

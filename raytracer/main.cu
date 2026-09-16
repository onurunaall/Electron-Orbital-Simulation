#include <cstdio>
#include <filesystem>
#include <vector>

#include "bvh.cuh"
#include "common/cuda_check.cuh"
#include "common/image.h"
#include "common/options.h"
#include "common/particles.cuh"
#include "tracer.cuh"

static void displayHelp(const CommonOptions& defaults) {
    std::printf("Usage: atom_raytracer [options]\n");
    printCommonUsage(defaults);
    std::printf(
        "  --sphere-radius <f>   Particle sphere radius (default: 0.25)\n"
        "  --color-scale <f>     Multiplier from density to heatmap input (default: 700.0)\n"
    );
}

int main(int argc, char** argv) {
    CommandLine cli{argc, argv};

    // Default configuration for offline ray tracer: 3d orbital state
    CommonOptions options{};
    options.n = 3;
    options.l = 1;
    options.m = 1;
    options.particleCount = 100000;
    options.outputDir = "frames_raytracer";

    if (cli.hasFlag("--help") || cli.hasFlag("-h")) {
        displayHelp(options);
        return 0;
    }

    options = parseCommonOptions(cli, options);
    if (!validateCommonOptions(options)) {
        return 1;
    }

    float sphereRadius = cli.getFloat("--sphere-radius", 0.25f);
    float colorScale   = cli.getFloat("--color-scale", static_cast<float>(1.0 / peakProbabilityDensity(options.n, options.l, options.m)));

    // Scene point light setup
    PointLight sceneLight{
        make_float3(0.0f, 50.0f, 50.0f),
        make_float3(0.2f, 0.2f, 0.2f),
        3.0f
    };

    std::printf("Initializing ray-traced simulation: %d particles (n=%d, l=%d, m=%d)\n",
                options.particleCount, options.n, options.l, options.m);

    // Initialize quantum particle cloud and acceleration structures
    ParticleCloud particleCloud(
        {options.n, options.l, options.m},
        options.particleCount,
        colorScale,
        options.seed
    );

    Bvh bvh(options.particleCount);
    Tracer tracer(options.width, options.height);

    // Conservative scene bounding radius for Morton code normalization
    float maxSceneRadius = particleCloud.maxRadius() + sphereRadius;

    // Buffer for compacted visible particle indices
    int* deviceVisibleIndices = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceVisibleIndices, options.particleCount * sizeof(int)));

    std::vector<uchar3> hostPixels;
    std::filesystem::create_directories(options.outputDir);

    // Animation rendering loop
    for (int frame = 0; frame < options.frames; ++frame) {
        // Step physics along quantum probability current
        particleCloud.advance(options.dt);

        // Cut away quadrant where y > 0 and z > 0 to expose internal nodal surfaces
        int visibleCount = selectParticles(
            particleCloud.positions(),
            particleCloud.count(),
            deviceVisibleIndices,
            [] __device__ (float3 p) {
                return (p.z < 0.0f || p.y < 0.0f);
            }
        );

        // Rebuild BVH over visible particles
        bvh.build(
            particleCloud.positions(),
            particleCloud.colors(),
            deviceVisibleIndices,
            visibleCount,
            sphereRadius,
            maxSceneRadius
        );

        Camera camera = cameraForFrame(options, frame);

        // Render frame with primary rays and shadows
        tracer.render(bvh.view(), camera, sceneLight);

        // Download and write frame to disk
        tracer.download(hostPixels);
        std::string filePath = framePath(options.outputDir, frame);

        if (!writePpm(filePath, options.width, options.height, hostPixels)) {
            CUDA_CHECK(cudaFree(deviceVisibleIndices));
            return 1;
        }

        std::printf("Rendered frame %d/%d -> %s (%d spheres traced)\n",
                    frame + 1, options.frames, filePath.c_str(), visibleCount);
    }

    CUDA_CHECK(cudaFree(deviceVisibleIndices));
    return 0;
}

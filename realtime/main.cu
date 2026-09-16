#include <cmath>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "common/cuda_check.cuh"
#include "common/image.h"
#include "common/options.h"
#include "common/particles.cuh"
#include "rasterizer.cuh"

// Generates ground grid segments on the plane y = 0
static std::vector<Segment> buildGroundGrid(float totalSize, int cellDivisions) {
    std::vector<Segment> segments;

    float stepSize = totalSize / static_cast<float>(cellDivisions);
    float halfSize = totalSize * 0.5f;
    float axisExtension = 3.0f * stepSize;
    int centerIndex = cellDivisions / 2;

    for (int i = 0; i <= cellDivisions; ++i) {
        float currentOffset = -halfSize + static_cast<float>(i) * stepSize;

        // Lines parallel to the x-axis
        float xStart = -halfSize;
        float xEnd   =  halfSize;
        if (i == centerIndex) {
            xStart -= axisExtension;
            xEnd   += axisExtension;
        }
        segments.push_back({
            make_float3(xStart, 0.0f, currentOffset),
            make_float3(xEnd,   0.0f, currentOffset)
        });

        // Lines parallel to the z-axis
        segments.push_back({
            make_float3(currentOffset, 0.0f, -halfSize),
            make_float3(currentOffset, 0.0f,  halfSize)
        });
    }

    return segments;
}


static void displayHelp(const CommonOptions& defaults) {
    std::printf("Usage: atom_realtime [options]\n");
    printCommonUsage(defaults);
    std::printf(
        "  --sphere-radius <f>   Particle sphere radius (default: 0.05 * n / 3)\n"
        "  --shaded              Enable Lambertian surface lighting\n"
    );
}


int main(int argc, char** argv) {
    CommandLine cli{argc, argv};

    // Default configuration for realtime preview
    CommonOptions options{};
    options.n = 2;
    options.l = 1;
    options.m = 0;
    options.particleCount = 250000;
    options.outputDir = "frames_realtime";

    if (cli.hasFlag("--help") || cli.hasFlag("-h")) {
        displayHelp(options);
        return 0;
    }

    options = parseCommonOptions(cli, options);
    if (!validateCommonOptions(options)) {
        return 1;
    }

    // Default sphere radius scaled by the orbital size (n / 3)
    float defaultRadius = 0.05f * static_cast<float>(options.n) / 3.0f;
    float sphereRadius = cli.getFloat("--sphere-radius", defaultRadius);
    bool enableShading = cli.hasFlag("--shaded");

    // Color normalization scaling factor for this orbital energy level
    float colorScale = 1.5f * powf(5.0f, static_cast<float>(options.n));

    std::printf("Initializing simulation: %d particles (n=%d, l=%d, m=%d)\n",
                options.particleCount, options.n, options.l, options.m);

    // Instantiate simulation components
    ParticleCloud particleCloud(
        {options.n, options.l, options.m},
        options.particleCount,
        colorScale,
        options.seed
    );

    Rasterizer rasterizer(options.width, options.height);
    std::vector<Segment> groundGrid = buildGroundGrid(500.0f, 2);

    // Allocate GPU buffer for compaction indices of visible particles
    int* deviceVisibleIndices = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceVisibleIndices, options.particleCount * sizeof(int)));

    std::vector<uchar3> hostPixels;
    std::filesystem::create_directories(options.outputDir);

    // Frame rendering loop
    for (int frame = 0; frame < options.frames; ++frame) {
        // Step physics along quantum probability current
        particleCloud.advance(options.dt);

        // Filter out quadrant (x < 0 and y > 0) to expose internal orbital shell structure
        int visibleCount = selectParticles(
            particleCloud.positions(),
            particleCloud.count(),
            deviceVisibleIndices,
            [] __device__ (float3 p) {
                return !(p.x < 0.0f && p.y > 0.0f);
            }
        );

        Camera camera = cameraForFrame(options, frame);

        // Rendering passes
        rasterizer.clear();
        rasterizer.drawGrid(groundGrid, camera);
        rasterizer.drawSpheres(
            particleCloud.positions(),
            deviceVisibleIndices,
            visibleCount,
            sphereRadius,
            camera
        );
        rasterizer.resolve(
            particleCloud.positions(),
            particleCloud.colors(),
            camera,
            enableShading
        );

        // Download and save frame
        rasterizer.download(hostPixels);
        std::string filePath = framePath(options.outputDir, frame);

        if (!writePpm(filePath, options.width, options.height, hostPixels)) {
            CUDA_CHECK(cudaFree(deviceVisibleIndices));
            return 1;
        }

        std::printf("Rendered frame %d/%d -> %s (%d particles visible)\n",
                    frame + 1, options.frames, filePath.c_str(), visibleCount);
    }

    CUDA_CHECK(cudaFree(deviceVisibleIndices));
    return 0;
}

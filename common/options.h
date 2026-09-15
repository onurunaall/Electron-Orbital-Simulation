#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "camera.cuh"


// Simple command-line parser supporting "--flag <value>" style arguments
struct CommandLine {
    int argumentCount{0};
    char** argumentValues{nullptr};

    [[nodiscard]] bool hasFlag(const char* flag) const {
        for (int i{1}; i < argumentCount; ++i) {
            if (std::strcmp(argumentValues[i], flag) == 0) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] const char* getArgumentValue(const char* flag) const {
        for (int i{1}; i + 1 < argumentCount; ++i) {
            if (std::strcmp(argumentValues[i], flag) == 0) {
                return argumentValues[i + 1];
            }
        }
        return nullptr;
    }

    [[nodiscard]] int getInt(const char* flag, int fallbackValue) const {
        const char* val{getArgumentValue(flag)};
        if (val == nullptr) {
            return fallbackValue;
        }
        return std::atoi(val);
    }

    [[nodiscard]] float getFloat(const char* flag, float fallbackValue) const {
        const char* val{getArgumentValue(flag)};
        if (val == nullptr) {
            return fallbackValue;
        }
        return static_cast<float>(std::atof(val));
    }

    [[nodiscard]] std::string getString(const char* flag, const std::string& fallbackValue) const {
        const char* val{getArgumentValue(flag)};
        if (val == nullptr) {
            return fallbackValue;
        }
        return std::string{val};
    }
};


// Simulation and rendering parameters with default values
struct CommonOptions {
    // Quantum numbers
    int n{2};
    int l{1};
    int m{0};

    // Particle cloud settings
    int particleCount{100000};
    int frames{1};
    float dt{0.5f};

    // Framebuffer resolution
    int width{800};
    int height{600};

    // Camera orbit parameters
    float cameraRadius{50.0f};
    float azimuthDeg{0.0f};
    float elevationDeg{90.0f};
    float spinDegPerFrame{0.0f};

    // Random generator & file export
    unsigned seed{42};
    std::string outputDir{"frames"};
};


inline void printCommonUsage(const CommonOptions& defaults) {
    std::printf(
        "  --n <int>           Principal quantum number (default %d)\n"
        "  --l <int>           Azimuthal quantum number, 0 <= l < n (default %d)\n"
        "  --m <int>           Magnetic quantum number, -l <= m <= l (default %d)\n"
        "  --particles <int>   Number of sampled particles (default %d)\n"
        "  --frames <int>      Number of frames to render (default %d)\n"
        "  --dt <float>        Probability-current time step per frame (default %g)\n"
        "  --width <int>       Image width in pixels (default %d)\n"
        "  --height <int>      Image height in pixels (default %d)\n"
        "  --radius <float>    Camera distance from the nucleus (default %g)\n"
        "  --azimuth <deg>     Camera azimuth in degrees (default %g)\n"
        "  --elevation <deg>   Camera elevation from +y axis in degrees (default %g)\n"
        "  --spin <deg>        Azimuth rotation added per frame (default %g)\n"
        "  --seed <int>        Random seed for particle generator (default %u)\n"
        "  --out <dir>         Output directory for rendered frames (default %s)\n",
        defaults.n, defaults.l, defaults.m,
        defaults.particleCount, defaults.frames, defaults.dt,
        defaults.width, defaults.height,
        defaults.cameraRadius, defaults.azimuthDeg, defaults.elevationDeg, defaults.spinDegPerFrame,
        defaults.seed, defaults.outputDir.c_str());
}


[[nodiscard]] inline CommonOptions parseCommonOptions(const CommandLine& cli, const CommonOptions& defaults) {
    CommonOptions opts{defaults};

    opts.n = cli.getInt("--n", opts.n);
    opts.l = cli.getInt("--l", opts.l);
    opts.m = cli.getInt("--m", opts.m);
    opts.particleCount = cli.getInt("--particles", opts.particleCount);
    opts.frames = cli.getInt("--frames", opts.frames);
    opts.dt = cli.getFloat("--dt", opts.dt);
    opts.width = cli.getInt("--width", opts.width);
    opts.height = cli.getInt("--height", opts.height);
    opts.cameraRadius = cli.getFloat("--radius", opts.cameraRadius);
    opts.azimuthDeg = cli.getFloat("--azimuth", opts.azimuthDeg);
    opts.elevationDeg = cli.getFloat("--elevation", opts.elevationDeg);
    opts.spinDegPerFrame = cli.getFloat("--spin", opts.spinDegPerFrame);
    opts.seed = static_cast<unsigned>(cli.getInt("--seed", static_cast<int>(opts.seed)));
    opts.outputDir = cli.getString("--out", opts.outputDir);

    return opts;
}


[[nodiscard]] inline bool validateCommonOptions(const CommonOptions& opts) {
    const bool validQuantumNumbers{
        (opts.n >= 1) &&
        (opts.l >= 0 && opts.l < opts.n) &&
        (opts.m >= -opts.l && opts.m <= opts.l)
    };

    if (!validQuantumNumbers) {
        std::fprintf(stderr,
            "Error: Invalid quantum numbers (n=%d, l=%d, m=%d). Requirements: n >= 1, 0 <= l < n, |m| <= l\n",
            opts.n, opts.l, opts.m);
        return false;
    }

    if (opts.particleCount < 1 || opts.frames < 1 || opts.width < 1 || opts.height < 1) {
        std::fprintf(stderr, "Error: particle count, frames, width, and height must all be >= 1\n");
        return false;
    }

    if (opts.cameraRadius < 1.0f) {
        std::fprintf(stderr, "Error: Camera radius must be >= 1.0\n");
        return false;
    }

    return true;
}


[[nodiscard]] inline Camera cameraForFrame(const CommonOptions& opts, int frameIndex) {
    constexpr float degToRad{kPi / 180.0f};

    const float currentAzimuth{(opts.azimuthDeg + frameIndex * opts.spinDegPerFrame) * degToRad};
    const float currentElevation{opts.elevationDeg * degToRad};
    constexpr float verticalFovDegrees{45.0f};
    constexpr float nearClipPlane{0.1f};

    return Camera::orbit(
        opts.cameraRadius,
        currentAzimuth,
        currentElevation,
        verticalFovDegrees,
        nearClipPlane,
        opts.width,
        opts.height
    );
}

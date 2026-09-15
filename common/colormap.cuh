#pragma once

#include "vec_math.cuh"
#include "hydrogen.cuh"


// Evaluates a 6-stop fire heatmap (black -> purple -> red -> orange -> yellow -> white) for t in [0, 1]
[[nodiscard]] __host__ __device__ inline float3 fireHeatmap(float t) {
    constexpr int colorStopsCount{6};
    constexpr float3 colorStops[colorStopsCount]{
        {0.0f, 0.0f, 0.0f},   // Black
        {0.3f, 0.0f, 0.6f},   // Dark purple
        {0.8f, 0.0f, 0.0f},   // Deep red
        {1.0f, 0.5f, 0.0f},   // Orange
        {1.0f, 1.0f, 0.0f},   // Yellow
        {1.0f, 1.0f, 1.0f},   // White
    };

    const float scaledValue{clampf(t, 0.0f, 1.0f) * static_cast<float>(colorStopsCount - 1)};
    const int currentIndex{static_cast<int>(scaledValue)};
    const int nextIndex{currentIndex + 1 < colorStopsCount ? currentIndex + 1 : colorStopsCount - 1};

    const float interpolationFactor{scaledValue - static_cast<float>(currentIndex)};

    const float3 startColor{colorStops[currentIndex]};
    const float3 endColor{colorStops[nextIndex]};

    return startColor + (endColor - startColor) * interpolationFactor;
}


// Computes the RGB particle color for orbital (n, l, m) at spherical coordinates (r, theta).
// `densityScale` stretches the probability density onto the [0, 1] heatmap range.
[[nodiscard]] __host__ __device__ inline float3 orbitalColor(
    int principalQuantumNumber,
    int azimuthalQuantumNumber,
    int magneticQuantumNumber,
    float radius,
    float theta,
    float densityScale) 
{
    const double probabilityDensityValue{probabilityDensity(
        principalQuantumNumber,
        azimuthalQuantumNumber,
        magneticQuantumNumber,
        static_cast<double>(radius),
        static_cast<double>(theta)
    )};

    const float normalizedIntensity{static_cast<float>(probabilityDensityValue) * densityScale};
    return fireHeatmap(normalizedIntensity);
}

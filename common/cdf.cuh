#pragma once

#include <cuda_runtime.h>
#include <vector>


// Tabulated Cumulative Distribution Function (CDF) sampled over a uniform grid:
// x_i = i * step, where i is in [0, size - 1], with values[0] = 0 and values[size - 1] = 1.
struct CdfTable {
    const float* values{nullptr};
    int size{0};
    float step{0.0f};
};


// Integrates an arbitrary 1D probability density function (PDF) across [0, xMax]
// using the trapezoid rule, then normalizes it to construct a 1D CDF table.
template <typename PdfFunction>
[[nodiscard]] std::vector<float> buildCdf(int tableSize, double xMax, PdfFunction evaluatePdf) {
    const double stepSize{xMax / static_cast<double>(tableSize - 1)};

    std::vector<double> accumulatedArea(tableSize, 0.0);
    double previousPdfValue{evaluatePdf(0.0)};

    for (int i{1}; i < tableSize; ++i) {
        const double currentX{static_cast<double>(i) * stepSize};
        const double currentPdfValue{evaluatePdf(currentX)};

        // Trapezoidal integration step: area = 0.5 * (f(x0) + f(x1)) * dx
        const double trapezoidArea{0.5 * (previousPdfValue + currentPdfValue) * stepSize};
        accumulatedArea[i] = accumulatedArea[i - 1] + trapezoidArea;

        previousPdfValue = currentPdfValue;
    }

    const double totalIntegral{accumulatedArea[tableSize - 1]};

    std::vector<float> normalizedCdf(tableSize, 0.0f);
    for (int i{0}; i < tableSize; ++i) {
        normalizedCdf[i] = static_cast<float>(accumulatedArea[i] / totalIntegral);
    }

    return normalizedCdf;
}


// Samples a random variable from the tabulated CDF given a uniform random float u in (0, 1]
// using binary search and linear interpolation between adjacent grid points.
[[nodiscard]] __host__ __device__ inline float sampleFromCdf(CdfTable table, float randomUniform) {
    // Binary search to find the smallest index where table.values[upperIndex] > randomUniform
    int lowerIndex{1};
    int upperIndex{table.size - 1};

    while (lowerIndex < upperIndex) {
        const int midIndex{(lowerIndex + upperIndex) / 2};
        if (table.values[midIndex] > randomUniform) {
            upperIndex = midIndex;
        } else {
            lowerIndex = midIndex + 1;
        }
    }

    const float cdfBefore{table.values[upperIndex - 1]};
    const float cdfAfter{table.values[upperIndex]};

    // Boundary guard for randomUniform >= 1.0f: clamp directly to the maximum domain value
    if (randomUniform >= cdfAfter) {
        return static_cast<float>(table.size - 1) * table.step;
    }

    // Linear interpolation across the interval [upperIndex - 1, upperIndex]
    const float fraction{(randomUniform - cdfBefore) / (cdfAfter - cdfBefore)};
    const float continuousIndex{static_cast<float>(upperIndex - 1) + fraction};

    return continuousIndex * table.step;
}

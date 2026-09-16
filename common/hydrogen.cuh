#pragma once

#include <cuda_runtime.h>
#include <cmath>


[[nodiscard]] constexpr __host__ __device__ inline double intPow(double base, int exp) {
    double result{1.0};
    while (exp > 0) {
        if (exp & 1) {
            result *= base;
        }
        base *= base;
        exp >>= 1;
    }
    return result;
}


// Generalized Laguerre polynomial L_degree^(alpha)(x) using a three-term recurrence
[[nodiscard]] constexpr __host__ __device__ inline double laguerre(int degree, int alpha, double x) {
    if (degree == 0) {
        return 1.0;
    }

    double p0{1.0};
    double p1{1.0 + alpha - x};

    for (int k{2}; k <= degree; ++k) {
        double pNext{((2 * k - 1 + alpha - x) * p1 - (k - 1 + alpha) * p0) / k};
        p0 = p1;
        p1 = pNext;
    }

    return p1;
}


// Associated Legendre polynomial P_l^m(x) for 0 <= m <= l
[[nodiscard]] __host__ __device__ inline double associatedLegendre(int l, int m, double x) {
    // Base case: P_m^m(x) = (-1)^m * (2m - 1)!! * (1 - x^2)^(m / 2)
    double pmm{1.0};

    if (m > 0) {
        double sinTheta{sqrt((1.0 - x) * (1.0 + x))};
        double factor{1.0};
        for (int i{1}; i <= m; ++i) {
            pmm *= -factor * sinTheta;
            factor += 2.0;
        }
    }

    if (l == m) {
        return pmm;
    }

    // P_{m+1}^m(x) = x * (2m + 1) * P_m^m(x)
    double p1{x * (2 * m + 1) * pmm};
    if (l == m + 1) {
        return p1;
    }

    // Upward recurrence to l
    double p0{pmm};
    for (int k{m + 2}; k <= l; ++k) {
        double pNext{((x * (2 * k - 1) * p1) - (k + m - 1) * p0) / (k - m)};
        p0 = p1;
        p1 = pNext;
    }

    return p1;
}


// Normalized radial wave function R_nl(r)
[[nodiscard]] __host__ __device__ inline double radialWavefunction(int n, int l, double radius) {
    const double invN{2.0 / n};
    const double rho{invN * radius};

    // Calculate (n - l - 1)! / (n + l)! directly, skipping heavy tgamma calls
    double factorialDenominator{1.0};
    for (int k{n - l}; k <= n + l; ++k) {
        factorialDenominator *= k;
    }

    const double invNCubed{(invN * invN) * invN};
    const double normConstant{invNCubed / (2.0 * n * factorialDenominator)};

    // Deconstruct into standard quantum mechanical terms
    const double amplitude{sqrt(normConstant)};
    const double decay{exp(-0.5 * rho)};
    const double centrifugal{intPow(rho, l)};
    const double polynomial{laguerre(n - l - 1, 2 * l + 1, rho)};

    return amplitude * decay * centrifugal * polynomial;
}


// Probability density |psi_{nlm}(r, theta, phi)|^2 without angular normalization
[[nodiscard]] __host__ __device__ inline double probabilityDensity(int n, int l, int m, double radius, double theta) {
    int absM{m};
    if (absM < 0) {
        absM = -absM;
    }

    const double R{radialWavefunction(n, l, radius)};
    const double P{associatedLegendre(l, absM, cos(theta))};

    // Spherical-harmonic normalization: (2l+1)/(4 pi) * (l-|m|)! / (l+|m|)!
    // The factorial ratio is 1 / [(l-|m|+1) * ... * (l+|m|)], so one loop does it.
    constexpr double fourPi{12.566370614359172};
    double angularNorm{(2.0 * l + 1.0) / fourPi};
    for (int k{l - absM + 1}; k <= l + absM; ++k) {
        angularNorm /= static_cast<double>(k);
    }

    return angularNorm * (R * R) * (P * P);
}
}

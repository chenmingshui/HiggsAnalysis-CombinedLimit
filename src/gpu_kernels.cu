// CUDA kernels used by the Combine GPU backend.
//
// This file is only compiled when Combine is configured with -DUSE_CUDA=TRUE.
// The host side wrappers (combine_gpu::*) take care of the device memory
// management through GpuManager, so that the kernels themselves stay free of
// any allocation logic.

#include "GpuKernels.h"

#ifdef HAS_CUDA

#include <cuda_runtime.h>

#include <cstddef>

#include "../interface/GpuManager.h"

namespace combine_gpu {
namespace {

int const kBlock = 256;

// ---------------------------------------------------------------------------
// Binned Poisson NLL
// ---------------------------------------------------------------------------
__global__ void poissonNllDouble(double const * __restrict__ expected,
                                 double const * __restrict__ observed,
                                 int nBins,
                                 double * __restrict__ nll) {
    int const p = blockIdx.x;
    double const * row = expected + static_cast<std::size_t>(p) * nBins;
    double sum = 0.0;
    for (int i = threadIdx.x; i < nBins; i += blockDim.x) {
        double const e = row[i];
        double const o = observed[i];
        if (e > 0.0) sum += e - o * log(e);
    }
    __shared__ double sdata[kBlock];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) nll[p] = sdata[0];
}

__global__ void poissonNllFloat(float const * __restrict__ expected,
                                float const * __restrict__ observed,
                                int nBins,
                                float * __restrict__ nll) {
    int const p = blockIdx.x;
    float const * row = expected + static_cast<std::size_t>(p) * nBins;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < nBins; i += blockDim.x) {
        float const e = row[i];
        float const o = observed[i];
        if (e > 0.0f) sum += e - o * logf(e);
    }
    __shared__ float sdata[kBlock];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) nll[p] = sdata[0];
}

// ---------------------------------------------------------------------------
// Gaussian constraints
// ---------------------------------------------------------------------------
__global__ void gaussianConstraintDouble(double const * __restrict__ values,
                                         double const * __restrict__ means,
                                         double const * __restrict__ sigmas,
                                         int nPars,
                                         double * __restrict__ constraints) {
    int const p = blockIdx.x;
    double const * row = values + static_cast<std::size_t>(p) * nPars;
    double sum = 0.0;
    for (int i = threadIdx.x; i < nPars; i += blockDim.x) {
        double const d = (row[i] - means[i]) / sigmas[i];
        sum += 0.5 * d * d;
    }
    __shared__ double sdata[kBlock];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) constraints[p] = sdata[0];
}

// ---------------------------------------------------------------------------
// Poisson constant term log(obs!)
// ---------------------------------------------------------------------------
__global__ void logFactorialDouble(double const * __restrict__ observed,
                                   int nBins,
                                   double * __restrict__ logFact) {
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nBins) logFact[i] = lgamma(observed[i] + 1.0);
}

}  // namespace

bool batchPoissonNll(double const * expected, double const * observed,
                     int nPoints, int nBins, double * nll) {
    if (!GpuManager::instance().available()) return false;
    if (nPoints <= 0 || nBins <= 0 || !expected || !observed || !nll) return false;

    GpuManager & gpu = GpuManager::instance();
    std::size_t const expectedBytes = static_cast<std::size_t>(nPoints) * nBins * sizeof(double);
    std::size_t const observedBytes = static_cast<std::size_t>(nBins) * sizeof(double);
    std::size_t const nllBytes = static_cast<std::size_t>(nPoints) * sizeof(double);

    double * dExpected = nullptr;
    double * dObserved = nullptr;
    double * dNll = nullptr;
    try {
        dExpected = static_cast<double *>(gpu.allocate(expectedBytes));
        dObserved = static_cast<double *>(gpu.allocate(observedBytes));
        dNll = static_cast<double *>(gpu.allocate(nllBytes));
        gpu.memcpyHostToDevice(dExpected, expected, expectedBytes);
        gpu.memcpyHostToDevice(dObserved, observed, observedBytes);
        poissonNllDouble<<<nPoints, kBlock>>>(dExpected, dObserved, nBins, dNll);
        gpu.deviceSynchronize();
        gpu.memcpyDeviceToHost(nll, dNll, nllBytes);
    } catch (...) {
        if (dExpected) gpu.free(dExpected);
        if (dObserved) gpu.free(dObserved);
        if (dNll) gpu.free(dNll);
        throw;
    }
    gpu.free(dExpected);
    gpu.free(dObserved);
    gpu.free(dNll);
    return true;
}

bool batchPoissonNll(float const * expected, float const * observed,
                     int nPoints, int nBins, float * nll) {
    if (!GpuManager::instance().available()) return false;
    if (nPoints <= 0 || nBins <= 0 || !expected || !observed || !nll) return false;

    GpuManager & gpu = GpuManager::instance();
    std::size_t const expectedBytes = static_cast<std::size_t>(nPoints) * nBins * sizeof(float);
    std::size_t const observedBytes = static_cast<std::size_t>(nBins) * sizeof(float);
    std::size_t const nllBytes = static_cast<std::size_t>(nPoints) * sizeof(float);

    float * dExpected = nullptr;
    float * dObserved = nullptr;
    float * dNll = nullptr;
    try {
        dExpected = static_cast<float *>(gpu.allocate(expectedBytes));
        dObserved = static_cast<float *>(gpu.allocate(observedBytes));
        dNll = static_cast<float *>(gpu.allocate(nllBytes));
        gpu.memcpyHostToDevice(dExpected, expected, expectedBytes);
        gpu.memcpyHostToDevice(dObserved, observed, observedBytes);
        poissonNllFloat<<<nPoints, kBlock>>>(dExpected, dObserved, nBins, dNll);
        gpu.deviceSynchronize();
        gpu.memcpyDeviceToHost(nll, dNll, nllBytes);
    } catch (...) {
        if (dExpected) gpu.free(dExpected);
        if (dObserved) gpu.free(dObserved);
        if (dNll) gpu.free(dNll);
        throw;
    }
    gpu.free(dExpected);
    gpu.free(dObserved);
    gpu.free(dNll);
    return true;
}

bool batchGaussianConstraint(double const * values, double const * means,
                             double const * sigmas, int nPoints, int nPars,
                             double * constraints) {
    if (!GpuManager::instance().available()) return false;
    if (nPoints <= 0 || nPars <= 0 || !values || !means || !sigmas || !constraints) return false;

    GpuManager & gpu = GpuManager::instance();
    std::size_t const valuesBytes = static_cast<std::size_t>(nPoints) * nPars * sizeof(double);
    std::size_t const parsBytes = static_cast<std::size_t>(nPars) * sizeof(double);
    std::size_t const outBytes = static_cast<std::size_t>(nPoints) * sizeof(double);

    double * dValues = nullptr;
    double * dMeans = nullptr;
    double * dSigmas = nullptr;
    double * dConstraints = nullptr;
    try {
        dValues = static_cast<double *>(gpu.allocate(valuesBytes));
        dMeans = static_cast<double *>(gpu.allocate(parsBytes));
        dSigmas = static_cast<double *>(gpu.allocate(parsBytes));
        dConstraints = static_cast<double *>(gpu.allocate(outBytes));
        gpu.memcpyHostToDevice(dValues, values, valuesBytes);
        gpu.memcpyHostToDevice(dMeans, means, parsBytes);
        gpu.memcpyHostToDevice(dSigmas, sigmas, parsBytes);
        gaussianConstraintDouble<<<nPoints, kBlock>>>(dValues, dMeans, dSigmas, nPars, dConstraints);
        gpu.deviceSynchronize();
        gpu.memcpyDeviceToHost(constraints, dConstraints, outBytes);
    } catch (...) {
        if (dValues) gpu.free(dValues);
        if (dMeans) gpu.free(dMeans);
        if (dSigmas) gpu.free(dSigmas);
        if (dConstraints) gpu.free(dConstraints);
        throw;
    }
    gpu.free(dValues);
    gpu.free(dMeans);
    gpu.free(dSigmas);
    gpu.free(dConstraints);
    return true;
}

bool batchLogFactorial(double const * observed, int nBins, double * logFact) {
    if (!GpuManager::instance().available()) return false;
    if (nBins <= 0 || !observed || !logFact) return false;

    GpuManager & gpu = GpuManager::instance();
    std::size_t const bytes = static_cast<std::size_t>(nBins) * sizeof(double);
    double * dObserved = nullptr;
    double * dLogFact = nullptr;
    try {
        dObserved = static_cast<double *>(gpu.allocate(bytes));
        dLogFact = static_cast<double *>(gpu.allocate(bytes));
        gpu.memcpyHostToDevice(dObserved, observed, bytes);
        int const blocks = (nBins + kBlock - 1) / kBlock;
        logFactorialDouble<<<blocks, kBlock>>>(dObserved, nBins, dLogFact);
        gpu.deviceSynchronize();
        gpu.memcpyDeviceToHost(logFact, dLogFact, bytes);
    } catch (...) {
        if (dObserved) gpu.free(dObserved);
        if (dLogFact) gpu.free(dLogFact);
        throw;
    }
    gpu.free(dObserved);
    gpu.free(dLogFact);
    return true;
}

}  // namespace combine_gpu

#endif  // HAS_CUDA

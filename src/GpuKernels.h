#ifndef HiggsAnalysis_CombinedLimit_GpuKernels_h
#define HiggsAnalysis_CombinedLimit_GpuKernels_h

// Internal interface to the CUDA kernels. This header is only included by
// src/gpu_kernels.cu (device + host wrappers) and src/GpuNllEngine.cc (host
// wrappers). It is not part of the public Combine API.

namespace combine_gpu {

/// Batch binned Poisson NLL.
/// @param expected  [nPoints][nBins] model predictions, row major
/// @param observed  [nBins] observed bin contents
/// @param nll       [nPoints] output, nll[p] = sum_i (e_i - o_i log(e_i))
bool batchPoissonNll(double const * expected, double const * observed,
                     int nPoints, int nBins, double * nll);

/// Single precision version of batchPoissonNll.
bool batchPoissonNll(float const * expected, float const * observed,
                     int nPoints, int nBins, float * nll);

/// Batch Gaussian constraint terms.
/// @param values       [nPoints][nPars] parameter values, row major
/// @param means        [nPars] constraint means
/// @param sigmas       [nPars] constraint widths (must be > 0)
/// @param constraints  [nPoints] output, 0.5 * sum_i ((v_i - m_i)/s_i)^2
bool batchGaussianConstraint(double const * values, double const * means,
                             double const * sigmas, int nPoints, int nPars,
                             double * constraints);

/// Batch Poisson constant term log(obs!).
bool batchLogFactorial(double const * observed, int nBins, double * logFact);

}  // namespace combine_gpu

#endif

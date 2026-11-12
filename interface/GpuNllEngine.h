#ifndef HiggsAnalysis_CombinedLimit_GpuNllEngine_h
#define HiggsAnalysis_CombinedLimit_GpuNllEngine_h

#include <memory>

class RooAbsReal;
class RooAbsPdf;
class RooAbsData;
class RooArgSet;

/**
 * @brief GPU accelerated likelihood evaluation for Combine.
 *
 * Two complementary mechanisms are exposed:
 *
 *  - createGpuNLL() builds a standard RooFit NLL whose evaluation is offloaded
 *    to the GPU through ROOT's own CUDA backend (RooFit::EvalBackend("cuda")).
 *    This is the general purpose path: it works for any RooFit model that the
 *    RooBatchCompute library knows how to evaluate.
 *
 *  - the batch*() helpers run Combine's own CUDA kernels for the operations
 *    that dominate toy based methods (binned Poisson NLL and Gaussian
 *    constraint terms). They are useful when the model predictions have already
 *    been computed and only the summation over bins/parameters remains.
 *
 * Everything degrades gracefully: if no CUDA device is present the batch
 * helpers return false and createGpuNLL() throws a descriptive exception.
 */
class GpuNllEngine {
public:
    GpuNllEngine();
    ~GpuNllEngine();

    /// true if a CUDA device usable by Combine is present.
    bool available() const;

    /// true if ROOT itself was built with the CUDA evaluation backend.
    bool rootCudaBackendAvailable() const;

    /// Create a NLL evaluated on the GPU through ROOT's CUDA backend.
    /// @throws std::runtime_error if the GPU or the ROOT CUDA backend is not
    /// available.
    std::unique_ptr<RooAbsReal> createGpuNLL(RooAbsPdf & pdf, RooAbsData & data,
                                             RooArgSet const * constraint = nullptr,
                                             bool offset = false);

    /// Batch binned Poisson NLL: nll[p] = sum_i (e_i - o_i log(e_i)).
    /// Returns false when the GPU path is unavailable.
    bool batchPoissonNll(double const * expected, double const * observed,
                         int nPoints, int nBins, double * nll) const;

    /// Batch Gaussian constraints: 0.5 * sum_i ((v_i - m_i)/s_i)^2.
    bool batchGaussianConstraint(double const * values, double const * means,
                                 double const * sigmas, int nPoints, int nPars,
                                 double * constraints) const;

    /// Batch Poisson constant term log(obs!).
    bool batchLogFactorial(double const * observed, int nBins, double * logFact) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

#endif

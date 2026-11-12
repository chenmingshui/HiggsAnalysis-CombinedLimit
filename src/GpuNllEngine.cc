#include "../interface/GpuNllEngine.h"
#include "../interface/GpuManager.h"
#include "../interface/Combine.h"
#include "../interface/CombineUtils.h"

#include "GpuKernels.h"

#include <RooAbsData.h>
#include <RooAbsPdf.h>
#include <RooAbsReal.h>
#include <RooArgSet.h>
#include <RooGlobalFunc.h>

#include <stdexcept>
#include <string>

class GpuNllEngine::Impl {
public:
    Impl() : cudaBackendProbed_(false), cudaBackendAvailable_(false) {}

    mutable bool cudaBackendProbed_;
    mutable bool cudaBackendAvailable_;
};

GpuNllEngine::GpuNllEngine() : impl_(new Impl) {}

GpuNllEngine::~GpuNllEngine() {}

bool GpuNllEngine::available() const { return GpuManager::instance().available(); }

bool GpuNllEngine::rootCudaBackendAvailable() const {
    if (!impl_->cudaBackendProbed_) {
        impl_->cudaBackendAvailable_ = false;
        try {
            std::string const name = RooFit::EvalBackend("cuda").name();
            impl_->cudaBackendAvailable_ = (name == "cuda");
        } catch (std::exception const &) {
            impl_->cudaBackendAvailable_ = false;
        }
        impl_->cudaBackendProbed_ = true;
    }
    return impl_->cudaBackendAvailable_;
}

std::unique_ptr<RooAbsReal> GpuNllEngine::createGpuNLL(RooAbsPdf & pdf,
                                                       RooAbsData & data,
                                                       RooArgSet const * constraint,
                                                       bool offset) {
    if (!available())
        throw std::runtime_error("GpuNllEngine: no CUDA device is available on this machine");
    if (!rootCudaBackendAvailable())
        throw std::runtime_error(
            "GpuNllEngine: ROOT was not built with the CUDA evaluation backend "
            "(a CUDA-enabled ROOT build, e.g. the LCG _cuda view, is required)");

    // Route the creation through Combine's single entry point so that the
    // CUDA backend gets exactly the same treatment as the other backends
    // (RooSimultaneousOpt passthrough, constraint handling, fallbacks, ...).
    Combine::nllBackend() = "cuda";
    return combineCreateNLL(pdf, data, constraint, offset, /*warnAboutDifferentBackend=*/false);
}

bool GpuNllEngine::batchPoissonNll(double const * expected, double const * observed,
                                   int nPoints, int nBins, double * nll) const {
    return combine_gpu::batchPoissonNll(expected, observed, nPoints, nBins, nll);
}

bool GpuNllEngine::batchGaussianConstraint(double const * values, double const * means,
                                           double const * sigmas, int nPoints, int nPars,
                                           double * constraints) const {
    return combine_gpu::batchGaussianConstraint(values, means, sigmas, nPoints, nPars, constraints);
}

bool GpuNllEngine::batchLogFactorial(double const * observed, int nBins, double * logFact) const {
    return combine_gpu::batchLogFactorial(observed, nBins, logFact);
}

// Host side stubs for the Combine GPU kernels.
//
// When Combine is built without CUDA support (the default) the CUDA kernels in
// gpu_kernels.cu are not compiled. This translation unit provides the host side
// symbols so that GpuNllEngine still links; every function simply reports that
// the GPU path is unavailable.
//
// When Combine is built with -DUSE_CUDA=TRUE the real implementations from
// gpu_kernels.cu are used instead and this file compiles to nothing.

#include "GpuKernels.h"

#ifndef HAS_CUDA

namespace combine_gpu {

bool batchPoissonNll(double const * /*expected*/, double const * /*observed*/,
                     int /*nPoints*/, int /*nBins*/, double * /*nll*/) {
    return false;
}

bool batchPoissonNll(float const * /*expected*/, float const * /*observed*/,
                     int /*nPoints*/, int /*nBins*/, float * /*nll*/) {
    return false;
}

bool batchGaussianConstraint(double const * /*values*/, double const * /*means*/,
                             double const * /*sigmas*/, int /*nPoints*/, int /*nPars*/,
                             double * /*constraints*/) {
    return false;
}

bool batchLogFactorial(double const * /*observed*/, int /*nBins*/, double * /*logFact*/) {
    return false;
}

}  // namespace combine_gpu

#endif  // !HAS_CUDA

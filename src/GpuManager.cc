#include "../interface/GpuManager.h"

#include <stdexcept>
#include <string>

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace {
// Translate a CUDA error code into an exception. Compiles away when Combine is
// built without CUDA support.
#ifdef HAS_CUDA
void cudaCheck(cudaError_t err, char const * what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " +
                                 cudaGetErrorString(err));
    }
}
#define COMBINE_CUDA_CHECK(expr) cudaCheck((expr), #expr)
#else
// When built without CUDA the macro must not even parse its argument, because
// the CUDA runtime functions are not declared in that configuration.
#define COMBINE_CUDA_CHECK(expr) do { } while (0)
#endif
}

class GpuManager::Impl {
public:
    Impl() : probed_(false), nDevices_(0), device_(-1) {}

    mutable bool probed_;
    int nDevices_;
    int device_;
};

GpuManager::GpuManager() : impl_(new Impl), enabled_(true) {}

GpuManager::~GpuManager() {}

GpuManager & GpuManager::instance() {
    static GpuManager mgr;
    return mgr;
}

bool GpuManager::available() const {
#ifdef HAS_CUDA
    if (!impl_->probed_) {
        int n = 0;
        if (cudaGetDeviceCount(&n) == cudaSuccess) impl_->nDevices_ = n;
        impl_->probed_ = true;
    }
    return impl_->nDevices_ > 0;
#else
    return false;
#endif
}

int GpuManager::numDevices() const {
#ifdef HAS_CUDA
    if (!impl_->probed_) {
        int n = 0;
        if (cudaGetDeviceCount(&n) == cudaSuccess) impl_->nDevices_ = n;
        impl_->probed_ = true;
    }
    return impl_->nDevices_;
#else
    return 0;
#endif
}

int GpuManager::device() const {
#ifdef HAS_CUDA
    return impl_->device_;
#else
    return -1;
#endif
}

void GpuManager::setDevice(int device) {
#ifndef HAS_CUDA
    (void)device;
    throw std::runtime_error("GpuManager: Combine was built without CUDA support");
#else
    if (device < 0 || device >= numDevices())
        throw std::runtime_error("GpuManager: device index out of range");
    COMBINE_CUDA_CHECK(cudaSetDevice(device));
    impl_->device_ = device;
#endif
}

std::string GpuManager::deviceName(int device) const {
#ifndef HAS_CUDA
    (void)device;
    return std::string();
#else
    if (device < 0 || device >= numDevices()) return std::string();
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return std::string();
    return std::string(prop.name);
#endif
}

std::size_t GpuManager::deviceMemory(int device) const {
#ifndef HAS_CUDA
    (void)device;
    return 0;
#else
    if (device < 0 || device >= numDevices()) return 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return 0;
    return static_cast<std::size_t>(prop.totalGlobalMem);
#endif
}

std::size_t GpuManager::deviceFreeMemory() const {
#ifndef HAS_CUDA
    return 0;
#else
    std::size_t freeBytes = 0, totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess) return 0;
    return freeBytes;
#endif
}

void GpuManager::setDevices(std::vector<int> const & devices) {
    int const n = numDevices();
    for (int d : devices) {
        if (d < 0 || (n > 0 && d >= n))
            throw std::runtime_error("GpuManager: device index out of range");
    }
    devices_ = devices;
}

void * GpuManager::allocate(std::size_t bytes) {
#ifndef HAS_CUDA
    (void)bytes;
    throw std::runtime_error("GpuManager: Combine was built without CUDA support");
#else
    void * ptr = nullptr;
    COMBINE_CUDA_CHECK(cudaMalloc(&ptr, bytes == 0 ? 1 : bytes));
    return ptr;
#endif
}

void GpuManager::free(void * ptr) {
#ifndef HAS_CUDA
    (void)ptr;
#else
    if (ptr) COMBINE_CUDA_CHECK(cudaFree(ptr));
#endif
}

void * GpuManager::allocatePinned(std::size_t bytes) {
#ifndef HAS_CUDA
    (void)bytes;
    throw std::runtime_error("GpuManager: Combine was built without CUDA support");
#else
    void * ptr = nullptr;
    COMBINE_CUDA_CHECK(cudaMallocHost(&ptr, bytes == 0 ? 1 : bytes));
    return ptr;
#endif
}

void GpuManager::freePinned(void * ptr) {
#ifndef HAS_CUDA
    (void)ptr;
#else
    if (ptr) COMBINE_CUDA_CHECK(cudaFreeHost(ptr));
#endif
}

void GpuManager::memcpyHostToDevice(void * dst, const void * src, std::size_t bytes) {
#ifndef HAS_CUDA
    (void)dst; (void)src; (void)bytes;
    throw std::runtime_error("GpuManager: Combine was built without CUDA support");
#else
    COMBINE_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
#endif
}

void GpuManager::memcpyDeviceToHost(void * dst, const void * src, std::size_t bytes) {
#ifndef HAS_CUDA
    (void)dst; (void)src; (void)bytes;
    throw std::runtime_error("GpuManager: Combine was built without CUDA support");
#else
    COMBINE_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
#endif
}

void GpuManager::memcpyDeviceToDevice(void * dst, const void * src, std::size_t bytes) {
#ifndef HAS_CUDA
    (void)dst; (void)src; (void)bytes;
    throw std::runtime_error("GpuManager: Combine was built without CUDA support");
#else
    COMBINE_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
#endif
}

void GpuManager::deviceSynchronize() {
#ifndef HAS_CUDA
    throw std::runtime_error("GpuManager: Combine was built without CUDA support");
#else
    COMBINE_CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

std::vector<std::pair<int, int> > GpuManager::splitWork(int nItems) const {
    std::vector<std::pair<int, int> > out;
    if (nItems <= 0) return out;

    std::vector<int> devs = devices_;
    if (devs.empty()) devs.push_back(device());

    int const nDev = static_cast<int>(devs.size());
    if (nDev <= 0) return out;

    int const base = nItems / nDev;
    int const rem = nItems % nDev;
    for (int i = 0; i < nDev; ++i) {
        int const count = base + (i < rem ? 1 : 0);
        if (count > 0) out.emplace_back(devs[i], count);
    }
    return out;
}

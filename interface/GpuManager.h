#ifndef HiggsAnalysis_CombinedLimit_GpuManager_h
#define HiggsAnalysis_CombinedLimit_GpuManager_h

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief Singleton that owns the CUDA context used by Combine.
 *
 * This is a thin wrapper around the CUDA runtime API. All CUDA calls are
 * hidden behind the PImpl idiom so that Combine can still be compiled without
 * a CUDA toolchain: in that case available() returns false and every other
 * method is either a no-op or throws a descriptive exception.
 *
 * The manager is deliberately physics agnostic: it only deals with devices,
 * memory and work distribution. The actual likelihood kernels live in
 * GpuNllEngine.
 */
class GpuManager {
public:
    /// Access the process wide instance.
    static GpuManager & instance();

    /// true if at least one CUDA capable device is visible.
    bool available() const;

    /// Number of visible CUDA devices.
    int numDevices() const;

    /// Index of the currently active device, or -1 if CUDA is unavailable.
    int device() const;

    /// Select the active device. Throws std::runtime_error if out of range or
    /// if Combine was built without CUDA support.
    void setDevice(int device);

    /// Globally enable or disable GPU offloading.
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool enabled() const { return enabled_; }

    /// Human readable name of a device (empty string if unavailable).
    std::string deviceName(int device) const;

    /// Total global memory of a device in bytes (0 if unavailable).
    std::size_t deviceMemory(int device) const;

    /// Free global memory of the active device in bytes (0 if unavailable).
    std::size_t deviceFreeMemory() const;

    /// Indices of the devices used for multi-GPU work.
    std::vector<int> const & devices() const { return devices_; }
    void setDevices(std::vector<int> const & devices);

    /// Device memory helpers. They throw if CUDA is unavailable.
    void * allocate(std::size_t bytes);
    void free(void * ptr);

    /// Page-locked host memory (faster host/device transfers).
    void * allocatePinned(std::size_t bytes);
    void freePinned(void * ptr);

    void memcpyHostToDevice(void * dst, const void * src, std::size_t bytes);
    void memcpyDeviceToHost(void * dst, const void * src, std::size_t bytes);
    void memcpyDeviceToDevice(void * dst, const void * src, std::size_t bytes);
    void deviceSynchronize();

    /// Split @p nItems independent work items over the selected devices.
    /// Returns a list of (device, itemCount) pairs. If no explicit device list
    /// was configured, the active device is used for all items.
    std::vector<std::pair<int, int> > splitWork(int nItems) const;

private:
    GpuManager();
    ~GpuManager();
    GpuManager(GpuManager const &) = delete;
    GpuManager & operator=(GpuManager const &) = delete;

    class Impl;
    std::unique_ptr<Impl> impl_;
    std::vector<int> devices_;
    bool enabled_;
};

#endif

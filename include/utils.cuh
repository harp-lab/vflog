
#pragma once

#include <cstdint>
#include <thrust/host_vector.h>

#include <assert.h>
// #include <cuda_runtime.h>
#include <iostream>

// use librmm
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/execution_policy.h>

#define checkCuda(ans)                                                         \
    { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort) {
            cudaDeviceReset();
            exit(code);
        }
    }
}

struct KernelTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

    KernelTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~KernelTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void start_timer() { cudaEventRecord(start, 0); }

    void stop_timer() { cudaEventRecord(stop, 0); }

    float get_spent_time() {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        elapsed /= 1000.0;
        return elapsed;
    }
};

enum RelationVersion { DELTA, FULL, NEWT };

// #define DEFAULT_DEVICE_POLICY thrust::device
// #define DEVICE_VECTOR thrust::device_vector
#define DEFAULT_DEVICE_POLICY rmm::exec_policy()
#define DEVICE_VECTOR rmm::device_vector
#define HOST_VECTOR thrust::host_vector
#define DEFAULT_SET_HASH_MAP true
#define DEFAULT_LOAD_FACTOR 0.9

namespace hisa {
using internal_data_type = uint32_t;

using offset_type = uint64_t;
using comp_range_t = uint64_t;
using comp_pair_t = uint64_t;

using tuple_type = thrust::host_vector<internal_data_type>;

using device_data_t = DEVICE_VECTOR<internal_data_type>;
using device_indices_t = DEVICE_VECTOR<internal_data_type>;
using device_bitmap_t = DEVICE_VECTOR<bool>;

using device_internal_data_ptr = thrust::device_ptr<internal_data_type>;

// higher 32 bit is the postion in sorted indices, lower is offset
using device_ranges_t = DEVICE_VECTOR<comp_range_t>;
using device_pairs_t = DEVICE_VECTOR<comp_pair_t>;

inline uint64_t __device__ __host__ compress_u32(uint32_t &a, uint32_t &b) {
    return ((uint64_t)a << 32) | b;
}

// functor to get the higher 32 bit
struct get_higher {
    __host__ __device__ uint32_t operator()(const uint64_t &x) const {
        return (uint32_t)(x >> 32);
    }
};

// functor to get the lower 32 bit
struct get_lower {
    __host__ __device__ uint32_t operator()(const uint64_t &x) const {
        return (uint32_t)(x & 0xFFFFFFFF);
    }
};

} // namespace hisa


#include <rmm/mr/device/pool_memory_resource.hpp>

inline void enable_rmm_allocator() {
    rmm::mr::cuda_memory_resource cuda_mr{};
    // rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr{
    //     &cuda_mr, 4 * 256 * 1024};
    // rmm::mr::set_current_device_resource(&mr);
}


#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <thrust/host_vector.h>

#include <assert.h>
// #include <cuda_runtime.h>
#include <iostream>

// use librmm
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <thrust/execution_policy.h>
#include <thrust/memory.h>

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

#define LAMBDA_TAG __device__
#define RAW_PTR data().get()

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


// #define EXE_POLICY thrust::device
// #define DEVICE_VECTOR thrust::device_vector
#define EXE_POLICY rmm::exec_policy()
#define DEVICE_VECTOR rmm::device_vector
#define HOST_VECTOR thrust::host_vector
#define DEFAULT_SET_HASH_MAP true
#define DEFAULT_LOAD_FACTOR 0.9

namespace vflog {
using internal_data_type = unsigned int;

using offset_type = unsigned long long;
using comp_range_t = unsigned long long;
using comp_pair_t = unsigned long long;

using device_data_t = DEVICE_VECTOR<internal_data_type>;
using device_indices_t = DEVICE_VECTOR<internal_data_type>;
using device_bitmap_t = DEVICE_VECTOR<bool>;

using device_internal_data_ptr = thrust::device_ptr<internal_data_type>;

// higher 32 bit is the postion in sorted indices, lower is offset
using device_ranges_t = DEVICE_VECTOR<comp_range_t>;
using device_pairs_t = DEVICE_VECTOR<comp_pair_t>;

using host_buf_ref_t = std::map<std::string, std::shared_ptr<device_indices_t>>;

using ptr_and_size_t =
    thrust::pair<thrust::pointer<internal_data_type, thrust::device_system_tag>,
                 std::ptrdiff_t>;
using thrust_buffer_ptr_t =
    thrust::pointer<internal_data_type, thrust::device_system_tag>;

inline unsigned long long __device__ __host__ compress_u32(unsigned int &a, unsigned int &b) {
    return ((unsigned long long)a << 32) | b;
}

// functor to get the higher 32 bit
struct get_higher {
    __host__ __device__ __host__ unsigned int operator()(const unsigned long long &x) const {
        return (unsigned int)(x >> 32);
    }
};

// functor to get the lower 32 bit
struct get_lower {
    __host__ __device__ __host__ unsigned int operator()(const unsigned long long &x) const {
        return (unsigned int)(x & 0xFFFFFFFF);
    }
};

enum IndexStrategy { EAGER, LAZY };

} // namespace vflog

#include <rmm/mr/device/pool_memory_resource.hpp>

inline void enable_rmm_allocator() {
    rmm::mr::cuda_memory_resource cuda_mr{};
    // rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr{
    //     &cuda_mr, 4 * 256 * 1024};
    // rmm::mr::set_current_device_resource(&mr);
}

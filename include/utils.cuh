
#pragma once

// #include "hisa.cuh"
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <thrust/host_vector.h>

#include <assert.h>

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

// a murmur hash function for device
__device__ __host__ inline
uint32_t murmur3_32(const uint32_t key) {
    uint32_t h = 0;
    uint32_t k = key;
    k *= 0xcc9e2d51;
    k = (k << 15) | (k >> 17);
    k *= 0x1b873593;
    h ^= k;
    h = (h << 13) | (h >> 19);
    h = h * 5 + 0xe6546b64;
    h ^= 4;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

// combine two uint32_t hash value to one uint32_t
__device__ __host__ inline
uint32_t combine_hash(uint32_t a, uint32_t b) {
    return a ^ (b + 0x9e3779b9 + (a << 6) + (a >> 2));
}

enum RelationVersion { DELTA, FULL, NEWT };

// #define EXE_POLICY thrust::device
// #define DEVICE_VECTOR thrust::device_vector
#define EXE_POLICY rmm::exec_policy()
#define DEVICE_VECTOR rmm::device_vector
#define HOST_VECTOR thrust::host_vector
#define DEFAULT_SET_HASH_MAP true
#define DEFAULT_LOAD_FACTOR 0.9

namespace vflog {
using internal_data_type = uint32_t;

using offset_type = uint64_t;
using comp_range_t = uint64_t;
using comp_pair_t = uint64_t;

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

inline uint64_t __device__ __host__ compress_u32(uint32_t &a, uint32_t &b) {
    return ((uint64_t)a << 32) | b;
}

// functor to get the higher 32 bit
struct get_higher {
    __host__ __device__ __host__ uint32_t operator()(const uint64_t &x) const {
        return (uint32_t)(x >> 32);
    }
};

// functor to get the lower 32 bit
struct get_lower {
    __host__ __device__ __host__ uint32_t operator()(const uint64_t &x) const {
        return (uint32_t)(x & 0xFFFFFFFF);
    }
};

enum IndexStrategy { EAGER, LAZY };

inline std::string index_strategy_to_string(IndexStrategy strategy) {
    switch (strategy) {
    case EAGER:
        return "EAGER";
    case LAZY:
        return "LAZY";
    default:
        return "UNKNOWN";
    }
}

// function to encode a 31-bit integer to a 32-bit unsigned integer
__device__ __host__ inline uint32_t i2d(int x) { return (x << 1) ^ (x >> 31); }

// function to decode a 32-bit unsigned integer to a 31-bit integer
__device__ __host__ inline int d2i(uint32_t x) {
    return (x >> 1) ^ (-(int)(x & 1));
}

inline std::string version_to_string(RelationVersion version) {
    switch (version) {
    case DELTA:
        return "DELTA";
    case FULL:
        return "FULL";
    case NEWT:
        return "NEWT";
    default:
        return "UNKNOWN";
    }
}

struct column_t {
    std::string rel;
    size_t idx;
    RelationVersion version;

    int frozen_idx = -1;

    column_t(std::string rel, size_t idx, RelationVersion version)
        : rel(rel), idx(idx), version(version) {}
    column_t(std::string rel, size_t idx, RelationVersion version,
             int frozen_idx)
        : rel(rel), idx(idx), version(version), frozen_idx(frozen_idx) {}

    bool is_frozen() { return frozen_idx != -1; }
    bool is_id() { return idx == UINT32_MAX; }

    std::string to_string() {
        std::string str = "column_t(\"" + rel + "\", " + std::to_string(idx) +
                          ", " + version_to_string(version) + ")";
        return str;
    }
};

inline __device__ __host__ bool
tuple_compare(uint32_t **full, uint32_t full_idx, uint32_t **newt,
              uint32_t newt_idx, int arity, int default_index_column) {
    if (full[default_index_column][full_idx] !=
        newt[default_index_column][newt_idx]) {
        return full[default_index_column][full_idx] <
               newt[default_index_column][newt_idx];
    }
    for (int i = 0; i < arity; i++) {
        if (i == default_index_column) {
            continue;
        }
        if (full[i][full_idx] != newt[i][newt_idx]) {
            return full[i][full_idx] < newt[i][newt_idx];
        }
    }
    return false;
}

inline __device__ __host__ bool tuple_eq(uint32_t **full, uint32_t full_idx,
                                         uint32_t **newt, uint32_t newt_idx,
                                         int arity, int default_index_column) {
    if (full[default_index_column][full_idx] !=
        newt[default_index_column][newt_idx]) {
        return false;
    }
    for (int i = 0; i < arity; i++) {
        if (i == default_index_column) {
            continue;
        }
        if (full[i][full_idx] != newt[i][newt_idx]) {
            return false;
        }
    }
    return true;
}

} // namespace vflog

#include <rmm/mr/device/pool_memory_resource.hpp>

#define ENABLE_RMM_POOL_MEMORY_RESOURCE                                        \
    rmm::mr::cuda_memory_resource cuda_mr{};                                   \
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr{           \
        &cuda_mr, 4 * 256 * 1024};                                             \
    rmm::mr::set_current_device_resource(&mr);

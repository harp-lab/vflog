#pragma once
#include "utils.cuh"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/optional.h>
#include <thrust/transform.h>

namespace fvlog {

//  32bit murmur3 hash function
inline __device__ unsigned int murmur3_32(unsigned int key) {
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key;
}

/**
 * a static GPU hash map, simplified version of what's in cuCollections
 */
class static_hash_map {

    float load_factor = 0.75;
    size_t capacity = 0;
    size_t size = 0;

    // internal data is on a thrust vector
    thrust::device_vector<unsigned int> keys;
    thrust::device_vector<unsigned long long> values;

  public:
    static_hash_map() = default;

    static_hash_map(size_t num) { resize(num); }

    void resize(size_t num) {
        size = 0;
        capacity = num / load_factor;
        keys.resize(capacity);
        values.resize(capacity);
        // clear the keys set to UINT32_MAX
        thrust::fill(keys.begin(), keys.end(), UINT32_MAX);
        thrust::fill(values.begin(), values.end(), UINT32_MAX);
    }

    template <typename PairIter> void insert(PairIter begin, PairIter end) {
        thrust::for_each(
            begin, end,
            [k_ptr = keys.data().get(), v_ptr = values.data().get(),
             capacity = capacity] __device__(auto pair) {
                unsigned int k = thrust::get<0>(pair);
                unsigned long long v = thrust::get<1>(pair);
                // printf("k: %llu , v: %lu \n", k, v);
                auto i = murmur3_32(k) % capacity;
                while (true) {
                    unsigned int old_k = atomicCAS(k_ptr + i, UINT32_MAX, k);
                    if (old_k == UINT32_MAX) {
                        atomicExch(v_ptr + i, v);
                        return;
                    }
                    if (old_k == k) {
                        return;
                    }
                    i = (i + 1) % capacity;
                }
            });
    }

    template <typename KeyIter, typename OutIter>
    void find(KeyIter begin, KeyIter end, OutIter out) {
        thrust::transform(
            begin, end, out,
            [k_ptr = keys.data().get(), v_ptr = values.data().get(),
             capacity = capacity] __device__(auto k) -> unsigned long long {
                auto i = murmur3_32(k) % capacity;
                while (true) {
                    auto found_k = k_ptr[i];
                    if (found_k == k) {
                        return v_ptr[i];
                    }
                    if (found_k == UINT32_MAX) {
                        return UINT32_MAX;
                    }
                    i = (i + 1) % capacity;
                }
            });
    }

    void clear() {
        thrust::fill(keys.begin(), keys.end(), UINT32_MAX);
        thrust::fill(values.begin(), values.end(), UINT32_MAX);
    }
};

} // namespace fvlog

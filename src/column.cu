
#include "column.cuh"

#include <thrust/binary_search.h>

namespace vflog {

// simple map
void GpuSimplMap::insert(device_data_t &keys, device_ranges_t &values) {
    // swap in
    this->keys.swap(keys);
    this->values.swap(values);
}

void GpuSimplMap::find(device_data_t &keys, device_ranges_t &result) {
    // keys is the input, values is the output
    result.resize(keys.size());
    // device_data_t found_keys(keys.size());

    thrust::transform(
        keys.begin(), keys.end(), result.begin(),
        [map_keys = this->keys.data().get(), map_vs = this->values.data().get(),
         ksize = this->keys.size()] __device__(internal_data_type key)
            -> comp_range_t {
            auto it = thrust::lower_bound(thrust::seq, map_keys,
                                          map_keys + ksize, key);
            return map_vs[it - map_keys];
        });
}

// multi_hisa
void VerticalColumnGpu::clear_unique_v() {
    if (!unique_v_map) {
        // delete unique_v_map;
        unique_v_map = nullptr;
    }
}

VerticalColumnGpu::~VerticalColumnGpu() { clear_unique_v(); }

} // namespace hisa
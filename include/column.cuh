
#pragma once

#include "utils.cuh"
#include <cuco/dynamic_map.cuh>
#include <cuco/static_map.cuh>

namespace vflog {

// a simple Device Map, its a wrapper of the device_vector
struct GpuSimplMap {
    device_data_t keys;
    device_ranges_t values;

    GpuSimplMap() = default;

    // bulk insert
    void insert(device_data_t &keys, device_ranges_t &values);

    // bulk find
    void find(device_data_t &keys, device_ranges_t &result);
};

// higher 32 bit is the value, lower is offset in data
// using index_value = uint64_t;
// using Map = std::unordered_map<internal_data_type, offset_type>;
using GpuMap = cuco::static_map<internal_data_type, comp_range_t>;
using GpuMapReadRef = GpuMap::ref_type<cuco::find_tag>;
// using GpuMap = cuco::dynamic_map<internal_data_type, comp_range_t>;
using GpuMapPair = cuco::pair<internal_data_type, comp_range_t>;

// using GpuMap = bght::bcht<internal_data_type, comp_range_t>;
// using GpuMapPair = bght::pair<internal_data_type, comp_range_t

// CSR stype column entryunique values aray in the column, sharing the same
// prefix
struct VerticalColumnGpu {

    // FIXME: remove this, this is redundant
    // all unique values in the column, sharing the same prefix
    device_data_t unique_v;
    // a mapping from the unique value to the range of tuple share the same
    // value in the next column
    std::shared_ptr<GpuMap> unique_v_map = nullptr;
    // std::unique_ptr<GpuMap> unique_v_map = nullptr;

    GpuSimplMap unique_v_map_simp;

    device_data_t sorted_indices;
    // thrust::device_vector<internal_data_type> raw_data;
    // device_internal_data_ptr raw_data = nullptr;
    size_t raw_offset = 0;

    VerticalColumnGpu() = default;

    size_t size() const { return raw_size; }

    bool indexed = false;

    size_t raw_size = 0;

    size_t column_idx = 0;

    IndexStrategy index_strategy = IndexStrategy::EAGER;

    bool use_real_map = DEFAULT_SET_HASH_MAP;

    void clear_unique_v();

    ~VerticalColumnGpu();
};

// a vertical column in CPU memory
struct VetricalColumnCpu {
    HOST_VECTOR<internal_data_type> full_sorted_indices;
    HOST_VECTOR<internal_data_type> delta_sorted_indices;
    HOST_VECTOR<internal_data_type> newt_sorted_indices;
    uint32_t full_size = 0;
    uint32_t delta_size = 0;
    uint32_t newt_size = 0;
    HOST_VECTOR<internal_data_type> raw_data;

    size_t full_head_offset = 0;
    size_t delta_head_offset = 0;
    size_t newt_head_offset = 0;
};

} // namespace vflog

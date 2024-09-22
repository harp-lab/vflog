
#pragma once

#include "buffer.cuh"
#include "utils.cuh"
#include <cuco/dynamic_map.cuh>
#include <cuco/static_map.cuh>
#include <cuda/functional>

#define CREATE_V_MAP(uniq_size)                                                \
    std::make_unique<vflog::GpuMap>(                                           \
        uniq_size, DEFAULT_LOAD_FACTOR,                                        \
        cuco::empty_key<vflog::internal_data_type>{UINT32_MAX},                \
        cuco::empty_value<vflog::offset_type>{UINT32_MAX})
#define HASH_NAMESPACE cuco

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
using GpuMap = cuco::static_map<
    internal_data_type, comp_range_t, cuco::extent<std::size_t>,
    cuda::thread_scope_device, thrust::equal_to<internal_data_type>,
    cuco::linear_probing<4, cuco::default_hash_function<internal_data_type>>>;
// using GpuMapReadRef = GpuMap::ref_type<cuco::find_tag>;
// using GpuMap = cuco::dynamic_map<internal_data_type, comp_range_t>;
using GpuMapPair = cuco::pair<internal_data_type, comp_range_t>;

// CSR stype column entryunique values aray in the column, sharing the same
// prefix
struct VerticalColumn {

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
    internal_data_type *raw_data = nullptr;
    size_t raw_offset = 0;

    VerticalColumn() = default;

    size_t size() const { return raw_size; }

    bool indexed = false;

    size_t raw_size = 0;

    size_t column_idx = 0;

    IndexStrategy index_strategy = IndexStrategy::LAZY;

    bool use_real_map = DEFAULT_SET_HASH_MAP;

    void clear_unique_v() {
        if (unique_v_map) {
            // delete unique_v_map;
            unique_v_map = nullptr;
        }
    }

    ~VerticalColumn() { clear_unique_v(); }

    template <typename IteratorFrom, typename IteratorTo>
    void map_find(IteratorFrom from_start, IteratorFrom from_end,
                  IteratorTo to) {
        if (unique_v_map) {
            unique_v_map->find(from_start, from_end, to);
        } else {
            throw std::runtime_error("non-map search  not implemented");
        }
    }

    template <typename IteratorFrom, typename IteratorTo, typename IteratorRes>
    void map_contains(IteratorFrom from_start, IteratorFrom from_end,
                      IteratorRes to) {
        if (unique_v_map) {
            unique_v_map->contains(from_start, from_end, to);
        } else {
            throw std::runtime_error("non-map search not implemented");
        }
    }

    template <typename IteratorInput, typename IteratorStencil,
              typename IteratorRes>
    void map_contains_if(IteratorInput from_start, IteratorInput from_end,
                         IteratorStencil stencil_start, IteratorRes to) {
        if (unique_v_map) {
            unique_v_map->contains_if(from_start, from_end, stencil_start,
                                      thrust::identity<bool>(), to);
        } else {
            throw std::runtime_error("non-map search not implemented");
        }
    }

    void map_insert(device_data_t &raw_data, size_t uniq_size,
                    d_buffer_ptr &buffer) {
        if (unique_v_map) {
            // columns[i].unique_v_map->reserve(uniq_size);
            if (unique_v_map->size() > uniq_size + 1) {
                unique_v_map->clear();
            } else {
                unique_v_map = nullptr;
                unique_v_map = CREATE_V_MAP(uniq_size);
            }
        } else {
            unique_v_map = CREATE_V_MAP(uniq_size);
        }
        auto insertpair_begin = thrust::make_transform_iterator(
            thrust::make_counting_iterator<uint32_t>(0),
            cuda::proclaim_return_type<GpuMapPair>(
                [uniq_offset_raw = buffer->data(), uniq_size,
                 sorted_idx = sorted_indices.RAW_PTR,
                 raw_head = raw_data.RAW_PTR + raw_offset,
                 column_size = raw_size] LAMBDA_TAG(auto &idx) {
                    // compute the offset by idx+1 - idx, if idx is the last
                    // one, then the offset is the size of the column - idx
                    auto val = raw_head[sorted_idx[uniq_offset_raw[idx]]];
                    auto range_size =
                        idx == uniq_size - 1
                            ? column_size - uniq_offset_raw[idx]
                            : uniq_offset_raw[idx + 1] - uniq_offset_raw[idx];
                    return HASH_NAMESPACE::make_pair(
                        val,
                        (static_cast<uint64_t>(uniq_offset_raw[idx]) << 32) +
                            (static_cast<uint64_t>(range_size)));
                }));
        unique_v_map->insert(insertpair_begin, insertpair_begin + uniq_size);
    }

    void clear_map() {
        if (unique_v_map) {
            unique_v_map->clear();
        } else {
            throw std::runtime_error("non-map clear not implemented");
        }
    }
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

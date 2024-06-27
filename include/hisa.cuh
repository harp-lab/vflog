/**
 */

#pragma once

#include <cuda/functional>

#include "utils.cuh"
#include <cstdint>
#include <cuco/dynamic_map.cuh>
#include <cuco/static_map.cuh>
#include <cuda/std/chrono>
#include <iostream>
#include <memory>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

#define CREATE_V_MAP(uniq_size)                                                \
    std::make_unique<hisa::GpuMap>(                                            \
        uniq_size, DEFAULT_LOAD_FACTOR,                                        \
        cuco::empty_key<hisa::internal_data_type>{UINT32_MAX},                 \
        cuco::empty_value<hisa::offset_type>{UINT32_MAX})
#define HASH_NAMESPACE cuco

namespace hisa {

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
// using GpuMap = cuco::dynamic_map<internal_data_type, comp_range_t>;
using GpuMapPair = cuco::pair<internal_data_type, comp_range_t>;

// using GpuMap = bght::bcht<internal_data_type, comp_range_t>;
// using GpuMapPair = bght::pair<internal_data_type, comp_range_t>;

enum IndexStrategy { EAGER, LAZY };

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
    device_internal_data_ptr raw_data = nullptr;

    VerticalColumnGpu() = default;

    size_t size() const { return raw_size; }

    bool indexed = false;

    size_t raw_size = 0;

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

struct multi_hisa {
    int arity;

    using VersionedColumns = HOST_VECTOR<VerticalColumnGpu>;
    VersionedColumns full_columns;
    VersionedColumns delta_columns;
    VersionedColumns newt_columns;

    offset_type total_tuples;

    multi_hisa(int arity);

    // HOST_VECTOR<int> indexed_columns;
    uint64_t hash_time = 0;
    uint64_t dedup_time = 0;
    uint64_t sort_time = 0;
    uint64_t load_time = 0;

    bool indexed = false;

    uint32_t full_size = 0;
    uint32_t delta_size = 0;
    uint32_t newt_size = 0;

    size_t default_index_column = 0;

    // data array, full/delta/newt all in one
    HOST_VECTOR<device_data_t> data;
    // lexical order of the data in the full
    // thrust::device_vector<uint32_t> full_lexical_order;

    // load data from CPU Memory to Full, this misa must be empty
    void
    init_load_vectical(HOST_VECTOR<HOST_VECTOR<internal_data_type>> &tuples,
                       size_t tuple_size);

    //
    void load_column_cpu(VetricalColumnCpu &columns_cpu, int column_idx);

    /**
     * @brief Build index for all columns in the relation
     * this will only build EAGER index
     */
    void build_index_all(RelationVersion version, bool sorted = false);

    /**
     * @brief Build index for a specific column in the relation,
     *        even if its lazy index or already indexed, this will force
     *       to build the index
     */
    void force_column_index(RelationVersion version, int column_idx,
                            bool rebuild = true, bool sorted = false);

    // deduplicate the data in the newt
    void deduplicate();

    // load newt to delta, this will clear the newt, the delta must be empty
    // before this operation
    void new_to_delta();

    // this will
    // 1. clear the index of delta
    // 2. merge newt to full
    // 3. create the index of newt, rename to delta
    // 4. swap the newt and delta
    void persist_newt();

    void print_all(bool sorted = false);

    void fit();

    void clear();

    void allocate_newt(size_t size);

    // get reference to the different version of columns
    VersionedColumns &get_versioned_columns(RelationVersion version) {
        switch (version) {
        case FULL:
            return full_columns;
        case DELTA:
            return delta_columns;
        case NEWT:
            return newt_columns;
        default:
            return full_columns;
        }
    }

    uint32_t get_versioned_size(RelationVersion version) {
        switch (version) {
        case FULL:
            return full_size;
        case DELTA:
            return delta_size;
        case NEWT:
            return newt_size;
        default:
            return full_size;
        }
    }

    // void deduplicate();
    uint32_t get_total_tuples() const { return total_tuples; }

  private:
    void build_index(VerticalColumnGpu &column, device_data_t &unique_offset,
                     device_data_t &unique_diff, bool sorted = false);
};

// filter the matched_pairs (id_1, id2) with respect to column2,
// only keep the matched pair
void column_match(VerticalColumnGpu &column1, VerticalColumnGpu &column2,
                  device_pairs_t &matched_pair);
void column_match(VerticalColumnGpu &column1, VerticalColumnGpu &column2,
                  device_indices_t &column1_indices,
                  device_indices_t &column2_indices,
                  DEVICE_VECTOR<bool> &unmatched);

void column_join(VerticalColumnGpu &inner_column,
                 VerticalColumnGpu &outer_column,
                 device_data_t &outer_tuple_indices,
                 device_pairs_t &matched_indices);

void column_join(VerticalColumnGpu &inner_column,
                 VerticalColumnGpu &outer_column,
                 device_indices_t &outer_tuple_indices,
                 device_indices_t &matched_indices,
                 DEVICE_VECTOR<bool> &unmatched);

} // namespace hisa

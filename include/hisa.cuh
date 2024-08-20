/**
 */

#pragma once

#include <cuda/functional>

#include "buffer.cuh"
#include "column.cuh"
// #include "utils.cuh"
#include <cstdint>
// #include <functional>

#include <cuda/std/chrono>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace vflog {

struct multi_hisa {
    size_t uid = 0;

    int arity;

    using VersionedColumns = HOST_VECTOR<VerticalColumn>;
    VersionedColumns full_columns;
    VersionedColumns delta_columns;
    VersionedColumns newt_columns;

    offset_type total_tuples = 0;

    offset_type capacity = 0;

    multi_hisa(int arity, d_buffer_ptr buf = nullptr, size_t default_idx = 0);

    // contstruct with init data
    multi_hisa(int arity, const char *filename, d_buffer_ptr buf = nullptr,
               size_t default_idx = 0);

    // HOST_VECTOR<int> indexed_columns;
    uint64_t hash_time = 0;
    uint64_t dedup_time = 0;
    uint64_t sort_time = 0;
    uint64_t load_time = 0;
    uint64_t merge_time = 0;

    bool indexed = false;
    bool unique_gather_flag = false;

    uint32_t full_size = 0;
    uint32_t delta_size = 0;
    uint32_t newt_size = 0;

    size_t default_index_column = 0;

    // data array, full/delta/newt all in one
    HOST_VECTOR<device_data_t> data;
    // lexical order of the data in the full
    // thrust::device_vector<uint32_t> full_lexical_order;

    d_buffer_ptr buffer = nullptr;
    device_data_t data_buffer;

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
    void newt_self_deduplicate();

    // deduplicate the newt data in the full
    void newt_full_deduplicate();

    // this will
    // 1. clear the index of delta
    // 2. merge newt to full
    // 3. create the index of newt, rename to delta
    // 4. swap the newt and delta
    void persist_newt(bool dedup = true);

    void diff(multi_hisa &other, RelationVersion version,
              device_indices_t &diff_indices);

    void print_all(bool sorted = false);

    void print_raw_data(RelationVersion version);

    void fit();

    void clear();

    void allocate_newt(size_t size);

    void set_versioned_size(RelationVersion version, uint32_t size) {
        switch (version) {
        case FULL:
            full_size = size;
            break;
        case DELTA:
            delta_size = size;
            break;
        case NEWT:
            newt_size = size;
            break;
        default:
            break;
        }
    }

    internal_data_type *get_raw_data_ptrs(RelationVersion version,
                                          int column_idx) {
        return data[version].RAW_PTR +
               get_versioned_columns(version)[column_idx].raw_offset;
    }

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

    // void newt_self_deduplicate();
    uint32_t get_total_tuples() const { return total_tuples; }

    offset_type get_capacity() const { return capacity; }

    /**
     * @brief Build index for a specific column in the relation
     */
    void build_index(VerticalColumn &column, device_data_t &unique_offset,
                     bool sorted = false);

    void set_index_startegy(int column_idx, RelationVersion version,
                            IndexStrategy strategy) {
        auto &column = get_versioned_columns(version)[column_idx];
        column.index_strategy = strategy;
    }

    void set_default_index_column(int column_idx) {
        default_index_column = column_idx;
        // set FULL of default index column to EAGER
        set_index_startegy(column_idx, FULL, EAGER);
    }

    void print_stats();
};

struct column_ref {
    std::reference_wrapper<multi_hisa> relation;
    RelationVersion version;
    size_t index;

    // std::reference_wrapper<device_bitmap_t> selected;

    column_ref(multi_hisa &rel, RelationVersion ver, size_t idx)
        : relation(rel), version(ver), index(idx) {}
};

} // namespace vflog

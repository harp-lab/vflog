/**
 */

#pragma once

#include <cuda/functional>

#include "buffer.cuh"
#include "column.cuh"
#include "utils.cuh"
#include <cstdint>
// #include <functional>

#include <cuda/std/chrono>
#include <string>
#include <sys/types.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace vflog {

enum split_mode_type { SPLIT_ITER, SPLIT_SIZE, SPLIT_NONE };

struct ClusteredIndex {
    std::vector<int> column_indices;

    device_data_t sorted_indices;

    int get_prime_column() const { return column_indices[0]; }
};

struct multi_hisa {
    size_t uid = 0;
    std::string name;

    // uint32_t max_tuples = 327'680'000;
    uint32_t max_tuples = UINT32_MAX;
    uint32_t max_splits = 4;

    split_mode_type split_mode = SPLIT_NONE;
    uint32_t split_iter_count = 0;

    int arity;

    using VersionedColumns = HOST_VECTOR<VerticalColumn>;
    VersionedColumns full_columns;
    VersionedColumns delta_columns;
    VersionedColumns newt_columns;

    std::vector<ClusteredIndex> clustered_indices_full;
    std::vector<ClusteredIndex> clustered_indices_newt;

    offset_type total_tuples = 0;

    offset_type capacity = 0;

    multi_hisa(std::string name, int arity, d_buffer_ptr buf = nullptr,
               size_t default_idx = 0);

    // contstruct with init data
    multi_hisa(std::string name, int arity, const char *filename,
               d_buffer_ptr buf = nullptr, size_t default_idx = 0);

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

    int iteration = 0;

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

    void sort_newt_clustered_index();
    void persist_newt_clustered_index();

    // this will
    // 1. clear the index of delta
    // 2. merge newt to full
    // 3. create the index of newt, rename to delta
    // 4. swap the newt and delta
    void persist_newt(bool dedup = true);

    bool tuple_exists(std::vector<internal_data_type> &tuple,
                      RelationVersion version);

    void diff(multi_hisa &other, RelationVersion version,
              device_indices_t &diff_indices);

    void print_all(bool sorted = false);

    void print_raw_data(RelationVersion version);

    void fit();

    void clear();

    void allocate_newt(size_t size);

    void split_when(uint32_t size, uint32_t max_splits) {
        max_tuples = size;
        max_splits = max_splits;
        split_mode = SPLIT_SIZE;
    }

    void split_iter(uint32_t iter, uint32_t max_splits) {
        split_iter_count = iter;
        split_mode = SPLIT_ITER;
        max_splits = max_splits;
    }

    void stop_split() {
        if (split_mode == SPLIT_SIZE) {
            max_tuples = UINT32_MAX;
        } else if (split_mode == SPLIT_ITER) {
            split_iter_count = UINT32_MAX;
        }
    }

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

    void copy_meta(multi_hisa &other) {
        arity = other.arity;
        max_tuples = other.max_tuples;
        max_splits = other.max_splits;
        default_index_column = other.default_index_column;
        split_mode = other.split_mode;
        split_iter_count = other.split_iter_count;
        for (int i = 0; i < arity; i++) {
            full_columns[i].index_strategy =
                other.full_columns[i].index_strategy;
            full_columns[i].raw_data = other.full_columns[i].raw_data;
            delta_columns[i].index_strategy =
                other.delta_columns[i].index_strategy;
            delta_columns[i].raw_data = other.delta_columns[i].raw_data;
            newt_columns[i].index_strategy =
                other.newt_columns[i].index_strategy;
            newt_columns[i].raw_data = other.newt_columns[i].raw_data;
        }
    }

    void add_clustered_index(std::vector<int> column_indices) {
        ClusteredIndex ci;
        ci.column_indices = column_indices;
        clustered_indices_full.push_back(ci);
    }

    void print_stats();

    void inc_iter() { iteration++; }
};

} // namespace vflog

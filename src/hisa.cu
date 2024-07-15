
#include "hisa.cuh"
#include <iostream>

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#define CREATE_V_MAP(uniq_size)                                                \
    std::make_unique<vflog::GpuMap>(                                           \
        uniq_size, DEFAULT_LOAD_FACTOR,                                        \
        cuco::empty_key<vflog::internal_data_type>{UINT32_MAX},                \
        cuco::empty_value<vflog::offset_type>{UINT32_MAX})
#define HASH_NAMESPACE cuco

namespace vflog {

multi_hisa::multi_hisa(int arity, d_buffer_ptr buffer, size_t default_idx) {
    this->arity = arity;
    newt_size = 0;
    full_size = 0;
    delta_size = 0;
    full_columns.resize(arity);
    delta_columns.resize(arity);
    newt_columns.resize(arity);
    data.resize(arity);
    if (buffer) {
        this->buffer = buffer;
    } else {
        this->buffer = std::make_shared<d_buffer>(40960);
    }

    for (int i = 0; i < arity; i++) {
        full_columns[i].column_idx = i;
        delta_columns[i].column_idx = i;
        newt_columns[i].column_idx = i;
    }
    set_default_index_column(default_idx);
}

void multi_hisa::init_load_vectical(
    HOST_VECTOR<HOST_VECTOR<internal_data_type>> &tuples, size_t rows) {
    auto load_start = std::chrono::high_resolution_clock::now();
    auto total_tuples = tuples[0].size();
    for (int i = 0; i < arity; i++) {
        // extract the i-th column
        // thrust::device_vector<internal_data_type> column_data(total_tuples);
        data[i].resize(total_tuples);
        cudaMemcpy(data[i].data().get(), tuples[i].data(),
                   tuples[i].size() * sizeof(internal_data_type),
                   cudaMemcpyHostToDevice);
        // save columns raw
    }
    this->total_tuples = total_tuples;
    this->newt_size = total_tuples;
    // set newt
    for (int i = 0; i < arity; i++) {
        newt_columns[i].raw_size = total_tuples;
    }
    auto load_end = std::chrono::high_resolution_clock::now();
    this->load_time += std::chrono::duration_cast<std::chrono::microseconds>(
                           load_end - load_start)
                           .count();
}

void multi_hisa::allocate_newt(size_t size) {
    auto old_size = capacity;
    if (total_tuples + size < capacity) {
        // std::cout << "no need to allocate newt" << std::endl;
        // return;  
        size = 0;
    }
    // compute offset of each version
    auto new_capacity = old_size + size;

    for (int i = 0; i < arity; i++) {
        data[i].resize(old_size + size);
    }
    capacity = old_size + size;

    // newt_size += size;
    // for (int i = 0; i < arity; i++) {
    //     newt_columns[i].raw_data = data[i].data() + old_size;
    // }
}

void multi_hisa::load_column_cpu(VetricalColumnCpu &columns_cpu,
                                 int column_idx) {
    auto total_tuples = columns_cpu.raw_data.size();
    capacity = total_tuples;
    data[column_idx].resize(total_tuples);
    cudaMemcpy(data[column_idx].data().get(), columns_cpu.raw_data.data(),
               columns_cpu.raw_data.size() * sizeof(internal_data_type),
               cudaMemcpyHostToDevice);
    this->total_tuples = total_tuples;
    this->newt_size = columns_cpu.newt_size;
    this->full_size = columns_cpu.full_size;
    this->delta_size = columns_cpu.delta_size;
    if (columns_cpu.full_size == 0) {
        return;
    }
    // set ptr
    full_columns[column_idx].raw_offset = 0;
    delta_columns[column_idx].raw_offset = columns_cpu.delta_head_offset;
    newt_columns[column_idx].raw_offset = columns_cpu.newt_head_offset;
    // copy sorted indices
    if (columns_cpu.full_size != 0) {
        full_columns[column_idx].sorted_indices.resize(columns_cpu.full_size);
        full_columns[column_idx].raw_size = columns_cpu.full_size;
        cudaMemcpy(full_columns[column_idx].sorted_indices.data().get(),
                   columns_cpu.full_sorted_indices.data(),
                   columns_cpu.full_sorted_indices.size() *
                       sizeof(internal_data_type),
                   cudaMemcpyHostToDevice);
    }
    if (columns_cpu.delta_size != 0) {
        delta_columns[column_idx].sorted_indices.resize(columns_cpu.delta_size);
        delta_columns[column_idx].raw_size = columns_cpu.delta_size;
        cudaMemcpy(delta_columns[column_idx].sorted_indices.data().get(),
                   columns_cpu.delta_sorted_indices.data(),
                   columns_cpu.delta_sorted_indices.size() *
                       sizeof(internal_data_type),
                   cudaMemcpyHostToDevice);
    }
    if (columns_cpu.newt_size != 0) {
        newt_columns[column_idx].sorted_indices.resize(columns_cpu.newt_size);
        newt_columns[column_idx].raw_size = columns_cpu.newt_size;
        cudaMemcpy(newt_columns[column_idx].sorted_indices.data().get(),
                   columns_cpu.newt_sorted_indices.data(),
                   columns_cpu.newt_sorted_indices.size() *
                       sizeof(internal_data_type),
                   cudaMemcpyHostToDevice);
    }
}

void multi_hisa::print_all(bool sorted) {
    // print all columns in full
    HOST_VECTOR<internal_data_type> column(total_tuples);
    HOST_VECTOR<internal_data_type> unique_value(total_tuples);
    for (int i = 0; i < arity; i++) {
        thrust::copy(data[i].begin() + full_columns[i].raw_offset,
                     data[i].begin() + full_columns[i].raw_offset +
                         full_columns[i].raw_size,
                     column.begin());
        std::cout << "column data " << i << " " << total_tuples << ":\n";
        for (int j = 0; j < column.size(); j++) {
            std::cout << column[j] << " ";
        }
        std::cout << std::endl;
        std::cout << "unique values " << i << " "
                  << full_columns[i].unique_v.size() << ":\n";
        unique_value.resize(full_columns[i].unique_v.size());
        thrust::copy(full_columns[i].unique_v.begin(),
                     full_columns[i].unique_v.end(), unique_value.begin());
        for (int j = 0; j < full_columns[i].unique_v.size(); j++) {
            std::cout << unique_value[j] << " ";
        }
        std::cout << std::endl;
    }
}

void multi_hisa::print_raw_data(RelationVersion ver) {
    std::printf("print raw data\n");
    HOST_VECTOR<HOST_VECTOR<internal_data_type>> columns_host(arity);
    for (int i = 0; i < arity; i++) {
        columns_host[i].resize(get_versioned_size(ver));
        cudaMemcpy(columns_host[i].data(),
                   data[i].data().get() +
                       get_versioned_columns(ver)[i].raw_offset,
                   get_versioned_size(ver) * sizeof(internal_data_type),
                   cudaMemcpyDeviceToHost);
    }
    // radix sort host
    thrust::host_vector<internal_data_type> column_host(
        get_versioned_size(ver));
    thrust::host_vector<internal_data_type> sorted_indices_host(
        get_versioned_size(ver));
    thrust::sequence(sorted_indices_host.begin(), sorted_indices_host.end());
    // for (int i = arity - 1; i >= 0; i--) {
    //     auto &column = columns_host[i];
    //     thrust::gather(sorted_indices_host.begin(),
    //     sorted_indices_host.end(),
    //                    column.begin(), column_host.begin());
    //     thrust::stable_sort_by_key(column_host.begin(), column_host.end(),
    //                                sorted_indices_host.begin());
    // }
    // permute the columns
    for (int i = 0; i < arity; i++) {
        thrust::gather(sorted_indices_host.begin(), sorted_indices_host.end(),
                       columns_host[i].begin(), column_host.begin());
        thrust::copy(column_host.begin(), column_host.end(),
                     columns_host[i].begin());
    }

    for (size_t i = 0; i < get_versioned_size(ver); i++) {
        for (int j = 0; j < arity; j++) {
            std::cout << columns_host[j][i] << " ";
        }
        std::cout << std::endl;
    }
}

void multi_hisa::build_index(VerticalColumnGpu &column,
                             device_data_t &unique_offset, bool sorted) {
    auto sorte_start = std::chrono::high_resolution_clock::now();
    auto &raw_data = data[column.column_idx];
    auto raw_ver_head = raw_data.begin() + column.raw_offset;
    size_t uniq_size = 0;
    if (!sorted) {
        // std::cout << ">>>>>> sort column while build index " <<
        // column.column_idx << std::endl;
        device_data_t column_data(column.raw_size);
        column.sorted_indices.resize(column.raw_size);
        thrust::sequence(DEFAULT_DEVICE_POLICY, column.sorted_indices.begin(),
                         column.sorted_indices.end());
        // sort all values in the column
        thrust::copy(DEFAULT_DEVICE_POLICY, raw_ver_head,
                     raw_ver_head + column.raw_size, column_data.begin());
        // if (i != 0) {
        thrust::stable_sort_by_key(DEFAULT_DEVICE_POLICY, column_data.begin(),
                                   column_data.end(),
                                   column.sorted_indices.begin());
        // }
        // count unique
        uniq_size = thrust::unique_count(
            DEFAULT_DEVICE_POLICY, column_data.begin(), column_data.end());
        // buffer_ptr_and_size =
        // thrust::get_temporary_buffer<internal_data_type>(
        //     device_sys, uniq_size);
        // offset_buffer_ptr = buffer_ptr_and_size.first;
        buffer->reserve(uniq_size);
        auto uniq_end = thrust::unique_by_key_copy(
            DEFAULT_DEVICE_POLICY, column_data.begin(), column_data.end(),
            thrust::make_counting_iterator<uint32_t>(0),
            thrust::make_discard_iterator(), buffer->data());
        // uniq_size = uniq_end.second - unique_offset.begin();
    } else {
        if (unique_gather_flag) {
            uniq_size = thrust::unique_count(
                DEFAULT_DEVICE_POLICY,
                thrust::make_permutation_iterator(
                    raw_ver_head, column.sorted_indices.begin()),
                thrust::make_permutation_iterator(raw_ver_head,
                                                  column.sorted_indices.end()));
            buffer->reserve(uniq_size);
        } else {
            buffer->reserve(column.raw_size);
        }
        auto uniq_end = thrust::unique_by_key_copy(
            DEFAULT_DEVICE_POLICY,
            thrust::make_permutation_iterator(raw_ver_head,
                                              column.sorted_indices.begin()),
            thrust::make_permutation_iterator(raw_ver_head,
                                              column.sorted_indices.end()),
            thrust::make_counting_iterator<uint32_t>(0),
            thrust::make_discard_iterator(), buffer->data());
        if (!unique_gather_flag) {
            uniq_size = uniq_end.second - buffer->data();
        }
    }
    auto sort_end = std::chrono::high_resolution_clock::now();
    this->sort_time += std::chrono::duration_cast<std::chrono::microseconds>(
                           sort_end - sorte_start)
                           .count();

    // update map
    auto start = std::chrono::high_resolution_clock::now();
    if (column.use_real_map) {
        if (column.unique_v_map) {
            // columns[i].unique_v_map->reserve(uniq_size);
            if (column.unique_v_map->size() > uniq_size + 1) {
                column.unique_v_map->clear();
            } else {
                column.unique_v_map = nullptr;
                column.unique_v_map = CREATE_V_MAP(uniq_size);
            }
        } else {
            column.unique_v_map = CREATE_V_MAP(uniq_size);
        }
        auto insertpair_begin = thrust::make_transform_iterator(
            thrust::make_counting_iterator<uint32_t>(0),
            cuda::proclaim_return_type<GpuMapPair>(
                [uniq_offset_raw = buffer->data(), uniq_size,
                 sorted_idx = column.sorted_indices.data().get(),
                 raw_head = raw_data.data().get() + column.raw_offset,
                 column_size = column.raw_size] __device__(auto &idx) {
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
        column.unique_v_map->insert(insertpair_begin,
                                    insertpair_begin + uniq_size);
    } else {
        throw std::runtime_error("not implemented");
        // device_ranges_t ranges(uniq_size);
        // // compress offset and diff to u64 ranges
        // thrust::transform(DEFAULT_DEVICE_POLICY, unique_offset.begin(),
        //                   unique_offset.end(), unique_diff.begin(),
        //                   ranges.begin(),
        //                   [] __device__(auto &offset, auto &diff) -> uint64_t
        //                   {
        //                       return (static_cast<uint64_t>(offset) << 32) +
        //                              (static_cast<uint64_t>(diff));
        //                   });
        // column.unique_v_map_simp.insert(uniq_val, ranges);
        // // std::cout << "ranges size: " << ranges.size() << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    this->hash_time +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    column.indexed = true;
}

void multi_hisa::force_column_index(RelationVersion version, int i,
                                    bool rebuild, bool sorted) {
    auto &columns = get_versioned_columns(version);
    if (columns[i].indexed && !rebuild) {
        return;
    }
    device_data_t unique_offset;
    build_index(columns[i], unique_offset, sorted);
}

void multi_hisa::build_index_all(RelationVersion version, bool sorted) {
    auto &columns = get_versioned_columns(version);
    auto version_size = get_versioned_size(version);
    if (version_size == 0) {
        return;
    }

    device_data_t unique_offset;
    for (size_t i = 0; i < arity; i++) {
        if (columns[i].index_strategy == IndexStrategy::EAGER) {
            // std::cout << "build index " << i << std::endl;
            if (i != default_index_column) {
                build_index(columns[i], unique_offset, sorted);
            } else {
                build_index(columns[i], unique_offset, true);
            }
            // std::cout << "indexed map size : " <<
            // columns[i].unique_v_map->size() << std::endl;
        }
    }
    indexed = true;
}

void multi_hisa::deduplicate() {
    // sort the raw data of each column in newt
    auto start = std::chrono::high_resolution_clock::now();
    VersionedColumns &columns = newt_columns;
    auto version_size = newt_size;
    // radix sort the raw data of each column
    device_data_t sorted_indices(version_size);
    // std::cout << "deduplicate newt size: " << version_size << std::endl;
    thrust::sequence(DEFAULT_DEVICE_POLICY, sorted_indices.begin(),
                     sorted_indices.end());
    // device_data_t tmp_raw(version_size);
    buffer->reserve(version_size);
    auto tmp_raw_ptr = buffer->data();
    for (int i = arity - 1; i >= 0; i--) {
        if (i == default_index_column) {
            continue;
        }
        auto &column = columns[i];
        auto column_data = data[i].begin() + column.raw_offset;
        // gather the column data
        thrust::gather(DEFAULT_DEVICE_POLICY, sorted_indices.begin(),
                       sorted_indices.end(), column_data, tmp_raw_ptr);
        thrust::stable_sort_by_key(DEFAULT_DEVICE_POLICY, tmp_raw_ptr,
                                   tmp_raw_ptr + version_size,
                                   sorted_indices.begin());
    }

    // sort the default index column
    auto &column = columns[default_index_column];
    auto column_data = data[default_index_column].begin() + column.raw_offset;
    thrust::gather(DEFAULT_DEVICE_POLICY, sorted_indices.begin(),
                   sorted_indices.end(), column_data,
                   tmp_raw_ptr); // gather the column data
    thrust::stable_sort_by_key(DEFAULT_DEVICE_POLICY, tmp_raw_ptr,
                               tmp_raw_ptr + version_size,
                               sorted_indices.begin());

    device_bitmap_t dup_flags(version_size, false);
    // check duplicates tuple
    DEVICE_VECTOR<internal_data_type *> all_col_ptrs(arity);
    for (int i = 0; i < arity; i++) {
        all_col_ptrs[i] = data[i].data().get() + columns[i].raw_offset;
    }
    thrust::transform(
        thrust::make_counting_iterator<uint32_t>(0),
        thrust::make_counting_iterator<uint32_t>(dup_flags.size()),
        dup_flags.begin(),
        [all_col_ptrs = all_col_ptrs.data(), total = dup_flags.size(),
         sorted_indices = sorted_indices.data().get(),
         arity = arity] __device__(uint32_t i) -> bool {
            if (i == 0) {
                return false;
            } else {
                for (int j = 0; j < arity; j++) {
                    if (all_col_ptrs[j][sorted_indices[i]] !=
                        all_col_ptrs[j][sorted_indices[i - 1]]) {
                        return false;
                    }
                }
                return true;
            }
        });
    // filter
    auto new_sorted_indices_end = thrust::remove_if(
        DEFAULT_DEVICE_POLICY, sorted_indices.begin(), sorted_indices.end(),
        dup_flags.begin(), thrust::identity<bool>());
    auto new_sorted_indices_size =
        new_sorted_indices_end - sorted_indices.begin();
    sorted_indices.resize(new_sorted_indices_size);
    sorted_indices.shrink_to_fit();
    // tmp_raw.resize(new_sorted_indices_size);
    // gather the newt data
    for (int i = 0; i < arity; i++) {
        auto column_data = data[i].begin() + columns[i].raw_offset;
        thrust::gather(DEFAULT_DEVICE_POLICY, sorted_indices.begin(),
                       sorted_indices.end(), column_data, tmp_raw_ptr);
        thrust::copy(DEFAULT_DEVICE_POLICY, tmp_raw_ptr,
                     tmp_raw_ptr + new_sorted_indices_size, column_data);
        columns[i].raw_size = new_sorted_indices_size;
    }
    // set the 0-th column's sorted indices
    thrust::sequence(DEFAULT_DEVICE_POLICY, sorted_indices.begin(),
                     sorted_indices.end());
    columns[default_index_column].sorted_indices.swap(sorted_indices);
    newt_size = new_sorted_indices_size;
    total_tuples = newt_size + full_size;
}

void multi_hisa::fit() {
    total_tuples = newt_size + full_size;
    for (int i = 0; i < arity; i++) {
        data[i].resize(total_tuples);
        data[i].shrink_to_fit();
    }
}

void merge_column0_index(multi_hisa &h) {
    auto &full_column0 = h.full_columns[h.default_index_column];
    auto &newt_column0 = h.newt_columns[h.default_index_column];

    DEVICE_VECTOR<internal_data_type *> all_col_ptrs(h.arity);
    for (int i = 0; i < h.arity; i++) {
        all_col_ptrs[i] = h.data[i].data().get();
    }
    device_indices_t merged_idx(h.full_size + h.newt_size);
    // std::cout << "merge column0 index " << merged_idx.size() << " "
    //           << h.full_size << " "
    //           << full_column0.sorted_indices.size() << " "
    //           << h.newt_size << " "
    //           << h.data[0].size() << " "
    //           << h.data[1].size() << std::endl;
    thrust::merge(
        DEFAULT_DEVICE_POLICY, full_column0.sorted_indices.begin(),
        full_column0.sorted_indices.end(),
        thrust::make_counting_iterator<uint32_t>(h.full_size),
        thrust::make_counting_iterator<uint32_t>(h.full_size + h.newt_size),
        merged_idx.begin(),
        [arity = h.arity, default_index_column = h.default_index_column,
         raw_data = all_col_ptrs.data().get()] __device__(auto full_idx,
                                                          auto new_idx) {
            // compare the default index column then the other in lexical order
            if (raw_data[default_index_column][full_idx] !=
                raw_data[default_index_column][new_idx]) {
                return raw_data[default_index_column][full_idx] <
                       raw_data[default_index_column][new_idx];
            }
            for (int i = 0; i < arity; i++) {
                if (i == default_index_column) {
                    continue;
                }
                if (raw_data[i][full_idx] != raw_data[i][new_idx]) {
                    return raw_data[i][full_idx] < raw_data[i][new_idx];
                }
            }
            return false;
        });
    full_column0.sorted_indices.swap(merged_idx);
}

inline __device__ bool tuple_compare(uint32_t **full, uint32_t full_idx,
                                     uint32_t **newt, uint32_t newt_idx,
                                     int arity, int default_index_column) {
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

inline __device__ bool tuple_eq(uint32_t **full, uint32_t full_idx,
                                uint32_t **newt, uint32_t newt_idx, int arity,
                                int default_index_column) {
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

void multi_hisa::persist_newt() {
    // clear the index of delta
    for (int i = 0; i < arity; i++) {
        delta_columns[i].sorted_indices.resize(0);
        delta_columns[i].sorted_indices.shrink_to_fit();
        delta_columns[i].unique_v.resize(0);
        delta_columns[i].unique_v.shrink_to_fit();
        delta_columns[i].clear_unique_v();
        if (delta_columns[i].unique_v_map) {
            delta_columns[i].unique_v_map->clear();
        }
        delta_columns[i].raw_offset = full_size;
        delta_columns[i].raw_size = 0;
    }
    delta_size = 0;

    if (newt_size == 0) {
        return;
    }
    // merge newt to full
    if (full_size == 0) {
        full_size = newt_size;
        delta_size = newt_size;
        // thrust::swap(newt_columns, full_columns);
        for (int i = 0; i < arity; i++) {
            full_columns[i].raw_size = newt_columns[i].raw_size;
            full_columns[i].sorted_indices.swap(newt_columns[i].sorted_indices);
            delta_columns[i].raw_size = newt_columns[i].raw_size;
            delta_columns[i].raw_offset = 0;
            newt_columns[i].raw_size = 0;
            newt_columns[i].raw_offset = full_size;
        }
        build_index_all(RelationVersion::FULL);
        build_index_all(RelationVersion::DELTA);
        newt_size = 0;
        return;
    }

    auto before_dedup = std::chrono::high_resolution_clock::now();
    // difference newt and full
    device_ranges_t matched_ranges(newt_size);
    // do a column match on the default index column
    auto default_col_newt_raw_begin =
        data[default_index_column].begin() + full_size;
    full_columns[default_index_column].unique_v_map->find(
        default_col_newt_raw_begin, default_col_newt_raw_begin + newt_size,
        matched_ranges.begin());
    // std::cout << "matched_ranges size: " << matched_ranges.size() <<
    // std::endl; for each range check if all column value is the same
    DEVICE_VECTOR<internal_data_type *> all_col_news_ptrs(arity);
    DEVICE_VECTOR<internal_data_type *> all_col_fulls_ptrs(arity);
    for (int i = 0; i < arity; i++) {
        all_col_news_ptrs[i] = data[i].data().get() + full_size;
        all_col_fulls_ptrs[i] = data[i].data().get();
    }
    device_bitmap_t dup_newt_flags(newt_size, false);
    // print full raw
    // print_raw_data(RelationVersion::FULL);
    thrust::transform(
        thrust::make_counting_iterator<uint32_t>(0),
        thrust::make_counting_iterator<uint32_t>(newt_size),
        matched_ranges.begin(), dup_newt_flags.begin(),
        [all_col_news_ptrs = all_col_news_ptrs.data().get(),
         all_col_fulls_ptrs = all_col_fulls_ptrs.data().get(),
         all_col_full_idx_ptrs =
             full_columns[default_index_column].sorted_indices.data().get(),
         default_index_column = default_index_column,
         arity = arity] __device__(auto newt_index, uint64_t range) -> bool {
            if (range == UINT32_MAX) {
                return false;
            }
            // higher 32 bit is the start index, lower 32 bit is the offset
            uint32_t full_pos_start = static_cast<uint32_t>(range >> 32);
            uint32_t full_pos_offset =
                static_cast<uint32_t>(range & 0xFFFFFFFF);
            // binary search
            int left = 0;
            int right = full_pos_offset - 1;
            while (left <= right) {
                int mid = left + (right - left) / 2;
                auto full_index = all_col_full_idx_ptrs[full_pos_start + mid];
                // printf("mid: %d\n", mid);
                if (tuple_eq(all_col_fulls_ptrs, full_index, all_col_news_ptrs,
                             newt_index, arity, default_index_column)) {
                    return true;
                }
                if (tuple_compare(all_col_fulls_ptrs, full_index,
                                  all_col_news_ptrs, newt_index, arity,
                                  default_index_column)) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            return false;
        });
    auto dup_size = thrust::count(DEFAULT_DEVICE_POLICY, dup_newt_flags.begin(),
                                  dup_newt_flags.end(), true);
    // std::cout << "dup size: " << dup_size << std::endl;
    auto after_dedup = std::chrono::high_resolution_clock::now();
    dedup_time += std::chrono::duration_cast<std::chrono::microseconds>(
                      after_dedup - before_dedup)
                      .count();

    // clear newt only keep match_newt
    if (dup_size != 0) {
        for (size_t i = 0; i < arity; i++) {
            auto newt_begin = data[i].begin() + newt_columns[i].raw_offset;
            auto new_newt_end = thrust::remove_if(
                DEFAULT_DEVICE_POLICY, newt_begin, newt_begin + newt_size,
                dup_newt_flags.begin(), thrust::identity<bool>());
            newt_columns[i].raw_size = new_newt_end - newt_begin;
        }
        newt_size = newt_size - dup_size;
    }
    if (newt_size == 0) {
        // set current delta to 0
        delta_size = 0;
        // drop newt
        for (int i = 0; i < arity; i++) {
            newt_columns[i].raw_offset = full_size;
            newt_columns[i].raw_size = 0;
        }
        return;
    }
    // std::cout << "newt size after match: " << newt_size << std::endl;
    // set total size
    total_tuples = full_size + newt_size;

    auto merge_start = std::chrono::high_resolution_clock::now();
    // sort and merge newt
    device_data_t tmp_newt_v(newt_size);
    // device_data_t tmp_full_v(full_size+newt_size);
    // merge column 0, this is different then the other columns
    merge_column0_index(*this);
    // std::cout << "merge column 0 done" << std::endl;
    for (size_t i = 0; i < arity; i++) {
        if (i == default_index_column) {
            continue;
        }
        if (full_columns[i].index_strategy == IndexStrategy::LAZY) {
            continue;
        }
        auto &newt_column = newt_columns[i];
        auto &full_column = full_columns[i];

        newt_column.sorted_indices.resize(newt_size);
        auto newt_head = data[i].begin() + newt_column.raw_offset;
        thrust::copy(DEFAULT_DEVICE_POLICY, newt_head, newt_head + newt_size,
                     tmp_newt_v.begin());
        thrust::sequence(DEFAULT_DEVICE_POLICY,
                         newt_column.sorted_indices.begin(),
                         newt_column.sorted_indices.end(), full_size);
        thrust::stable_sort_by_key(
            DEFAULT_DEVICE_POLICY,
            thrust::make_permutation_iterator(
                data[i].begin(), newt_column.sorted_indices.begin()),
            thrust::make_permutation_iterator(data[i].begin(),
                                              newt_column.sorted_indices.end()),
            newt_column.sorted_indices.begin());

        // merge
        // device_data_t merged_column(full_size + newt_size);
        // buffer->reserve(full_size + newt_size);
        data_buffer.resize(full_size + newt_size);
        // auto merged_column = buffer->data();
        auto &merged_column = data_buffer;
        // std::cout << "data size: " << data[i].size() << std::endl;

        thrust::merge_by_key(
            DEFAULT_DEVICE_POLICY,
            thrust::make_permutation_iterator(
                data[i].begin(), newt_column.sorted_indices.begin()),
            thrust::make_permutation_iterator(data[i].begin(),
                                              newt_column.sorted_indices.end()),
            thrust::make_permutation_iterator(
                data[i].begin(), full_column.sorted_indices.begin()),
            thrust::make_permutation_iterator(data[i].begin(),
                                              full_column.sorted_indices.end()),
            newt_column.sorted_indices.begin(),
            full_column.sorted_indices.begin(), thrust::make_discard_iterator(),
            merged_column.begin());
        full_column.sorted_indices.swap(merged_column);
        // buffer->swap(full_column.sorted_indices);
        // minus size of full on all newt indices
        thrust::transform(DEFAULT_DEVICE_POLICY,
                          newt_column.sorted_indices.begin(),
                          newt_column.sorted_indices.end(),
                          thrust::make_constant_iterator(full_size),
                          newt_column.sorted_indices.begin(),
                          thrust::minus<internal_data_type>());
        full_column.raw_size = full_size + newt_size;
    }
    full_size += newt_size;
    tmp_newt_v.resize(0);
    tmp_newt_v.shrink_to_fit();
    auto merge_end = std::chrono::high_resolution_clock::now();
    merge_time += std::chrono::duration_cast<std::chrono::microseconds>(
                      merge_end - merge_start)
                      .count();

    // swap newt and delta
    delta_size = newt_size;
    newt_size = 0;
    // thrust::swap(newt_columns, delta_columns);
    for (int i = 0; i < arity; i++) {
        delta_columns[i].raw_offset = newt_columns[i].raw_offset;
        delta_columns[i].raw_size = newt_columns[i].raw_size;
        delta_columns[i].sorted_indices.swap(newt_columns[i].sorted_indices);
        full_columns[i].raw_size = full_size;
        // set newt to the end of full
        newt_columns[i].raw_offset = full_size;
        newt_columns[i].raw_size = 0;
    }
    // std::cout << ">>>>>>>>>>" << std::endl;

    // build indices on full
    build_index_all(RelationVersion::FULL);

    build_index_all(RelationVersion::DELTA);
}

void multi_hisa::print_stats() {
    std::cout << "sort time: " << sort_time / 1000000.0 << std::endl;
    std::cout << "hash time: " << hash_time / 1000000.0 << std::endl;
    std::cout << "dedup time: " << dedup_time / 1000000.0 << std::endl;
    std::cout << "merge time: " << merge_time / 1000000.0 << std::endl;
}

void multi_hisa::clear() {
    for (int i = 0; i < arity; i++) {
        full_columns[i].raw_offset = 0;
        full_columns[i].sorted_indices.resize(0);
        full_columns[i].sorted_indices.shrink_to_fit();
        full_columns[i].clear_unique_v();

        delta_columns[i].raw_offset = 0;
        delta_columns[i].sorted_indices.resize(0);
        delta_columns[i].sorted_indices.shrink_to_fit();
        delta_columns[i].clear_unique_v();

        newt_columns[i].raw_offset = 0;
        newt_columns[i].sorted_indices.resize(0);
        newt_columns[i].sorted_indices.shrink_to_fit();
        newt_columns[i].clear_unique_v();

        data[i].resize(0);
        data[i].shrink_to_fit();
    }
    newt_size = 0;
    full_size = 0;
    delta_size = 0;
    total_tuples = 0;
}

} // namespace vflog

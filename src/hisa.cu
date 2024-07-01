
#include "hisa.cuh"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

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

namespace hisa {

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

multi_hisa::multi_hisa(int arity) {
    this->arity = arity;
    newt_size = 0;
    full_size = 0;
    delta_size = 0;
    full_columns.resize(arity);
    delta_columns.resize(arity);
    newt_columns.resize(arity);
    data.resize(arity);

    for (int i = 0; i < arity; i++) {
        full_columns[i].column_idx = i;
        delta_columns[i].column_idx = i;
        newt_columns[i].column_idx = i;
    }
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
                             device_data_t &unique_offset,
                             device_data_t &unique_diff, bool sorted) {
    device_data_t column_data(column.raw_size);
    auto sorte_start = std::chrono::high_resolution_clock::now();
    auto &raw_data = data[column.column_idx];
    auto raw_ver_head = raw_data.begin() + column.raw_offset;
    if (!sorted) {
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
    } else {
        thrust::gather(DEFAULT_DEVICE_POLICY, column.sorted_indices.begin(),
                       column.sorted_indices.end(), raw_ver_head,
                       column_data.begin());
    }
    auto sort_end = std::chrono::high_resolution_clock::now();
    this->sort_time += std::chrono::duration_cast<std::chrono::microseconds>(
                           sort_end - sorte_start)
                           .count();
    // compress the column, save unique values and their counts
    thrust::sequence(DEFAULT_DEVICE_POLICY, unique_offset.begin(),
                     unique_offset.end());
    // using thrust parallel algorithm to compress the column
    // mark non-unique values as 0
    // print column_data
    // std::cout << ">>>>>> raw size: " << column.sorted_indices.size() <<
    // std::endl;
    // // print column.sorted_indices
    // HOST_VECTOR<internal_data_type> h_column_data = column.sorted_indices;
    // for (int i = 0; i < h_column_data.size(); i++) {
    //     std::cout << h_column_data[i] << " ";
    // }
    // std::cout << std::endl;
    auto uniq_end =
        thrust::unique_by_key(DEFAULT_DEVICE_POLICY, column_data.begin(),
                              column_data.end(), unique_offset.begin());
    auto uniq_size = uniq_end.first - column_data.begin();
    auto &uniq_val = column_data;
    uniq_val.resize(uniq_size);

    unique_offset.resize(uniq_size);
    // device_data_t unique_diff = unique_offset;
    unique_diff.resize(uniq_size + 1);
    thrust::copy(DEFAULT_DEVICE_POLICY, unique_offset.begin(),
                 unique_offset.end(), unique_diff.begin());
    unique_diff[uniq_size] = column.raw_size;
    // calculate offset by minus the previous value
    thrust::adjacent_difference(DEFAULT_DEVICE_POLICY, unique_diff.begin(),
                                unique_diff.end(), unique_diff.begin());
    // the first value is always 0, in following code, we will use the
    // 1th as the start index
    unique_diff.erase(unique_diff.begin());
    unique_diff.resize(uniq_size);

    // update map
    auto start = std::chrono::high_resolution_clock::now();
    if (column.use_real_map) {
        if (column.unique_v_map) {
            // columns[i].unique_v_map->reserve(uniq_size);
            if (column.unique_v_map->size() <= uniq_size + 1) {
                column.unique_v_map->clear();
            } else {
                column.unique_v_map = nullptr;
                column.unique_v_map = CREATE_V_MAP(uniq_size);
            }
        } else {
            column.unique_v_map = CREATE_V_MAP(uniq_size);
        }
        auto insertpair_begin = thrust::make_transform_iterator(
            thrust::make_zip_iterator(thrust::make_tuple(
                uniq_val.begin(), unique_offset.begin(), unique_diff.begin())),
            cuda::proclaim_return_type<GpuMapPair>([] __device__(auto &t) {
                return HASH_NAMESPACE::make_pair(
                    thrust::get<0>(t),
                    (static_cast<uint64_t>(thrust::get<1>(t)) << 32) +
                        (static_cast<uint64_t>(thrust::get<2>(t))));
            }));
        column.unique_v_map->insert(insertpair_begin,
                                    insertpair_begin + uniq_size);
    } else {
        device_ranges_t ranges(uniq_size);
        // compress offset and diff to u64 ranges
        thrust::transform(DEFAULT_DEVICE_POLICY, unique_offset.begin(),
                          unique_offset.end(), unique_diff.begin(),
                          ranges.begin(),
                          [] __device__(auto &offset, auto &diff) -> uint64_t {
                              return (static_cast<uint64_t>(offset) << 32) +
                                     (static_cast<uint64_t>(diff));
                          });
        column.unique_v_map_simp.insert(uniq_val, ranges);
        // std::cout << "ranges size: " << ranges.size() << std::endl;
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
    device_data_t unique_offset(total_tuples);
    device_data_t unique_diff(total_tuples);
    build_index(columns[i], unique_offset, unique_diff, sorted);
}

void multi_hisa::build_index_all(RelationVersion version, bool sorted) {
    auto &columns = get_versioned_columns(version);
    auto version_size = get_versioned_size(version);

    device_data_t unique_offset(version_size);
    device_data_t unique_diff(version_size);
    for (size_t i = 0; i < arity; i++) {
        if (columns[i].index_strategy == IndexStrategy::EAGER) {
            std::cout << "build index " << i << std::endl;
            if (i != default_index_column) {
                build_index(columns[i], unique_offset, unique_diff, sorted);
            } else {
                build_index(columns[i], unique_offset, unique_diff, true);
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
    std::cout << "deduplicate newt size: " << version_size << std::endl;
    thrust::sequence(DEFAULT_DEVICE_POLICY, sorted_indices.begin(),
                     sorted_indices.end());
    device_data_t tmp_raw(version_size);
    for (int i = arity - 1; i >= 0; i--) {
        if (i == default_index_column) {
            continue;
        }
        auto &column = columns[i];
        auto column_data = data[i].begin() + column.raw_offset;
        // gather the column data
        thrust::gather(DEFAULT_DEVICE_POLICY, sorted_indices.begin(),
                       sorted_indices.end(), column_data, tmp_raw.begin());
        thrust::stable_sort_by_key(DEFAULT_DEVICE_POLICY, tmp_raw.begin(),
                                   tmp_raw.end(), sorted_indices.begin());
    }

    // sort the default index column
    auto &column = columns[default_index_column];
    auto column_data = data[default_index_column].begin() + column.raw_offset;
    thrust::gather(DEFAULT_DEVICE_POLICY, sorted_indices.begin(),
                   sorted_indices.end(), column_data, tmp_raw.begin());
    thrust::stable_sort_by_key(DEFAULT_DEVICE_POLICY, tmp_raw.begin(),
                               tmp_raw.end(), sorted_indices.begin());

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
    tmp_raw.resize(new_sorted_indices_size);
    // gather the newt data
    for (int i = 0; i < arity; i++) {
        auto column_data = data[i].begin() + columns[i].raw_offset;
        thrust::gather(DEFAULT_DEVICE_POLICY, sorted_indices.begin(),
                       sorted_indices.end(), column_data, tmp_raw.begin());
        thrust::copy(DEFAULT_DEVICE_POLICY, tmp_raw.begin(), tmp_raw.end(),
                     column_data);
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

void column_match(multi_hisa &inner, RelationVersion inner_ver,
                  size_t inner_column_idx, multi_hisa &outer,
                  RelationVersion outer_ver, size_t outer_column_idx,
                  device_pairs_t &matched_pair) {
    auto &column_inner =
        inner.get_versioned_columns(inner_ver)[inner_column_idx];
    auto &column_outer =
        outer.get_versioned_columns(outer_ver)[outer_column_idx];
    auto matched_size = matched_pair.size();
    // check if they agree on the value
    thrust::device_vector<bool> unmatched_flags(matched_size, true);
    device_data_t tmp_inner(matched_size);
    auto raw_inner = inner.data[column_inner.column_idx].data().get() +
                     column_inner.raw_offset;
    thrust::transform(
        DEFAULT_DEVICE_POLICY, matched_pair.begin(), matched_pair.end(),
        tmp_inner.begin(),
        [raw_inner] __device__(auto &t) { return raw_inner[t & 0xFFFFFFFF]; });
    // print tmp_inner
    // HOST_VECTOR<internal_data_type> h_tmp_inner = tmp_inner;
    // for (int i = 0; i < h_tmp_inner.size(); i++) {
    //     std::cout << h_tmp_inner[i] << " ";
    // }
    // std::cout << std::endl;

    auto raw_outer = outer.data[column_outer.column_idx].data().get() +
                     column_outer.raw_offset;
    thrust::transform(
        matched_pair.begin(), matched_pair.end(), unmatched_flags.begin(),
        [raw_outer, raw_inner] __device__(auto &t) {
            return raw_outer[t >> 32] != raw_inner[t & 0xFFFFFFFF];
        });
    // filter
    auto new_matched_pair_end =
        thrust::remove_if(matched_pair.begin(), matched_pair.end(),
                          unmatched_flags.begin(), thrust::identity<bool>());
    auto new_matched_pair_size = new_matched_pair_end - matched_pair.begin();
    matched_pair.resize(new_matched_pair_size);
}

void column_match(multi_hisa &h1, RelationVersion ver1, size_t column1_idx,
                  multi_hisa &h2, RelationVersion ver2, size_t column2_idx,
                  device_indices_t &column1_indices,
                  device_indices_t &column2_indices,
                  DEVICE_VECTOR<bool> &unmatched) {
    auto &column1 = h1.get_versioned_columns(ver1)[column1_idx];
    auto &column2 = h2.get_versioned_columns(ver2)[column2_idx];
    auto matched_size = column1_indices.size();
    // check if they agree on the value
    unmatched.resize(matched_size);

    auto column1_raw_begin =
        h1.data[column1.column_idx].begin() + column1.raw_offset;
    auto column2_raw_begin =
        h2.data[column2.column_idx].begin() + column2.raw_offset;
    thrust::transform(
        DEFAULT_DEVICE_POLICY,
        thrust::make_permutation_iterator(column1_raw_begin,
                                          column1_indices.begin()),
        thrust::make_permutation_iterator(column1_raw_begin,
                                          column1_indices.end()),
        thrust::make_permutation_iterator(column2_raw_begin,
                                          column2_indices.begin()),
        unmatched.begin(), thrust::not_equal_to<internal_data_type>());
    // filter
    auto new_column1_indices_end = thrust::remove_if(
        DEFAULT_DEVICE_POLICY, column1_indices.begin(), column1_indices.end(),
        unmatched.begin(), thrust::identity<bool>());
    auto new_column1_indices_size =
        new_column1_indices_end - column1_indices.begin();
    column1_indices.resize(new_column1_indices_size);
    auto new_column2_indices_end = thrust::remove_if(
        DEFAULT_DEVICE_POLICY, column2_indices.begin(), column2_indices.end(),
        unmatched.begin(), thrust::identity<bool>());
    auto new_column2_indices_size =
        new_column2_indices_end - column2_indices.begin();
    column2_indices.resize(new_column2_indices_size);
}

void merge_column0_index(multi_hisa &h) {
    auto &full_column0 = h.full_columns[h.default_index_column];
    auto &newt_column0 = h.newt_columns[h.default_index_column];

    DEVICE_VECTOR<internal_data_type *> all_col_ptrs(h.arity);
    for (int i = 0; i < h.arity; i++) {
        all_col_ptrs[i] = h.data[i].data().get();
    }

    device_indices_t merged_idx(h.full_size + h.newt_size);
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
        delta_columns[i].unique_v_map = nullptr;
        delta_columns[i].raw_offset = full_size;
        delta_columns[i].raw_size = 0;
    }
    // merge newt to full
    if (full_size == 0) {
        full_size = newt_size;
        // thrust::swap(newt_columns, full_columns);
        for (int i = 0; i < arity; i++) {
            full_columns[i].raw_size = newt_columns[i].raw_size;
            full_columns[i].sorted_indices.swap(newt_columns[i].sorted_indices);
            newt_columns[i].raw_size = 0;
            newt_columns[i].raw_offset = full_size;
        }
        build_index_all(RelationVersion::FULL);
        // TODO: need some way to also set the delta index
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

            // // I know full is sorted by lexical order, so I can do a binary
            // // search
            // for (int i = 0; i < full_pos_offset; i++) {
            //     auto full_index = all_col_full_idx_ptrs[full_pos_start + i];
            //     bool is_duplicated = tuple_eq(all_col_fulls_ptrs, full_index,
            //                                   all_col_news_ptrs, newt_index,
            //                                   arity, default_index_column);
            //     if (is_duplicated) {
            //         printf("duplicated >>>>>>> %d\n", i);
            //         return true;
            //     }
            // }
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
    std::cout << "dup size: " << dup_size << std::endl;
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
        auto &merged_column = data_buffer;
        data_buffer.resize(full_size + newt_size);
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
    sort_time += std::chrono::duration_cast<std::chrono::microseconds>(
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
    std::cout << "sort time: " << sort_time << std::endl;
    std::cout << "hash time: " << hash_time << std::endl;
    std::cout << "dedup time: " << dedup_time << std::endl;
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
    total_tuples = 0;
}

void column_join(multi_hisa &inner, RelationVersion inner_ver, size_t inner_idx,
                 multi_hisa &outer, RelationVersion outer_ver, size_t outer_idx,
                 device_data_t &outer_tuple_indices,
                 device_pairs_t &matched_indices) {
    auto &inner_column = inner.get_versioned_columns(inner_ver)[inner_idx];
    auto &outer_column = outer.get_versioned_columns(outer_ver)[outer_idx];
    auto outer_size = outer_tuple_indices.size();

    device_ranges_t range_result(outer_tuple_indices.size());

    auto outer_raw_begin =
        outer.data[outer_column.column_idx].begin() + outer_column.raw_offset;
    inner_column.unique_v_map->find(
        thrust::make_permutation_iterator(outer_raw_begin,
                                          outer_tuple_indices.begin()),
        thrust::make_permutation_iterator(outer_raw_begin,
                                          outer_tuple_indices.end()),
        range_result.begin());

    // mark the outer_tuple_indices as UINT32_MAX if corresponding range result
    // is UINT32_MAX
    thrust::transform(
        DEFAULT_DEVICE_POLICY, range_result.begin(), range_result.end(),
        outer_tuple_indices.begin(), outer_tuple_indices.begin(),
        [] __device__(auto &range, auto &outer_tuple_index) {
            return range == UINT32_MAX ? UINT32_MAX : outer_tuple_index;
        });
    // remove unmatched U32_MAX
    auto new_outer_tuple_end =
        thrust::remove(DEFAULT_DEVICE_POLICY, outer_tuple_indices.begin(),
                       outer_tuple_indices.end(), UINT32_MAX);
    outer_tuple_indices.resize(new_outer_tuple_end -
                               outer_tuple_indices.begin());
    // remove unmatched range result
    auto new_range_end =
        thrust::remove(DEFAULT_DEVICE_POLICY, range_result.begin(),
                       range_result.end(), UINT32_MAX);
    range_result.resize(new_range_end - range_result.begin());

    // print range_result
    // std::cout << "range_result:\n";
    // HOST_VECTOR<uint64_t> h_range_result = range_result;
    // for (int i = 0; i < h_range_result.size(); i++) {
    //     std::cout << "(" << (h_range_result[i] >> 32) << " "
    //               << (h_range_result[i] & 0xffffffff) << ") ";
    // }
    // std::cout << std::endl;

    // materialize the comp_ranges

    // fecth all range size
    device_data_t size_vec(range_result.size());
    thrust::transform(
        DEFAULT_DEVICE_POLICY, range_result.begin(), range_result.end(),
        size_vec.begin(),
        [] __device__(auto &t) { return static_cast<uint32_t>(t); });

    device_ranges_t offset_vec;
    offset_vec.swap(range_result);
    thrust::transform(
        DEFAULT_DEVICE_POLICY, offset_vec.begin(), offset_vec.end(),
        offset_vec.begin(),
        [] __device__(auto &t) { return static_cast<uint64_t>(t >> 32); });

    uint32_t total_matched_size =
        thrust::reduce(DEFAULT_DEVICE_POLICY, size_vec.begin(), size_vec.end());
    device_data_t size_offset_tmp(outer_size);
    thrust::exclusive_scan(DEFAULT_DEVICE_POLICY, size_vec.begin(),
                           size_vec.end(), size_offset_tmp.begin());
    // std::cout << "total_matched_size: " << total_matched_size << std::endl;
    // // print size_vec
    // std::cout << "size_offset_tmp:\n";
    // HOST_VECTOR<uint32_t> h_size_vec = offset_vec;
    // for (int i = 0; i < h_size_vec.size(); i++) {
    //     std::cout << h_size_vec[i] << " ";
    // }
    // std::cout << std::endl;

    // pirnt outer tuple indices
    // std::cout << "outer_tuple_indices:\n";
    // HOST_VECTOR<uint32_t> h_outer_tuple_indices =
    // outer_tuple_indices; for (int i = 0; i < h_outer_tuple_indices.size();
    // i++) {
    //     std::cout << h_outer_tuple_indices[i] << " ";
    // }
    // std::cout << std::endl;

    // materialize the matched_indices
    matched_indices.resize(total_matched_size);
    thrust::for_each(
        DEFAULT_DEVICE_POLICY,
        thrust::make_zip_iterator(
            thrust::make_tuple(outer_tuple_indices.begin(), offset_vec.begin(),
                               size_vec.begin(), size_offset_tmp.begin())),
        thrust::make_zip_iterator(
            thrust::make_tuple(outer_tuple_indices.end(), offset_vec.end(),
                               size_vec.end(), size_offset_tmp.end())),
        [res = matched_indices.data().get(),
         inner_sorted_idx =
             inner_column.sorted_indices.data().get()] __device__(auto &t) {
            auto outer_pos = thrust::get<0>(t);
            auto &inner_pos = thrust::get<1>(t);
            auto &size = thrust::get<2>(t);
            auto &start = thrust::get<3>(t);
            for (int i = 0; i < size; i++) {
                res[start + i] =
                    compress_u32(outer_pos, inner_sorted_idx[inner_pos + i]);
            }
        });
}

void column_join(multi_hisa &inner, RelationVersion inner_ver, size_t inner_idx,
                 multi_hisa &outer, RelationVersion outer_ver, size_t outer_idx,
                 device_indices_t &outer_tuple_indices,
                 device_indices_t &matched_indices,
                 DEVICE_VECTOR<bool> &unmatched) {
    auto &inner_column = inner.get_versioned_columns(inner_ver)[inner_idx];
    auto &outer_column = outer.get_versioned_columns(outer_ver)[outer_idx];
    auto outer_size = outer_tuple_indices.size();
    unmatched.resize(outer_size);
    thrust::fill(DEFAULT_DEVICE_POLICY, unmatched.begin(), unmatched.end(),
                 false);

    device_ranges_t range_result(outer_tuple_indices.size());

    auto outer_raw_begin =
        outer.data[outer_column.column_idx].begin() + outer_column.raw_offset;
    inner_column.unique_v_map->find(
        thrust::make_permutation_iterator(outer_raw_begin,
                                          outer_tuple_indices.begin()),
        thrust::make_permutation_iterator(outer_raw_begin,
                                          outer_tuple_indices.end()),
        range_result.begin());

    // mark the outer_tuple_indices as UINT32_MAX if corresponding range result
    // is UINT32_MAX
    thrust::transform(
        DEFAULT_DEVICE_POLICY, range_result.begin(), range_result.end(),
        outer_tuple_indices.begin(), outer_tuple_indices.begin(),
        [] __device__(auto &range, auto &outer_tuple_index) {
            return range == UINT32_MAX ? UINT32_MAX : outer_tuple_index;
        });
    // mark the unmatched as true
    thrust::transform(DEFAULT_DEVICE_POLICY, range_result.begin(),
                      range_result.end(), unmatched.begin(),
                      [] __device__(auto &t) { return t == UINT32_MAX; });
    // remove unmatched U32_MAX
    auto new_outer_tuple_end =
        thrust::remove(DEFAULT_DEVICE_POLICY, outer_tuple_indices.begin(),
                       outer_tuple_indices.end(), UINT32_MAX);
    outer_tuple_indices.resize(new_outer_tuple_end -
                               outer_tuple_indices.begin());
    // remove unmatched range result
    auto new_range_end =
        thrust::remove(DEFAULT_DEVICE_POLICY, range_result.begin(),
                       range_result.end(), UINT32_MAX);
    range_result.resize(new_range_end - range_result.begin());

    // materialize the comp_ranges

    // fecth all range size
    device_data_t size_vec(range_result.size());
    thrust::transform(
        DEFAULT_DEVICE_POLICY, range_result.begin(), range_result.end(),
        size_vec.begin(),
        [] __device__(auto &t) { return static_cast<uint32_t>(t); });

    device_ranges_t offset_vec;
    offset_vec.swap(range_result);
    thrust::transform(
        DEFAULT_DEVICE_POLICY, offset_vec.begin(), offset_vec.end(),
        offset_vec.begin(),
        [] __device__(auto &t) { return static_cast<uint64_t>(t >> 32); });

    uint32_t total_matched_size =
        thrust::reduce(DEFAULT_DEVICE_POLICY, size_vec.begin(), size_vec.end());
    device_data_t size_offset_tmp(outer_size);
    thrust::exclusive_scan(DEFAULT_DEVICE_POLICY, size_vec.begin(),
                           size_vec.end(), size_offset_tmp.begin());

    // materialize the matched_indices
    device_data_t materialized_outer(total_matched_size);
    matched_indices.resize(total_matched_size);
    thrust::for_each(
        DEFAULT_DEVICE_POLICY,
        thrust::make_zip_iterator(
            thrust::make_tuple(outer_tuple_indices.begin(), offset_vec.begin(),
                               size_vec.begin(), size_offset_tmp.begin())),
        thrust::make_zip_iterator(
            thrust::make_tuple(outer_tuple_indices.end(), offset_vec.end(),
                               size_vec.end(), size_offset_tmp.end())),
        [res_inner = matched_indices.data().get(),
         res_outer = materialized_outer.data().get(),
         inner_sorted_idx =
             inner_column.sorted_indices.data().get()] __device__(auto &t) {
            auto outer_pos = thrust::get<0>(t);
            auto &inner_pos = thrust::get<1>(t);
            auto &size = thrust::get<2>(t);
            auto &start = thrust::get<3>(t);
            for (int i = 0; i < size; i++) {
                // res[start + i] =
                //     compress_u32(outer_pos, inner_sorted_idx[inner_pos + i]);
                res_inner[start + i] = inner_sorted_idx[inner_pos + i];
                res_outer[start + i] = outer_pos;
            }
        });
    outer_tuple_indices.swap(materialized_outer);
}

void column_copy(multi_hisa &src, RelationVersion src_ver, size_t src_idx,
                 multi_hisa &dst, RelationVersion dst_ver, size_t dst_idx,
                 device_indices_t &indices) {
    // print indices
    // gather
    auto src_raw_begin = src.data[src_idx].begin() +
                         src.get_versioned_columns(src_ver)[src_idx].raw_offset;
    auto dst_raw_begin = dst.data[dst_idx].begin() +
                         dst.get_versioned_columns(dst_ver)[dst_idx].raw_offset;
    thrust::gather(DEFAULT_DEVICE_POLICY, indices.begin(), indices.end(),
                   src_raw_begin, dst_raw_begin);
    // TODO: check this
    dst.get_versioned_columns(dst_ver)[dst_idx].raw_size = indices.size();
}

std::string trim(const std::string &str) {
    size_t first = str.find_first_not_of(" \t\n\v\f\r");
    if (std::string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(" \t\n\v\f\r");
    return str.substr(first, (last - first + 1));
}

// function read in a k-arity relation file, load into a hisa
void read_kary_relation(const std::string &filename, hisa::multi_hisa &h,
                        int k) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: file not found" << std::endl;
        return;
    }

    HOST_VECTOR<HOST_VECTOR<hisa::internal_data_type>> tuples_vertical(k);

    std::string line;
    uint32_t line_count = 0;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        HOST_VECTOR<hisa::internal_data_type> tokens;
        std::string token;
        while (std::getline(iss, token, '\t')) {
            // trim whitespace on the token and convert into a uint32_t
            auto trimmed = trim(token);
            tokens.push_back(std::stoi(trimmed));
        }

        if (tokens.size() != k) {
            std::cerr << "Error: invalid arity" << std::endl;
            return;
        }

        for (size_t i = 0; i < k; i++) {
            tuples_vertical[i].push_back(tokens[i]);
        }
        line_count++;
    }

    h.init_load_vectical(tuples_vertical, line_count);
    // h.deduplicate();
}

} // namespace hisa


#include "../include/hisa.cuh"

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
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
        newt_columns[i].raw_data = data[i].data();
        newt_columns[i].raw_size = total_tuples;
    }
    auto load_end = std::chrono::high_resolution_clock::now();
    this->load_time += std::chrono::duration_cast<std::chrono::microseconds>(
                           load_end - load_start)
                           .count();
}

void multi_hisa::allocate_newt(size_t size) {
    auto old_size = total_tuples;
    for (int i = 0; i < arity; i++) {
        data[i].resize(old_size + size);
    }
    newt_size = size;
    for (int i = 0; i < arity; i++) {
        newt_columns[i].raw_data = data[i].data() + old_size;
    }
}

void multi_hisa::load_column_cpu(VetricalColumnCpu &columns_cpu,
                                 int column_idx) {
    auto total_tuples = columns_cpu.raw_data.size();
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
    full_columns[column_idx].raw_data = data[column_idx].data();
    delta_columns[column_idx].raw_data =
        data[column_idx].data() + columns_cpu.delta_head_offset;
    newt_columns[column_idx].raw_data =
        data[column_idx].data() + columns_cpu.newt_head_offset;
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
        thrust::copy(full_columns[i].raw_data,
                     full_columns[i].raw_data + full_size, column.begin());
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

void multi_hisa::build_index(VerticalColumnGpu &column,
                             device_data_t &unique_offset,
                             device_data_t &unique_diff, bool sorted = false) {
    device_data_t column_data(column.raw_size);
    // std::cout << "build index " << i << std::endl;
    auto sorte_start = std::chrono::high_resolution_clock::now();
    if (!sorted) {
        column.sorted_indices.resize(total_tuples);
        thrust::sequence(DEFAULT_DEVICE_POLICY, column.sorted_indices.begin(),
                         column.sorted_indices.end());
        // sort all values in the column
        thrust::copy(DEFAULT_DEVICE_POLICY, column.raw_data,
                     column.raw_data + column.raw_size, column_data.begin());
        // if (i != 0) {
        thrust::sort_by_key(DEFAULT_DEVICE_POLICY, column_data.begin(),
                            column_data.end(), column.sorted_indices.begin());
        // }
    } else {
        thrust::gather(DEFAULT_DEVICE_POLICY, column.sorted_indices.begin(),
                       column.sorted_indices.end(), column.raw_data,
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
    // for (int j = 0; j < column_data.size(); j++) {
    //     std::cout << column_data[j] << " ";
    // }s
    // std::cout << std::endl;

    auto uniq_end =
        thrust::unique_by_key(DEFAULT_DEVICE_POLICY, column_data.begin(),
                              column_data.end(), unique_offset.begin());
    auto uniq_size = uniq_end.first - column_data.begin();
    // column_data.resize(uniq_size);
    // unique_offset.resize(uniq_size);
    // columns[i].unique_v.swap(column_data);
    auto &uniq_val = column_data;
    uniq_val.resize(uniq_size);
    // columns[i].unique_v.resize(uniq_size);
    // thrust::copy(DEFAULT_DEVICE_POLICY, column_data.begin(),
    //              column_data.begin() + uniq_size,
    //              columns[i].unique_v.begin());
    // column_data.resize(0);
    // column_data.shrink_to_fit();

    unique_offset.resize(uniq_size);
    // device_data_t unique_diff = unique_offset;
    thrust::copy(DEFAULT_DEVICE_POLICY, unique_offset.begin(),
                 unique_offset.end(), unique_diff.begin());
    unique_diff.resize(uniq_size);
    unique_diff.push_back(total_tuples);
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
            column.unique_v_map = nullptr;
            column.unique_v_map = CREATE_V_MAP(uniq_size);
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

    device_data_t unique_offset(total_tuples);
    device_data_t unique_diff(total_tuples);
    for (size_t i = 0; i < arity; i++) {
        if (columns[i].index_strategy == IndexStrategy::EAGER) {
            build_index(columns[i], unique_offset, unique_diff, sorted);
        }
    }
    indexed = true;
}

void multi_hisa::deduplicate() {
    auto start = std::chrono::high_resolution_clock::now();
    VersionedColumns &columns = newt_columns;
    auto version_size = newt_size;
    // radix sort the raw data of each column
    device_data_t dup_indices(version_size);
    thrust::sequence(DEFAULT_DEVICE_POLICY, dup_indices.begin(),
                     dup_indices.end());
    device_data_t tmp_raw(version_size);
    thrust::device_vector<bool> dup_flags(version_size, false);
    for (int i = arity - 1; i >= 0; i--) {
        auto &column = columns[i];
        auto &column_data = column.raw_data;
        // gather the column data
        thrust::gather(DEFAULT_DEVICE_POLICY, dup_indices.begin(),
                       dup_indices.end(), column_data, tmp_raw.begin());
        thrust::stable_sort_by_key(DEFAULT_DEVICE_POLICY, tmp_raw.begin(),
                                   tmp_raw.end(), dup_indices.begin());
        // check if a value is duplicated to the left & right
        if (i != 0) {
            thrust::transform(
                DEFAULT_DEVICE_POLICY,
                thrust::make_counting_iterator<uint32_t>(0),
                thrust::make_counting_iterator<uint32_t>(dup_flags.size()),
                dup_flags.begin(),
                [tmp_raw = tmp_raw.data().get(),
                 total = dup_flags.size()] __device__(uint32_t i) -> bool {
                    if (i == 0) {
                        return tmp_raw[i] == tmp_raw[i + 1];
                    } else if (i == total - 1) {
                        return tmp_raw[i] == tmp_raw[i - 1];
                    } else {
                        return (tmp_raw[i] == tmp_raw[i - 1]) ||
                               (tmp_raw[i] == tmp_raw[i + 1]);
                    }
                });
        } else {
            // last column, we only need check the left, because the right is
            // what we want to keep
            thrust::transform(
                DEFAULT_DEVICE_POLICY,
                thrust::make_counting_iterator<uint32_t>(0),
                thrust::make_counting_iterator<uint32_t>(dup_flags.size()),
                dup_flags.begin(),
                [tmp_raw = tmp_raw.data().get(),
                 total = dup_flags.size()] __device__(uint32_t i) -> bool {
                    if (i == 0) {
                        return false;
                    } else {
                        return tmp_raw[i] == tmp_raw[i - 1];
                    }
                });
        }

        // print tmp_raw here
        // std::cout << "tmp_raw:\n";
        // HOST_VECTOR<internal_data_type> h_tmp_raw = tmp_raw;
        // for (int i = 0; i < h_tmp_raw.size(); i++) {
        //     std::cout << h_tmp_raw[i] << " ";
        // }
        // std::cout << std::endl;
        // std::cout << "dup_indices:\n";
        // HOST_VECTOR<internal_data_type> h_dup_indices = dup_indices;
        // for (int j = 0; j < h_dup_indices.size(); j++) {
        //     std::cout << h_dup_indices[j] << " ";
        // }
        // std::cout << std::endl;
        // std::cout << "dup_flags:\n";
        // HOST_VECTOR<bool> h_dup_flags = dup_flags;
        // for (int j = 0; j < h_dup_flags.size(); j++) {
        //     std::cout << h_dup_flags[j] << " ";
        // }
        // std::cout << std::endl;
        //  filter keep only duplicates, the one whose dup_flags is true
        auto new_dup_indices_end = thrust::remove_if(
            DEFAULT_DEVICE_POLICY, dup_indices.begin(), dup_indices.end(),
            dup_flags.begin(), thrust::logical_not<bool>());
        auto new_dup_size = new_dup_indices_end - dup_indices.begin();
        dup_indices.resize(new_dup_size);
        // resize and reset flags
        dup_flags.resize(new_dup_size);
        tmp_raw.resize(new_dup_size);
        if (new_dup_size == 0) {
            break;
        }
    }
    // print dup_indices
    // HOST_VECTOR<internal_data_type> h_dup_indices = dup_indices;
    // for (int i = 0; i < h_dup_indices.size(); i++) {
    //     std::cout << h_dup_indices[i] << " ";
    // }
    // std::cout << std::endl;

    // tmp_raw.resize(0);
    // tmp_raw.shrink_to_fit();

    for (int i = 0; i < arity; i++) {
        thrust::fill(DEFAULT_DEVICE_POLICY,
                     thrust::make_permutation_iterator(columns[i].raw_data,
                                                       dup_indices.begin()),
                     thrust::make_permutation_iterator(columns[i].raw_data,
                                                       dup_indices.end()),
                     UINT32_MAX);
    }

    // remove dup in raw_data
    uint32_t new_size = 0;
    for (int i = 0; i < arity; i++) {
        auto new_col_end =
            thrust::remove(DEFAULT_DEVICE_POLICY, columns[i].raw_data,
                           columns[i].raw_data + version_size, UINT32_MAX);
        new_size = new_col_end - columns[i].raw_data;
        columns[i].indexed = false;
    }
    newt_size = new_size;
    total_tuples = newt_size + full_size;
    for (size_t i = 0; i < arity; i++) {
        data[i].resize(total_tuples);
    }
    indexed = false;
    auto end = std::chrono::high_resolution_clock::now();
    dedup_time +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
}

void multi_hisa::fit() {
    total_tuples = newt_size + full_size;
    for (int i = 0; i < arity; i++) {
        data[i].resize(total_tuples);
        data[i].shrink_to_fit();
    }
}

void multi_hisa::new_to_delta() {
    // if (delta_size != 0) {
    //     throw std::runtime_error("delta is not empty before load new to
    //     delta");
    // }

    delta_size = newt_size;
    thrust::swap(newt_columns, delta_columns);
    // This is redundant
    for (int i = 0; i < arity; i++) {
        newt_columns[i].raw_data = nullptr;
    }
}

void column_match(VerticalColumnGpu &column_inner,
                  VerticalColumnGpu &column_outer,
                  device_pairs_t &matched_pair) {
    auto matched_size = matched_pair.size();
    // check if they agree on the value
    thrust::device_vector<bool> unmatched_flags(matched_size, true);
    device_data_t tmp_inner(matched_size);
    thrust::transform(DEFAULT_DEVICE_POLICY, matched_pair.begin(),
                      matched_pair.end(), tmp_inner.begin(),
                      [raw_inner = column_inner.raw_data.get()] __device__(
                          auto &t) { return raw_inner[t & 0xFFFFFFFF]; });
    // print tmp_inner
    // HOST_VECTOR<internal_data_type> h_tmp_inner = tmp_inner;
    // for (int i = 0; i < h_tmp_inner.size(); i++) {
    //     std::cout << h_tmp_inner[i] << " ";
    // }
    // std::cout << std::endl;

    thrust::transform(
        matched_pair.begin(), matched_pair.end(), unmatched_flags.begin(),
        [raw_outer = column_outer.raw_data.get(),
         raw_inner = column_inner.raw_data.get()] __device__(auto &t) {
            return raw_outer[t >> 32] != raw_inner[t & 0xFFFFFFFF];
        });
    // filter
    auto new_matched_pair_end =
        thrust::remove_if(matched_pair.begin(), matched_pair.end(),
                          unmatched_flags.begin(), thrust::identity<bool>());
    auto new_matched_pair_size = new_matched_pair_end - matched_pair.begin();
    matched_pair.resize(new_matched_pair_size);
}

void column_match(VerticalColumnGpu &column1, VerticalColumnGpu &column2,
                  device_indices_t &column1_indices,
                  device_indices_t &column2_indices,
                  DEVICE_VECTOR<bool> &unmatched) {
    auto matched_size = column1_indices.size();
    // check if they agree on the value
    unmatched.resize(matched_size);

    thrust::transform(
        DEFAULT_DEVICE_POLICY,
        thrust::make_permutation_iterator(column1.raw_data,
                                          column1_indices.begin()),
        thrust::make_permutation_iterator(column1.raw_data,
                                          column1_indices.end()),
        thrust::make_permutation_iterator(column2.raw_data,
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

void multi_hisa::persist_newt() {
    // clear the index of delta
    for (int i = 0; i < arity; i++) {
        delta_columns[i].sorted_indices.resize(0);
        delta_columns[i].sorted_indices.shrink_to_fit();
        delta_columns[i].unique_v.resize(0);
        delta_columns[i].unique_v.shrink_to_fit();
        delta_columns[i].clear_unique_v();
        delta_columns[i].unique_v_map = nullptr;
        delta_columns[i].raw_data = nullptr;
    }
    // merge newt to full
    if (full_size == 0) {
        full_size = newt_size;
        // thrust::swap(newt_columns, full_columns);
        for (int i = 0; i < arity; i++) {
            full_columns[i].raw_data = newt_columns[i].raw_data;
            newt_columns[i].raw_data = nullptr;
        }
        build_index_all(RelationVersion::FULL);
        return;
    }

    // difference newt and full, this a join
    // thrust::device_vector<internal_data_type> newt_full_diff(newt_size);
    device_pairs_t matched_pair;
    thrust::sequence(DEFAULT_DEVICE_POLICY, matched_pair.begin(),
                     matched_pair.end());
    device_data_t match_newt(newt_size);
    thrust::sequence(DEFAULT_DEVICE_POLICY, match_newt.begin(),
                     match_newt.end());
    column_join(full_columns[default_index_column],
                newt_columns[default_index_column], match_newt, matched_pair);
    for (size_t i = 0; i < arity; i++) {
        auto &newt_column = newt_columns[i];
        auto &full_column = full_columns[i];
        if (matched_pair.size() != 0) {
            column_match(full_column, newt_column, matched_pair);
            match_newt.resize(matched_pair.size());
            match_newt.shrink_to_fit();
            // populate the tuple need match later
            thrust::transform(
                DEFAULT_DEVICE_POLICY, matched_pair.begin(), matched_pair.end(),
                match_newt.begin(),
                cuda::proclaim_return_type<uint32_t>([] __device__(auto &t) {
                    return static_cast<uint32_t>(t >> 32);
                }));
        }
        if (match_newt.size() == 0) {
            break;
        }
    }

    // clear newt only keep match_newt
    if (match_newt.size() != 0) {
        device_data_t dup_newt_flags(newt_size, false);
        thrust::fill(DEFAULT_DEVICE_POLICY,
                     thrust::make_permutation_iterator(dup_newt_flags.begin(),
                                                       match_newt.begin()),
                     thrust::make_permutation_iterator(dup_newt_flags.begin(),
                                                       match_newt.end()),
                     true);
        auto new_newt_end =
            thrust::remove_if(DEFAULT_DEVICE_POLICY, newt_columns[0].raw_data,
                              newt_columns[0].raw_data + newt_size,
                              dup_newt_flags.begin(), thrust::identity<bool>());
        newt_size = new_newt_end - newt_columns[0].raw_data;
    }

    // sort and merge newt
    device_data_t merged_column(full_size + newt_size);
    device_data_t tmp_newt_v(newt_size);
    for (size_t i = 0; i < arity; i++) {
        auto &newt_column = newt_columns[i];
        auto &full_column = full_columns[i];

        newt_column.sorted_indices.resize(newt_size);
        thrust::copy(DEFAULT_DEVICE_POLICY, newt_column.raw_data,
                     newt_column.raw_data + newt_size, tmp_newt_v.begin());
        thrust::sequence(DEFAULT_DEVICE_POLICY,
                         newt_column.sorted_indices.begin(),
                         newt_column.sorted_indices.end(), full_size);
        thrust::sort_by_key(tmp_newt_v.begin(), tmp_newt_v.end(),
                            newt_column.sorted_indices.begin());

        // merge
        thrust::merge_by_key(
            DEFAULT_DEVICE_POLICY,
            thrust::make_permutation_iterator(
                full_column.raw_data, newt_column.sorted_indices.begin()),
            thrust::make_permutation_iterator(full_column.raw_data,
                                              newt_column.sorted_indices.end()),
            thrust::make_permutation_iterator(
                full_column.raw_data, full_column.sorted_indices.begin()),
            thrust::make_permutation_iterator(full_column.raw_data,
                                              full_column.sorted_indices.end()),
            newt_column.sorted_indices.begin(),
            full_column.sorted_indices.begin(), thrust::make_discard_iterator(),
            merged_column.begin());
        full_column.sorted_indices.swap(newt_column.sorted_indices);
        // minus size of full on all newt indices
        thrust::transform(DEFAULT_DEVICE_POLICY,
                          newt_column.sorted_indices.begin(),
                          newt_column.sorted_indices.end(),
                          thrust::make_constant_iterator(full_size),
                          newt_column.sorted_indices.begin(),
                          thrust::minus<internal_data_type>());
    }
    full_size += newt_size;

    // swap newt and delta
    delta_size = newt_size;
    newt_size = 0;
    // thrust::swap(newt_columns, delta_columns);
    for (int i = 0; i < arity; i++) {
        delta_columns[i].raw_data = newt_columns[i].raw_data;
        newt_columns[i].raw_data = nullptr;
    }

    // build indices on full
    build_index_all(RelationVersion::FULL, true);
    build_index_all(RelationVersion::DELTA, true);
}

void multi_hisa::clear() {
    for (int i = 0; i < arity; i++) {
        full_columns[i].raw_data = nullptr;
        full_columns[i].sorted_indices.resize(0);
        full_columns[i].sorted_indices.shrink_to_fit();
        full_columns[i].clear_unique_v();

        delta_columns[i].raw_data = nullptr;
        delta_columns[i].sorted_indices.resize(0);
        delta_columns[i].sorted_indices.shrink_to_fit();
        delta_columns[i].clear_unique_v();

        newt_columns[i].raw_data = nullptr;
        newt_columns[i].sorted_indices.resize(0);
        newt_columns[i].sorted_indices.shrink_to_fit();
        newt_columns[i].clear_unique_v();

        data[i].resize(0);
        data[i].shrink_to_fit();
    }
    total_tuples = 0;
}

void column_join(VerticalColumnGpu &inner_column,
                 VerticalColumnGpu &outer_column,
                 device_data_t &outer_tuple_indices,
                 device_pairs_t &matched_indices) {
    auto outer_size = outer_tuple_indices.size();

    device_ranges_t range_result(outer_tuple_indices.size());

    inner_column.unique_v_map->find(
        thrust::make_permutation_iterator(outer_column.raw_data,
                                          outer_tuple_indices.begin()),
        thrust::make_permutation_iterator(outer_column.raw_data,
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
    auto new_outer_tuple_end = thrust::remove_if(
        DEFAULT_DEVICE_POLICY, outer_tuple_indices.begin(),
        outer_tuple_indices.end(),
        thrust::make_transform_iterator(
            range_result.begin(),
            cuda::proclaim_return_type<bool>(
                [] __device__(auto &t) { return t == UINT32_MAX; })));
    outer_tuple_indices.resize(new_outer_tuple_end -
                               outer_tuple_indices.begin());
    // remove unmatched range result
    auto new_range_end = thrust::remove_if(
        DEFAULT_DEVICE_POLICY, range_result.begin(), range_result.end(),
        thrust::make_transform_iterator(
            range_result.begin(),
            cuda::proclaim_return_type<bool>(
                [] __device__(auto &t) { return t == UINT32_MAX; })));
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

void column_join(VerticalColumnGpu &inner_column,
                 VerticalColumnGpu &outer_column,
                 device_indices_t &outer_tuple_indices,
                 device_indices_t &matched_indices,
                 DEVICE_VECTOR<bool> &unmatched) {
    auto outer_size = outer_tuple_indices.size();
    unmatched.resize(outer_size);
    thrust::fill(DEFAULT_DEVICE_POLICY, unmatched.begin(), unmatched.end(),
                 false);

    device_ranges_t range_result(outer_tuple_indices.size());

    inner_column.unique_v_map->find(
        thrust::make_permutation_iterator(outer_column.raw_data,
                                          outer_tuple_indices.begin()),
        thrust::make_permutation_iterator(outer_column.raw_data,
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
    auto new_outer_tuple_end = thrust::remove_if(
        DEFAULT_DEVICE_POLICY, outer_tuple_indices.begin(),
        outer_tuple_indices.end(),
        thrust::make_transform_iterator(
            range_result.begin(),
            cuda::proclaim_return_type<bool>(
                [] __device__(auto &t) { return t == UINT32_MAX; })));
    outer_tuple_indices.resize(new_outer_tuple_end -
                               outer_tuple_indices.begin());
    // remove unmatched range result
    auto new_range_end = thrust::remove_if(
        DEFAULT_DEVICE_POLICY, range_result.begin(), range_result.end(),
        thrust::make_transform_iterator(
            range_result.begin(),
            cuda::proclaim_return_type<bool>(
                [] __device__(auto &t) { return t == UINT32_MAX; })));
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

} // namespace hisa

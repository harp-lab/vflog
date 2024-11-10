
#include "hisa.cuh"

#include <cstdint>
#include <thrust/count.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace vflog {

void multi_hisa::build_index(VerticalColumn &column,
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
        thrust::sequence(EXE_POLICY, column.sorted_indices.begin(),
                         column.sorted_indices.end());
        // sort all values in the column
        thrust::copy(EXE_POLICY, raw_ver_head, raw_ver_head + column.raw_size,
                     column_data.begin());
        // if (i != 0) {
        thrust::stable_sort_by_key(EXE_POLICY, column_data.begin(),
                                   column_data.end(),
                                   column.sorted_indices.begin());
        // }
        // count unique
        uniq_size = thrust::unique_count(EXE_POLICY, column_data.begin(),
                                         column_data.end());
        buffer->reserve(uniq_size);
        auto uniq_end = thrust::unique_by_key_copy(
            EXE_POLICY, column_data.begin(), column_data.end(),
            thrust::make_counting_iterator<uint32_t>(0),
            thrust::make_discard_iterator(), buffer->data());
    } else {
        if (unique_gather_flag) {
            uniq_size = thrust::unique_count(
                EXE_POLICY,
                thrust::make_permutation_iterator(
                    raw_ver_head, column.sorted_indices.begin()),
                thrust::make_permutation_iterator(raw_ver_head,
                                                  column.sorted_indices.end()));
            buffer->reserve(uniq_size);
        } else {
            buffer->reserve(column.raw_size);
        }
        auto uniq_end = thrust::unique_by_key_copy(
            EXE_POLICY,
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
        column.map_insert(raw_data, uniq_size, buffer);
    } else {
        throw std::runtime_error("unreal map not implemented");
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
            if (i != default_index_column) {
                build_index(columns[i], unique_offset, sorted);
            } else {
                build_index(columns[i], unique_offset, true);
            }
        }
    }
    indexed = true;

    sort_newt_clustered_index();
    persist_newt_clustered_index();
}

void multi_hisa::sort_newt_clustered_index() {
    auto start = std::chrono::high_resolution_clock::now();
    for (auto &clustered_index : clustered_indices_newt) {
        clustered_index.sorted_indices.resize(newt_size);
        thrust::sequence(EXE_POLICY, clustered_index.sorted_indices.begin(),
                         clustered_index.sorted_indices.end());
        for (int i = clustered_index.column_indices.size() - 1; i >= 0; i--) {
            auto &column = newt_columns[clustered_index.column_indices[i]];
            auto column_head =
                data[clustered_index.column_indices[i]].begin() + full_size;
            thrust::stable_sort_by_key(
                EXE_POLICY,
                thrust::make_permutation_iterator(
                    column_head, clustered_index.sorted_indices.begin()),
                thrust::make_permutation_iterator(
                    column_head, clustered_index.sorted_indices.end()),
                clustered_index.sorted_indices.begin());
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    this->sort_time +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
}

void multi_hisa::persist_newt_clustered_index() {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < clustered_indices_newt.size(); i++) {
        auto &cols_numbers = clustered_indices_newt[i].column_indices;
        auto &newt_sorted_indices = clustered_indices_newt[i].sorted_indices;
        auto &full_sorted_indices =
            newt_columns[cols_numbers[0]].sorted_indices;

        DEVICE_VECTOR<internal_data_type *> all_col_ptrs(arity);
        for (int j = 0; j < cols_numbers.size(); j++) {
            all_col_ptrs[j] = data[cols_numbers[j]].RAW_PTR + full_size;
        }
        device_data_t merged_column(full_size + newt_size);
        thrust::merge(EXE_POLICY, full_sorted_indices.begin(),
                      full_sorted_indices.end(), newt_sorted_indices.begin(),
                      newt_sorted_indices.end(), merged_column.begin(),
                      [arity = arity,
                       all_col_ptrs =
                           all_col_ptrs.data().get()] LAMBDA_TAG(auto full_idx,
                                                                 auto new_idx) {
                          for (int j = 0; j < arity; j++) {
                              if (all_col_ptrs[j][full_idx] !=
                                  all_col_ptrs[j][new_idx]) {
                                  return all_col_ptrs[j][full_idx] <
                                         all_col_ptrs[j][new_idx];
                              }
                          }
                          return false;
                      });
        full_sorted_indices.swap(merged_column);
    }
    auto end = std::chrono::high_resolution_clock::now();
    this->merge_time +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
}

void multi_hisa::build_cluster_index() {
    auto start = std::chrono::high_resolution_clock::now();
    for (auto &clustered_index : clustered_indices_full) {
        if (clustered_index.joinable) {
            // TODO: implement joinable clustered index
            throw std::runtime_error(
                "joinable clustered index not implemented");
        }
        // create a buffer has full_size
        buffer->reserve(full_size);
        // fill the buffer to 0
        thrust::fill(EXE_POLICY, buffer->data(), buffer->data() + full_size, 0);
        for (int i : clustered_index.column_indices) {
            auto &column = full_columns[i];
            auto raw_head = get_raw_data_ptrs(FULL, i);
            // hash digest use current buffer value add the
            thrust::transform(EXE_POLICY, raw_head, raw_head + full_size,
                              buffer->data(), buffer->data(),
                              [] LAMBDA_TAG(auto &raw, auto &buf_h) {
                                  return combine_hash(buf_h, murmur3_32(raw));
                              });
        }
        // store the buffer to hash map of index
        if (clustered_index.unique_v_map) {
            clustered_index.unique_v_map->clear();
        } else {
            clustered_index.unique_v_map = nullptr;
            clustered_index.unique_v_map = CREATE_V_MAP(full_size);
        }
        // TODO: solve hash collision
        auto insertpair_begin = thrust::make_transform_iterator(
            thrust::make_zip_iterator(thrust::make_tuple(
                buffer->data(), thrust::make_counting_iterator<uint32_t>(0))),
            cuda::proclaim_return_type<GpuMapPair>([] LAMBDA_TAG(auto &pair) {
                return HASH_NAMESPACE::make_pair(
                    thrust::get<0>(pair),
                    (static_cast<uint64_t>(thrust::get<1>(pair)) << 32) + 1);
            }));
        clustered_index.unique_v_map->insert(insertpair_begin,
                                             insertpair_begin + full_size);
    }
}

} // namespace vflog

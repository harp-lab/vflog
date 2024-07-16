/**
 * @file persistence.cu
 * @brief Implementation of the persistence (store newt into full/delta)
 */

#include "hisa.cuh"

#include <thrust/merge.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h> 
#include <thrust/unique.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/iterator/discard_iterator.h>

namespace vflog {

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

    newt_full_deduplicate();

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

} // namespace vflog

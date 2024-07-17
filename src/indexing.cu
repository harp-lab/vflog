
#include "hisa.cuh"

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
        thrust::copy(EXE_POLICY, raw_ver_head,
                     raw_ver_head + column.raw_size, column_data.begin());
        // if (i != 0) {
        thrust::stable_sort_by_key(EXE_POLICY, column_data.begin(),
                                   column_data.end(),
                                   column.sorted_indices.begin());
        // }
        // count unique
        uniq_size = thrust::unique_count(
            EXE_POLICY, column_data.begin(), column_data.end());
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
        throw std::runtime_error("not implemented");
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
}
} // namespace vflog


#include "hisa.cuh"

#include <thrust/count.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#define CREATE_V_MAP(uniq_size)                                                \
    std::make_unique<vflog::GpuMap>(                                           \
        uniq_size, DEFAULT_LOAD_FACTOR,                                        \
        cuco::empty_key<vflog::internal_data_type>{UINT32_MAX},                \
        cuco::empty_value<vflog::offset_type>{UINT32_MAX})
#define HASH_NAMESPACE cuco

namespace vflog {

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
} // namespace vflog

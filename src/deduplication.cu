
#include "hisa.cuh"

#include <iostream>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace vflog {

void multi_hisa::newt_self_deduplicate() {
    // sort the raw data of each column in newt
    auto start = std::chrono::high_resolution_clock::now();
    VersionedColumns &columns = newt_columns;
    auto version_size = newt_size;
    // radix sort the raw data of each column
    device_data_t sorted_indices(version_size);
    // std::cout << "deduplicate newt size: " << version_size << std::endl;
    thrust::sequence(EXE_POLICY, sorted_indices.begin(), sorted_indices.end());
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
        thrust::gather(EXE_POLICY, sorted_indices.begin(), sorted_indices.end(),
                       column_data, tmp_raw_ptr);
        thrust::stable_sort_by_key(EXE_POLICY, tmp_raw_ptr,
                                   tmp_raw_ptr + version_size,
                                   sorted_indices.begin());
    }

    // sort the default index column
    auto &column = columns[default_index_column];
    auto column_data = data[default_index_column].begin() + column.raw_offset;
    thrust::gather(EXE_POLICY, sorted_indices.begin(), sorted_indices.end(),
                   column_data,
                   tmp_raw_ptr); // gather the column data
    thrust::stable_sort_by_key(EXE_POLICY, tmp_raw_ptr,
                               tmp_raw_ptr + version_size,
                               sorted_indices.begin());

    device_bitmap_t dup_flags(version_size, false);
    // check duplicates tuple
    DEVICE_VECTOR<internal_data_type *> all_col_ptrs(arity);
    for (int i = 0; i < arity; i++) {
        all_col_ptrs[i] = data[i].RAW_PTR + columns[i].raw_offset;
    }
    thrust::transform(
        thrust::make_counting_iterator<uint32_t>(0),
        thrust::make_counting_iterator<uint32_t>(dup_flags.size()),
        dup_flags.begin(),
        [all_col_ptrs = all_col_ptrs.data(), total = dup_flags.size(),
         sorted_indices = sorted_indices.RAW_PTR,
         arity = arity] LAMBDA_TAG(uint32_t i) -> bool {
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
        EXE_POLICY, sorted_indices.begin(), sorted_indices.end(),
        dup_flags.begin(), thrust::identity<bool>());
    auto new_sorted_indices_size =
        new_sorted_indices_end - sorted_indices.begin();
    sorted_indices.resize(new_sorted_indices_size);
    sorted_indices.shrink_to_fit();
    // tmp_raw.resize(new_sorted_indices_size);
    // gather the newt data
    for (int i = 0; i < arity; i++) {
        auto column_data = data[i].begin() + columns[i].raw_offset;
        thrust::gather(EXE_POLICY, sorted_indices.begin(), sorted_indices.end(),
                       column_data, tmp_raw_ptr);
        thrust::copy(EXE_POLICY, tmp_raw_ptr,
                     tmp_raw_ptr + new_sorted_indices_size, column_data);
        columns[i].raw_size = new_sorted_indices_size;
    }
    // set the 0-th column's sorted indices
    thrust::sequence(EXE_POLICY, sorted_indices.begin(), sorted_indices.end());
    columns[default_index_column].sorted_indices.swap(sorted_indices);
    newt_size = new_sorted_indices_size;
    total_tuples = newt_size + full_size;
}

inline __device__ __host__ bool
tuple_compare(uint32_t **full, uint32_t full_idx, uint32_t **newt,
              uint32_t newt_idx, int arity, int default_index_column) {
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

inline __device__ __host__ bool tuple_eq(uint32_t **full, uint32_t full_idx,
                                         uint32_t **newt, uint32_t newt_idx,
                                         int arity, int default_index_column) {
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

void multi_hisa::newt_full_deduplicate() {
    auto before_dedup = std::chrono::high_resolution_clock::now();
    // difference newt and full
    device_ranges_t matched_ranges(newt_size);
    // do a column match on the default index column
    auto default_col_newt_raw_begin =
        data[default_index_column].begin() + full_size;
    full_columns[default_index_column].map_find(
        default_col_newt_raw_begin, default_col_newt_raw_begin + newt_size,
        matched_ranges.begin());
    // std::cout << "matched_ranges size: " << matched_ranges.size() <<
    // std::endl; for each range check if all column value is the same
    DEVICE_VECTOR<internal_data_type *> all_col_news_ptrs(arity);
    DEVICE_VECTOR<internal_data_type *> all_col_fulls_ptrs(arity);
    for (int i = 0; i < arity; i++) {
        all_col_news_ptrs[i] = data[i].RAW_PTR + full_size;
        all_col_fulls_ptrs[i] = data[i].RAW_PTR;
    }
    device_bitmap_t dup_newt_flags(newt_size, false);
    thrust::transform(
        thrust::make_counting_iterator<uint32_t>(0),
        thrust::make_counting_iterator<uint32_t>(newt_size),
        matched_ranges.begin(), dup_newt_flags.begin(),
        [all_col_news_ptrs = all_col_news_ptrs.RAW_PTR,
         all_col_fulls_ptrs = all_col_fulls_ptrs.RAW_PTR,
         all_col_full_idx_ptrs =
             full_columns[default_index_column].sorted_indices.RAW_PTR,
         default_index_column = default_index_column,
         arity = arity] LAMBDA_TAG(auto newt_index, uint64_t range) -> bool {
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
    auto dup_size = thrust::count(EXE_POLICY, dup_newt_flags.begin(),
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
                EXE_POLICY, newt_begin, newt_begin + newt_size,
                dup_newt_flags.begin(), thrust::identity<bool>());
            newt_columns[i].raw_size = new_newt_end - newt_begin;
        }
        newt_size = newt_size - dup_size;
    }
}

void multi_hisa::diff(multi_hisa &other, RelationVersion version,
                      device_indices_t &diff_indices) {
    // deduplicate version of others use diff_indices as id, with full of this
    // version
    auto before_dedup = std::chrono::high_resolution_clock::now();
    // difference other.ver and full
    auto other_ver_size = other.get_versioned_size(version);
    device_ranges_t matched_ranges(other_ver_size);
    // do a column match on the default index column
    auto default_col_other_raw_begin =
        other.get_raw_data_ptrs(version, default_index_column);
    if (diff_indices.size() != 0) {
        full_columns[default_index_column].map_find(
            thrust::make_permutation_iterator(default_col_other_raw_begin,
                                              diff_indices.begin()),
            thrust::make_permutation_iterator(default_col_other_raw_begin,
                                              diff_indices.end()),
            matched_ranges.begin());
    } else {
        // std::cout << "other_ver_size: " << other_ver_size << std::endl;
        default_col_other_raw_begin =
            other.data[default_index_column].data().get() + other.full_size;
        full_columns[default_index_column].map_find(
            default_col_other_raw_begin,
            default_col_other_raw_begin + other_ver_size,
            matched_ranges.begin());
    }
    // std::cout << "matched_ranges size: " << matched_ranges.size() <<
    // std::endl; for each range check if all column value is the same
    DEVICE_VECTOR<internal_data_type *> all_col_others_ptrs(arity);
    DEVICE_VECTOR<internal_data_type *> all_col_fulls_ptrs(arity);

    for (int i = 0; i < arity; i++) {
        all_col_others_ptrs[i] = other.data[i].data().get() + other.full_size;
        all_col_fulls_ptrs[i] = data[i].RAW_PTR;
    }

    device_bitmap_t dup_other_flags(other_ver_size, false);
    thrust::transform(
        thrust::make_counting_iterator<uint32_t>(0),
        thrust::make_counting_iterator<uint32_t>(other_ver_size),
        matched_ranges.begin(), dup_other_flags.begin(),
        [all_col_others_ptrs = all_col_others_ptrs.RAW_PTR,
         all_col_fulls_ptrs = all_col_fulls_ptrs.RAW_PTR,
         all_col_full_idx_ptrs =
             full_columns[default_index_column].sorted_indices.RAW_PTR,
         default_index_column = default_index_column,
         arity = arity] LAMBDA_TAG(auto other_index, uint64_t range) -> bool {
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
                if (tuple_eq(all_col_fulls_ptrs, full_index,
                             all_col_others_ptrs, other_index, arity,
                             default_index_column)) {
                    return true;
                }
                if (tuple_compare(all_col_fulls_ptrs, full_index,
                                  all_col_others_ptrs, other_index, arity,
                                  default_index_column)) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            return false;
        });

    auto dup_size = thrust::count(EXE_POLICY, dup_other_flags.begin(),
                                  dup_other_flags.end(), true);
    // std::cout << "dup size: " << dup_size << std::endl;
    auto after_dedup = std::chrono::high_resolution_clock::now();
    dedup_time += std::chrono::duration_cast<std::chrono::microseconds>(
                      after_dedup - before_dedup)
                      .count();
    // clear other only keep match_other
    if (dup_size != 0) {
        for (size_t i = 0; i < arity; i++) {
            auto other_begin = other.data[i].data().get() + other.full_size;
            auto new_other_end = thrust::remove_if(
                EXE_POLICY, other_begin, other_begin + other_ver_size,
                dup_other_flags.begin(), thrust::identity<bool>());
            other.get_versioned_columns(version)[i].raw_size =
                new_other_end - other_begin;
        }
        other.set_versioned_size(version, other_ver_size - dup_size);
    }
    std::cout << "default_index_column: " << default_index_column << std::endl;

}

} // namespace vflog

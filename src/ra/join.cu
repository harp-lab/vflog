
#include "ra.cuh"

#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>

namespace vflog {

void column_join(multi_hisa &inner, RelationVersion inner_ver, size_t inner_idx,
                 multi_hisa &outer, RelationVersion outer_ver, size_t outer_idx,
                 device_indices_t &outer_tuple_indices,
                 device_indices_t &matched_indices,
                 DEVICE_VECTOR<bool> &unmatched_outer) {
    if (outer_tuple_indices.size() == 0) {
        return;
    }

    auto &inner_column = inner.get_versioned_columns(inner_ver)[inner_idx];
    auto &outer_column = outer.get_versioned_columns(outer_ver)[outer_idx];
    auto outer_size = outer_tuple_indices.size();
    // unmatched_outer.resize(outer_size);
    // thrust::fill(DEFAULT_DEVICE_POLICY, unmatched_outer.begin(),
    //              unmatched_outer.end(), false);

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
    // thrust::transform(DEFAULT_DEVICE_POLICY, range_result.begin(),
    //                   range_result.end(), unmatched_outer.begin(),
    //                   [] __device__(auto &t) { return t == UINT32_MAX; });
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

/**
 * filter each joined columns
 */
// void binary_join_filter(HOST_VECTOR<column_ref> &columns,
//                         device_data_t &outer_indices_buf,
//                         device_ranges_t &matched_ranges) {

//     // find the largest column
//     auto largest_col_size = 0;
//     for (auto &col : columns) {
//         auto &rel = col.relation.get();
//         auto &column = rel.get_versioned_columns(col.version)[col.index];
//         if (column.size() > largest_col_size) {
//             largest_col_size = column.size();
//         }
//     }
//     outer_indices_buf.resize(largest_col_size);
//     matched_ranges.resize(largest_col_size);
//     device_bitmap_t result_bitmap(largest_col_size);

//     for (size_t i = 0; i < columns.size() - 1; i++) {

//         auto &outer_rel = columns[i].relation.get();
//         auto &inner_rel = columns[i + 1].relation.get();
//         auto &outer_bitmap = columns[i].selected;
//         auto outer_size = outer_rel.get_versioned_size(columns[i].version);
//         auto outer_column = outer_rel.get_versioned_columns(
//             columns[i].version)[columns[i].index];
//         auto outer_raw_ver_begin =
//             outer_rel.data[outer_column.column_idx].begin() +
//             outer_column.raw_offset;
//         auto &inner_bitmap = columns[i + 1].selected;
//         auto inner_size = inner_rel.get_versioned_size(columns[i +
//         1].version);

//         // find the outer indices
//         auto &inner_uniq_map =
//             inner_rel
//                 .get_versioned_columns(
//                     columns[i + 1].version)[columns[i + 1].index]
//                 .unique_v_map;
//         inner_uniq_map->find(outer_raw_ver_begin,
//                              outer_raw_ver_begin + outer_size,
//                              matched_ranges.begin());
//         // update the outer bitmap
//         thrust::transform(DEFAULT_DEVICE_POLICY, matched_ranges.begin(),
//                           matched_ranges.end(), outer_bitmap.get().begin(),
//                           outer_bitmap.get().begin(),
//                           [] __device__(auto &t, auto &b) {
//                               return t == UINT32_MAX ? false : b;
//                           });
//         // update the inner bitmap
//         // fill result
//         thrust::fill(DEFAULT_DEVICE_POLICY, result_bitmap.begin(),
//                      result_bitmap.end(), false);
//         thrust::for_each(
//             DEFAULT_DEVICE_POLICY, matched_ranges.begin(),
//             matched_ranges.end(), [res_bits = result_bitmap.data().get()]
//             __device__(auto &range) {
//                 if (range != UINT32_MAX) {
//                     auto start_pos = range >> 32;
//                     auto size = range & 0xFFFFFFFF;
//                     for (int i = 0; i < size; i++) {
//                         res_bits[start_pos + i] = true;
//                     }
//                 }
//             });
//         thrust::transform(DEFAULT_DEVICE_POLICY, inner_bitmap.get().begin(),
//                           inner_bitmap.get().end(), result_bitmap.begin(),
//                           inner_bitmap.get().begin(),
//                           [] __device__(auto &b, auto &r) { return b && r;
//                           });
//     }
// }

void column_join(multi_hisa &inner, RelationVersion inner_ver, size_t inner_idx,
                 multi_hisa &outer, RelationVersion outer_ver, size_t outer_idx,
                 host_buf_ref_t &cached_indices, std::string meta_var,
                 std::shared_ptr<device_indices_t> matched_indices,
                 bool pop_outer) {
    auto &outer_tuple_indices = cached_indices[meta_var];
    if (outer_tuple_indices->size() == 0) {
        // clear all other in cache
        for (auto &meta : cached_indices) {
            meta.second->resize(0);
        }
        matched_indices->resize(0);
        return;
    }
    device_ranges_t matched_ranges(outer_tuple_indices->size());
    // gather the outer values
    auto &outer_column = outer.get_versioned_columns(outer_ver)[outer_idx];
    auto &inner_column = inner.get_versioned_columns(inner_ver)[inner_idx];
    if (inner.get_versioned_size(inner_ver) == 0) {
        return;
    }
    auto outer_raw_begin =
        outer.data[outer_idx].begin() + outer_column.raw_offset;
    inner_column.unique_v_map->find(
        thrust::make_permutation_iterator(outer_raw_begin,
                                          outer_tuple_indices->begin()),
        thrust::make_permutation_iterator(outer_raw_begin,
                                          outer_tuple_indices->end()),
        matched_ranges.begin());
    // if (pop_outer) {
    //     outer_tuple_indices->clear();
    //     outer_tuple_indices->shrink_to_fit();
    // }
    // clear the unmatched tuples
    for (auto &meta : cached_indices) {
        auto &outer_ts = meta.second;
        thrust::transform(
            DEFAULT_DEVICE_POLICY, matched_ranges.begin(), matched_ranges.end(),
            outer_ts->begin(), outer_ts->begin(),
            [] __device__(auto &range, auto &outer_tuple_index) {
                return range == UINT32_MAX ? UINT32_MAX : outer_tuple_index;
            });

        auto new_outer_tuple_end =
            thrust::remove(DEFAULT_DEVICE_POLICY, outer_ts->begin(),
                           outer_ts->end(), UINT32_MAX);
        outer_ts->resize(new_outer_tuple_end - outer_ts->begin());
    }
    auto new_range_end =
        thrust::remove(DEFAULT_DEVICE_POLICY, matched_ranges.begin(),
                       matched_ranges.end(), UINT32_MAX);
    matched_ranges.resize(new_range_end - matched_ranges.begin());

    // materialize the matched_indices
    device_data_t size_vec(matched_ranges.size());
    thrust::transform(
        DEFAULT_DEVICE_POLICY, matched_ranges.begin(), matched_ranges.end(),
        size_vec.begin(),
        [] __device__(auto &t) { return static_cast<uint32_t>(t); });

    device_ranges_t &offset_vec = matched_ranges;
    thrust::transform(
        DEFAULT_DEVICE_POLICY, offset_vec.begin(), offset_vec.end(),
        offset_vec.begin(),
        [] __device__(auto &t) { return static_cast<uint64_t>(t >> 32); });

    uint32_t total_matched_size =
        thrust::reduce(DEFAULT_DEVICE_POLICY, size_vec.begin(), size_vec.end());
    device_data_t size_offset_tmp(size_vec.size());
    thrust::exclusive_scan(DEFAULT_DEVICE_POLICY, size_vec.begin(),
                           size_vec.end(), size_offset_tmp.begin());

    device_data_t materialized_outer(total_matched_size);
    matched_indices->resize(total_matched_size);
    auto cnt = 0;
    for (auto &meta : cached_indices) {
        materialized_outer.resize(total_matched_size);
        auto &outer_ts = meta.second;
        bool is_outer = meta.first == meta_var;
        if (is_outer) {
            // in the first iteration also update the matched inner indices
            if (!pop_outer) {
                thrust::for_each(
                    DEFAULT_DEVICE_POLICY,
                    thrust::make_zip_iterator(thrust::make_tuple(
                        outer_ts->begin(), offset_vec.begin(), size_vec.begin(),
                        size_offset_tmp.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        outer_ts->end(), offset_vec.end(), size_vec.end(),
                        size_offset_tmp.end())),
                    [res_inner = matched_indices->data().get(),
                     res_outer = materialized_outer.data().get(),
                     inner_sorted_idx = inner_column.sorted_indices.data()
                                            .get()] __device__(auto &t) {
                        auto outer_pos = thrust::get<0>(t);
                        auto &inner_pos = thrust::get<1>(t);
                        auto &size = thrust::get<2>(t);
                        auto &start = thrust::get<3>(t);
                        for (int i = 0; i < size; i++) {
                            res_inner[start + i] =
                                inner_sorted_idx[inner_pos + i];
                            res_outer[start + i] = outer_pos;
                        }
                    });
                outer_ts->swap(materialized_outer);
            } else {
                // no need care about the outer any more don't touch it
                // only mod the inner indices
                thrust::for_each(
                    DEFAULT_DEVICE_POLICY,
                    thrust::make_zip_iterator(
                        thrust::make_tuple(offset_vec.begin(), size_vec.begin(),
                                           size_offset_tmp.begin())),
                    thrust::make_zip_iterator(
                        thrust::make_tuple(offset_vec.end(), size_vec.end(),
                                           size_offset_tmp.end())),
                    [res_inner = matched_indices->data().get(),
                     inner_sorted_idx = inner_column.sorted_indices.data()
                                            .get()] __device__(auto &t) {
                        auto &inner_pos = thrust::get<0>(t);
                        auto &size = thrust::get<1>(t);
                        auto &start = thrust::get<2>(t);
                        for (int i = 0; i < size; i++) {
                            res_inner[start + i] =
                                inner_sorted_idx[inner_pos + i];
                        }
                    });
            }
        } else {
            // only update the outer indices
            thrust::for_each(
                DEFAULT_DEVICE_POLICY,
                thrust::make_zip_iterator(
                    thrust::make_tuple(outer_ts->begin(), size_vec.begin(),
                                       size_offset_tmp.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(
                    outer_ts->end(), size_vec.end(), size_offset_tmp.end())),
                [res_outer =
                     materialized_outer.data().get()] __device__(auto &t) {
                    auto outer_pos = thrust::get<0>(t);
                    auto &size = thrust::get<1>(t);
                    auto &start = thrust::get<2>(t);
                    for (int i = 0; i < size; i++) {
                        res_outer[start + i] = outer_pos;
                    }
                });
            outer_ts->swap(materialized_outer);
        }
        cnt++;
    }
    if (pop_outer) {
        cached_indices.erase(meta_var);
    }
}

} // namespace vflog

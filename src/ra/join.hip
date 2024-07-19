
#include "ra.cuh"

#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>

namespace vflog {

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
    inner_column.map_find(
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
            EXE_POLICY, matched_ranges.begin(), matched_ranges.end(),
            outer_ts->begin(), outer_ts->begin(),
            [] LAMBDA_TAG(auto &range, auto &outer_tuple_index) {
                return range == UINT32_MAX ? UINT32_MAX : outer_tuple_index;
            });

        auto new_outer_tuple_end = thrust::remove(EXE_POLICY, outer_ts->begin(),
                                                  outer_ts->end(), UINT32_MAX);
        outer_ts->resize(new_outer_tuple_end - outer_ts->begin());
    }
    auto new_range_end = thrust::remove(EXE_POLICY, matched_ranges.begin(),
                                        matched_ranges.end(), UINT32_MAX);
    matched_ranges.resize(new_range_end - matched_ranges.begin());

    // materialize the matched_indices
    device_data_t size_vec(matched_ranges.size());
    thrust::transform(EXE_POLICY, matched_ranges.begin(), matched_ranges.end(),
                      size_vec.begin(), [] LAMBDA_TAG(auto &t) {
                          return static_cast<unsigned int>(t);
                      });

    device_ranges_t &offset_vec = matched_ranges;
    thrust::transform(
        EXE_POLICY, offset_vec.begin(), offset_vec.end(), offset_vec.begin(),
        [] LAMBDA_TAG(auto &t) { return static_cast<unsigned long long>(t >> 32); });

    unsigned int total_matched_size =
        thrust::reduce(EXE_POLICY, size_vec.begin(), size_vec.end());
    device_data_t size_offset_tmp(size_vec.size());
    thrust::exclusive_scan(EXE_POLICY, size_vec.begin(), size_vec.end(),
                           size_offset_tmp.begin());

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
                    EXE_POLICY,
                    thrust::make_zip_iterator(thrust::make_tuple(
                        outer_ts->begin(), offset_vec.begin(), size_vec.begin(),
                        size_offset_tmp.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        outer_ts->end(), offset_vec.end(), size_vec.end(),
                        size_offset_tmp.end())),
                    [res_inner = matched_indices->RAW_PTR,
                     res_outer = materialized_outer.RAW_PTR,
                     inner_sorted_idx = inner_column.sorted_indices
                                            .RAW_PTR] LAMBDA_TAG(auto &t) {
                        auto outer_pos = thrust::get<0>(t);
                        auto &inner_pos = thrust::get<1>(t);
                        auto &size = thrust::get<2>(t);
                        auto &start = thrust::get<3>(t);
                        // printf("outer_pos: %d, inner_pos: %d, size: %d, start: %d\n", outer_pos, inner_pos, size, start);
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
                    EXE_POLICY,
                    thrust::make_zip_iterator(
                        thrust::make_tuple(offset_vec.begin(), size_vec.begin(),
                                           size_offset_tmp.begin())),
                    thrust::make_zip_iterator(
                        thrust::make_tuple(offset_vec.end(), size_vec.end(),
                                           size_offset_tmp.end())),
                    [res_inner = matched_indices->RAW_PTR,
                     inner_sorted_idx = inner_column.sorted_indices
                                            .RAW_PTR] LAMBDA_TAG(auto &t) {
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
                EXE_POLICY,
                thrust::make_zip_iterator(
                    thrust::make_tuple(outer_ts->begin(), size_vec.begin(),
                                       size_offset_tmp.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(
                    outer_ts->end(), size_vec.end(), size_offset_tmp.end())),
                [res_outer = materialized_outer.RAW_PTR] LAMBDA_TAG(auto &t) {
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


#include "ram.cuh"
#include "utils.cuh"

#include <cstdint>
#include <cstdio>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>

namespace vflog {

void materialize_matched_range(
    vflog::VerticalColumn &inner_column, vflog::VerticalColumn &outer_column,
    device_ranges_t &matched_ranges, host_buf_ref_t &cached_indices,
    std::string outer_meta_var,
    std::shared_ptr<device_indices_t> matched_indices, bool pop_outer) {
    device_data_t size_vec(matched_ranges.size());
    thrust::transform(EXE_POLICY, matched_ranges.begin(), matched_ranges.end(),
                      size_vec.begin(), [] LAMBDA_TAG(auto &t) {
                          return static_cast<uint32_t>(t);
                      });

    device_ranges_t &offset_vec = matched_ranges;
    thrust::transform(
        EXE_POLICY, offset_vec.begin(), offset_vec.end(), offset_vec.begin(),
        [] LAMBDA_TAG(auto &t) { return static_cast<uint64_t>(t >> 32); });

    uint32_t total_matched_size =
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
        bool is_outer = meta.first == outer_meta_var;
        auto size_diff = total_matched_size - outer_ts->size();
        if (is_outer) {
            // in the first iteration also update the matched inner indices
            if (!pop_outer) {
                thrust::for_each(
                    EXE_POLICY, thrust::make_counting_iterator<unsigned int>(0),
                    thrust::make_counting_iterator<unsigned int>(
                        total_matched_size),
                    [res_inner = matched_indices->RAW_PTR,
                     res_outer = materialized_outer.RAW_PTR,
                     inner_sorted_idx = inner_column.sorted_indices.RAW_PTR,
                     outer_pos = outer_ts->RAW_PTR,
                     inner_offset = offset_vec.RAW_PTR,
                     size_offset = size_offset_tmp.RAW_PTR,
                     outer_size = outer_ts->size(),
                     size_diff = size_diff] LAMBDA_TAG(unsigned int res_pos) {
                        int upper_idx = res_pos + 1;
                        int lower_idx = res_pos - size_diff;
                        if (lower_idx < 0) {
                            lower_idx = 0;
                        }
                        if (upper_idx > outer_size) {
                            upper_idx = outer_size;
                        }
                        // binary search the lower bound
                        unsigned int *found_outer_ptr = thrust::lower_bound(
                            thrust::seq, size_offset + lower_idx,
                            size_offset + upper_idx, res_pos);
                        unsigned int found_outer_idx =
                            found_outer_ptr - size_offset;
                        if (*found_outer_ptr != res_pos) {
                            found_outer_idx--;
                        }
                        // printf("%u : found_outer_idx: %d\n", res_pos,
                        // found_outer_idx);
                        unsigned int outer_val = outer_pos[found_outer_idx];
                        unsigned int inner_idx = res_pos -
                                                 size_offset[found_outer_idx] +
                                                 inner_offset[found_outer_idx];
                        // unsigned int inner_val = inner_sorted_idx[inner_idx];
                        // printf("inner_val: %u outer_val: %u\n", inner_val,
                        //        outer_val);
                        res_inner[res_pos] = inner_sorted_idx[inner_idx];
                        res_outer[res_pos] = outer_val;
                    });
                // std::cout << "outer size: " << outer_ts->size() << std::endl;

                outer_ts->swap(materialized_outer);
            } else {
                // no need care about the outer any more don't touch it
                // only mod the inner indices

                // instead iterate the outer, we distribute the job based on the
                // result position
                thrust::for_each(
                    EXE_POLICY, thrust::make_counting_iterator<unsigned int>(0),
                    thrust::make_counting_iterator<unsigned int>(
                        total_matched_size),
                    [res_inner = matched_indices->RAW_PTR,
                     inner_sorted_idx = inner_column.sorted_indices.RAW_PTR,
                     inner_offset = offset_vec.RAW_PTR,
                     size_offset = size_offset_tmp.RAW_PTR,
                     outer_size = outer_ts->size(),
                     size_diff = size_diff] LAMBDA_TAG(unsigned int res_pos) {
                        int upper_idx = res_pos + 1;
                        int lower_idx = res_pos - size_diff;
                        if (lower_idx < 0) {
                            lower_idx = 0;
                        }
                        if (upper_idx > outer_size) {
                            upper_idx = outer_size;
                        }
                        // binary search the lower bound
                        unsigned int *found_outer_ptr = thrust::lower_bound(
                            thrust::seq, size_offset + lower_idx,
                            size_offset + upper_idx, res_pos);
                        unsigned int found_outer_idx =
                            found_outer_ptr - size_offset;
                        if (*found_outer_ptr != res_pos) {
                            found_outer_idx--;
                        }
                        unsigned int inner_idx = res_pos -
                                                 size_offset[found_outer_idx] +
                                                 inner_offset[found_outer_idx];
                        res_inner[res_pos] = inner_sorted_idx[inner_idx];
                    });
            }
        } else {
            // only update the outer indices

            thrust::for_each(
                EXE_POLICY, thrust::make_counting_iterator<unsigned int>(0),
                thrust::make_counting_iterator<unsigned int>(
                    total_matched_size),
                [res_outer = materialized_outer.RAW_PTR,
                 outer_pos = outer_ts->RAW_PTR,
                 size_offset = size_offset_tmp.RAW_PTR,
                 outer_size = outer_ts->size(),
                 size_diff = size_diff] LAMBDA_TAG(unsigned int res_pos) {
                    int upper_idx = res_pos + 1;
                    int lower_idx = res_pos - size_diff;
                    if (lower_idx < 0) {
                        lower_idx = 0;
                    }
                    if (upper_idx > outer_size) {
                        upper_idx = outer_size;
                    }
                    // binary search the lower bound
                    unsigned int *found_outer_ptr = thrust::lower_bound(
                        thrust::seq, size_offset + lower_idx,
                        size_offset + upper_idx, res_pos);
                    unsigned int found_outer_idx =
                        found_outer_ptr - size_offset;
                    if (*found_outer_ptr != res_pos) {
                        found_outer_idx--;
                    }
                    res_outer[res_pos] = outer_pos[found_outer_idx];
                });

            outer_ts->swap(materialized_outer);
        }
        cnt++;
    }
    if (pop_outer) {
        cached_indices.erase(outer_meta_var);
    }
}

void column_join(multi_hisa &inner, RelationVersion inner_ver, size_t inner_idx,
                 multi_hisa &outer, RelationVersion outer_ver, size_t outer_idx,
                 host_buf_ref_t &cached_indices, std::string outer_meta_var,
                 std::shared_ptr<device_indices_t> matched_indices,
                 bool pop_outer) {
    auto &outer_tuple_indices = cached_indices[outer_meta_var];
    if (outer_tuple_indices->size() == 0) {
        // clear all other in cache
        for (auto &meta : cached_indices) {
            meta.second->resize(0);
        }
        matched_indices->resize(0);
        return;
    }

    // gather the outer values
    auto &outer_column = outer.get_versioned_columns(outer_ver)[outer_idx];
    auto &inner_column = inner.get_versioned_columns(inner_ver)[inner_idx];
    if (inner.get_versioned_size(inner_ver) == 0) {
        return;
    }

    device_ranges_t matched_ranges(outer_tuple_indices->size());
    auto outer_raw_begin =
        outer.data[outer_idx].begin() + outer_column.raw_offset;
    inner_column.map_find(thrust::make_permutation_iterator(
                              outer_raw_begin, outer_tuple_indices->begin()),
                          thrust::make_permutation_iterator(
                              outer_raw_begin, outer_tuple_indices->end()),
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
    materialize_matched_range(inner_column, outer_column, matched_ranges,
                              cached_indices, outer_meta_var, matched_indices,
                              pop_outer);
}

namespace ram {
void FreeJoinOperator::execute(RelationalAlgebraMachine &ram) {
    // the outer most relation
    auto outer_rel = ram.rels[columns[0].rel];
    auto outer_ver = columns[0].version;
    auto outer_col =
        outer_rel->get_versioned_columns(outer_ver)[columns[0].idx];
    auto outer_size = outer_rel->get_versioned_size(outer_ver);
    auto outer_indices = ram.cached_indices[meta_vars[0]];
    auto outer_raw_begin =
        outer_rel->data[columns[0].idx].begin() + outer_col.raw_offset;

    // match every value on outer column in the other column
    device_bitmap_t matched_outer_bitmap(outer_size, true);
    device_bitmap_t matched_outer_bitmap_tmp(outer_size, true);
    for (int i = 1; i < columns.size(); i++) {
        auto inner_rel = ram.rels[columns[i].rel];
        auto inner_ver = columns[i].version;
        auto inner_col =
            inner_rel->get_versioned_columns(inner_ver)[columns[i].idx];
        auto inner_size = inner_rel->get_versioned_size(inner_ver);
        inner_col.map_contains_if(
            thrust::make_permutation_iterator(outer_raw_begin,
                                              outer_indices->begin()),
            thrust::make_permutation_iterator(outer_raw_begin,
                                              outer_indices->end()),
            matched_outer_bitmap.begin(), matched_outer_bitmap_tmp.begin());
        matched_outer_bitmap.swap(matched_outer_bitmap_tmp);
    }
    // filter the matched indices
    auto matched_outer_end = thrust::remove_if(
        EXE_POLICY, outer_indices->begin(), outer_indices->end(),
        matched_outer_bitmap.begin(), thrust::logical_not<bool>());
    outer_indices->resize(matched_outer_end - outer_indices->begin());
}

std::string FreeJoinOperator::to_string() {
    std::string column_str = "{";
    for (auto &col : columns) {
        column_str += col.to_string() + ", ";
    }
    column_str += "}";
    std::string meta_str = "{";
    for (auto &meta : meta_vars) {
        meta_str += "\"" + meta + "\", ";
    }
    meta_str += "}";
    return "free_join_op(" + column_str + ", " + meta_str + ")";
}
} // namespace ram

} // namespace vflog

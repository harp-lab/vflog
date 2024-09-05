
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

void column_join_unfilter(ram::RelationalAlgebraMachine &ram, multi_hisa &inner,
                          RelationVersion inner_ver, size_t inner_idx,
                          multi_hisa &outer, RelationVersion outer_ver,
                          size_t outer_idx, std::string outer_meta_var,
                          std::shared_ptr<device_indices_t> matched_indices,
                          bool pop_outer) {
    auto &outer_tuple_indices = ram.cached_indices[outer_meta_var];
    // std::cout << ">>>>>>>>>>>>>>>>>>> " << outer_meta_var << std::endl;
    // std::cout << "outer size: " << outer_tuple_indices->size() << std::endl;
    if (outer_tuple_indices->size() == 0) {
        // clear all other in cache
        for (auto &meta : ram.cached_indices) {
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
    for (auto &meta : ram.cached_indices) {
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

    auto &inner_bitmap = ram.get_bitmap(inner.name);
    inner_bitmap.resize(inner.get_total_tuples(), false);
    auto cnt = 0;
    for (auto &meta : ram.cached_indices) {
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
                     inner_bits = inner_bitmap.RAW_PTR,
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
                        inner_bits[inner_sorted_idx[inner_idx]] = true;
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
                     inner_bits = inner_bitmap.RAW_PTR,
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
                        inner_bits[inner_sorted_idx[inner_idx]] = true;
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
        ram.cached_indices.erase(outer_meta_var);
    }
}

void column_join(ram::RelationalAlgebraMachine &ram, multi_hisa &inner,
                 RelationVersion inner_ver, size_t inner_idx, multi_hisa &outer,
                 RelationVersion outer_ver, size_t outer_idx,
                 std::string inner_reg, std::string outer_meta_var,
                 bool pop_outer) {
    auto &outer_tuple_indices = ram.cached_indices[outer_meta_var];
    auto &inner_tuple_indices = ram.register_map[inner_reg];
    // std::cout << "inner reg : " << inner_reg << " outer meta var " <<
    // outer_meta_var <<  std::endl;
    if (ram.rel_bitmap.find(inner.name) == ram.rel_bitmap.end()) {
        column_join_unfilter(ram, inner, inner_ver, inner_idx, outer, outer_ver,
                             outer_idx, outer_meta_var, inner_tuple_indices,
                             pop_outer);
        return;
    }
    device_bitmap_t &inner_bitmap = ram.get_bitmap(inner.name);

    if (outer_tuple_indices->size() == 0) {
        // clear all other in cache
        for (auto &meta : ram.cached_indices) {
            meta.second->resize(0);
        }
        inner_tuple_indices->resize(0);
        // inner_bitmap.resize(inner.get_total_tuples(), false);
        return;
    }

    auto &outer_column = outer.get_versioned_columns(outer_ver)[outer_idx];
    auto &inner_column = inner.get_versioned_columns(inner_ver)[inner_idx];
    if (inner.get_versioned_size(inner_ver) == 0) {
        return;
    }
    // calculate inner bitmaps
    // inner_bitmap.resize(inner.get_total_tuples());
    // thrust::fill(thrust::device, inner_bitmap.begin(), inner_bitmap.end(),
    //              false);
    // thrust::for_each(thrust::device, inner_tuple_indices->begin(),
    //                  inner_tuple_indices->end(),
    //                  [bits = inner_bitmap.RAW_PTR] LAMBDA_TAG(auto idx) {
    //                      bits[idx] = true;
    //                  });

    device_ranges_t matched_ranges(outer_tuple_indices->size());
    auto outer_raw_begin =
        outer.data[outer_idx].begin() + outer_column.raw_offset;
    inner_column.map_find(thrust::make_permutation_iterator(
                              outer_raw_begin, outer_tuple_indices->begin()),
                          thrust::make_permutation_iterator(
                              outer_raw_begin, outer_tuple_indices->end()),
                          matched_ranges.begin());
    // TODO: need optimization

    thrust::transform(
        EXE_POLICY, matched_ranges.begin(), matched_ranges.end(),
        matched_ranges.begin(),
        [inner_bits = inner_bitmap.RAW_PTR,
         inner_pos =
             inner_column.sorted_indices.RAW_PTR] LAMBDA_TAG(comp_range_t range)
            -> comp_range_t {
            if (range == UINT32_MAX) {
                return UINT32_MAX;
            }
            unsigned int start = static_cast<unsigned int>(range >> 32);
            unsigned int size = static_cast<unsigned int>(range);
            unsigned int new_size = 0;
            for (int i = 0; i < size; i++) {
                if (inner_bits[inner_pos[start + i]]) {
                    new_size++;
                }
            }
            if (new_size == 0) {
                return UINT32_MAX;
            }
            return (static_cast<comp_range_t>(start) << 32) +
                   static_cast<comp_range_t>(new_size);
        });

    // remove unmatched outer tuples
    for (auto &meta : ram.cached_indices) {
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
    inner_tuple_indices->resize(total_matched_size);
    auto cnt = 0;

    device_bitmap_t inner_bitmap_tmp(inner_bitmap.size(), false);

    for (auto &meta : ram.cached_indices) {
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
                    [res_inner = inner_tuple_indices->RAW_PTR,
                     res_outer = materialized_outer.RAW_PTR,
                     inner_sorted_idx = inner_column.sorted_indices.RAW_PTR,
                     outer_pos = outer_ts->RAW_PTR,
                     inner_offset = offset_vec.RAW_PTR,
                     size_offset = size_offset_tmp.RAW_PTR,
                     outer_size = outer_ts->size(),
                     inner_bits = inner_bitmap.RAW_PTR,
                     inner_bits_tmp = inner_bitmap_tmp.RAW_PTR,
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
                        // unsigned int inner_val = inner_sorted_idx[inner_idx];
                        // printf("inner_val: %u outer_val: %u\n", inner_val,
                        //        outer_val);
                        res_outer[res_pos] = outer_val;
                        unsigned int inner_idx_f =
                            res_pos - size_offset[found_outer_idx];
                        unsigned int i = 0;
                        unsigned int real_inner_idx =
                            inner_offset[found_outer_idx];
                        while (i < inner_idx_f) {
                            if (inner_bits[inner_sorted_idx[real_inner_idx]]) {
                                i++;
                            }
                            real_inner_idx++;
                        }
                        inner_bits_tmp[inner_sorted_idx[real_inner_idx]] = true;
                        res_inner[res_pos] = inner_sorted_idx[real_inner_idx];
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
                    [res_inner = inner_tuple_indices->RAW_PTR,
                     inner_sorted_idx = inner_column.sorted_indices.RAW_PTR,
                     inner_offset = offset_vec.RAW_PTR,
                     size_offset = size_offset_tmp.RAW_PTR,
                     outer_size = outer_ts->size(),
                     inner_bits = inner_bitmap.RAW_PTR,
                     inner_bits_tmp = inner_bitmap_tmp.RAW_PTR,
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
                        unsigned int inner_idx_f =
                            res_pos - size_offset[found_outer_idx];
                        unsigned int i = 0;
                        unsigned int real_inner_idx =
                            inner_offset[found_outer_idx];
                        while (i < inner_idx_f) {
                            if (inner_bits[inner_sorted_idx[real_inner_idx]]) {
                                i++;
                            }
                            real_inner_idx++;
                        }

                        inner_bits_tmp[inner_sorted_idx[real_inner_idx]] = true;
                        res_inner[res_pos] = inner_sorted_idx[real_inner_idx];
                    });
            }
            inner_bitmap.swap(inner_bitmap_tmp);
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
        ram.cached_indices.erase(outer_meta_var);
    }
}

} // namespace vflog


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
    vflog::VerticalColumn &inner_column, device_ranges_t &matched_ranges,
    host_buf_ref_t &cached_indices, std::string outer_meta_var,
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
                        if (found_outer_idx >= outer_size) {
                            found_outer_idx = outer_size - 1;
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

} // namespace vflog

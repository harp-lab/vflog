
#include "ram.cuh"
#include "utils.cuh"

#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>

namespace vflog {

void column_negate(multi_hisa &inner, RelationVersion inner_ver,
                   size_t inner_idx, multi_hisa &outer,
                   RelationVersion outer_ver, size_t outer_idx,
                   host_buf_ref_t &cached_indices, std::string meta_var,
                   bool pop_outer) {
    auto &outer_tuple_indices = cached_indices[meta_var];
    if (outer_tuple_indices->size() == 0) {
        // clear all other in cache
        for (auto &meta : cached_indices) {
            meta.second->resize(0);
        }
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
    inner_column.map_find(thrust::make_permutation_iterator(
                              outer_raw_begin, outer_tuple_indices->begin()),
                          thrust::make_permutation_iterator(
                              outer_raw_begin, outer_tuple_indices->end()),
                          matched_ranges.begin());

    // clear the unmatched tuples
    for (auto &meta : cached_indices) {
        auto &outer_ts = meta.second;
        thrust::transform(
            EXE_POLICY, matched_ranges.begin(), matched_ranges.end(),
            outer_ts->begin(), outer_ts->begin(),
            [] LAMBDA_TAG(auto &range, auto &outer_tuple_index) {
                return range != UINT32_MAX ? UINT32_MAX : outer_tuple_index;
            });

        auto new_outer_tuple_end = thrust::remove(EXE_POLICY, outer_ts->begin(),
                                                  outer_ts->end(), UINT32_MAX);
        outer_ts->resize(new_outer_tuple_end - outer_ts->begin());
    }

    if (pop_outer) {
        cached_indices.erase(meta_var);
    }
}

} // namespace vflog

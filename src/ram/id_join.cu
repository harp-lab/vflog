

#include "ram.cuh"
#include "utils.cuh"

#include <cstdio>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>

namespace vflog::ram {

void IdJoinOperator::execute(RelationalAlgebraMachine &ram) {
    // check which one is id
    if (inner.is_id() && outer.is_id()) {
        throw std::runtime_error("RAM error : Both inner and outer are id!");
    }
    auto &result_register_ptr = ram.register_map[result_register];
    auto outer_tuple_indices = ram.cached_indices[outer_meta_var];
    auto &outer_rel_ptr = ram.rels[outer.rel];

    if (inner.is_id()) {
        auto &outer_column =
            outer_rel_ptr->get_versioned_columns(outer.version)[outer.idx];
        auto outer_raw_begin =
            outer_rel_ptr->data[outer.idx].begin() + outer_column.raw_offset;
        // trivial case, just assign the outer ids to res
        result_register_ptr->resize(outer_tuple_indices->size());
        thrust::copy(EXE_POLICY,
                     thrust::make_permutation_iterator(
                         outer_raw_begin, outer_tuple_indices->begin()),
                     thrust::make_permutation_iterator(
                         outer_raw_begin, outer_tuple_indices->end()),
                     result_register_ptr->begin());
    } else {
        auto &inner_rel_ptr = ram.rels[inner.rel];
        auto &inner_column =
            inner_rel_ptr->get_versioned_columns(inner.version)[inner.idx];
        device_ranges_t matched_ranges(outer_tuple_indices->size());
        inner_column.map_find(outer_tuple_indices->begin(),
                              outer_tuple_indices->end(),
                              matched_ranges.begin());
        // clear the unmatched tuples
        for (auto &meta : ram.cached_indices) {
            auto &outer_ts = meta.second;
            thrust::transform(
                EXE_POLICY, matched_ranges.begin(), matched_ranges.end(),
                outer_ts->begin(), outer_ts->begin(),
                [] LAMBDA_TAG(auto &range, auto &outer_tuple_index) {
                    return range == UINT32_MAX ? UINT32_MAX : outer_tuple_index;
                });

            auto new_outer_tuple_end = thrust::remove(
                EXE_POLICY, outer_ts->begin(), outer_ts->end(), UINT32_MAX);
            outer_ts->resize(new_outer_tuple_end - outer_ts->begin());
        }

        // materialize the matched tuples
        materialize_matched_range(inner_column, matched_ranges,
                                  ram.cached_indices, outer_meta_var,
                                  result_register_ptr, false);
    }
}

std::string IdJoinOperator::to_string() {
    return "join_id_op(" + inner.to_string() + ", " + outer.to_string() +
           ", \"" + outer_meta_var + "\", \"" + result_register + "\")";
}

} // namespace vflog::ram


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
    materialize_matched_range(inner_column, matched_ranges, cached_indices,
                              outer_meta_var, matched_indices, pop_outer);
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

void JoinOperator::execute(RelationalAlgebraMachine &ram) {
    auto outer_rel_p = ram.rels[outer.rel];
    if (ram.overflow_rel_name == outer.rel && outer.version == NEWT) {
        outer_rel_p = ram.overflow_rel;
    }

    auto inner_rel_p = ram.rels[inner.rel];
    if (inner.is_frozen()) {
        inner_rel_p = ram.get_frozen(inner.rel, inner.frozen_idx);
        if (inner_rel_p == nullptr) {
            // frozen relation not generated yet
            // clear all other in cache
            for (auto &meta : ram.cached_indices) {
                meta.second->resize(0);
            }
            matched_indices->resize(0);
            return;
        }
    }

    auto result_reg_ptr = matched_indices;
    if (result_reg_ptr == nullptr) {
        result_reg_ptr = ram.register_map[result_register];
    }

    if (inner_meta_var == "") {
        column_join(*inner_rel_p, inner.version, inner.idx, *outer_rel_p,
                    outer.version, outer.idx, ram.cached_indices,
                    outer_meta_var, result_reg_ptr, pop_outer);
    } else {
        column_join(ram, *inner_rel_p, inner.version, inner.idx, *outer_rel_p,
                    outer.version, outer.idx, result_register, outer_meta_var,
                    pop_outer);
    }
}

std::string JoinOperator::to_string() {
    if (inner_meta_var == "") {
        return "join_op(" + inner.to_string() + ", " + outer.to_string() +
               ", \"" + result_register + "\", \"" + outer_meta_var + "\")";
    } else {
        return "join_op(" + inner.to_string() + ", " + outer.to_string() +
               ", \"" + inner_meta_var + "\", \"" + outer_meta_var + "\")";
    }
}

} // namespace ram

} // namespace vflog

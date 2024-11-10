
#include "ram.cuh"
#include "utils.cuh"

#include <cstddef>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <vector>

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

namespace ram {
void NegateOperator::execute(RelationalAlgebraMachine &ram) {
    auto inner_rel_p = ram.rels[inner.rel];
    if (inner.is_frozen()) {
        inner_rel_p = ram.get_frozen(inner.rel, inner.frozen_idx);
        if (inner_rel_p == nullptr) {
            return;
        }
    }
    if (ram.overflow_rel_name == outer.rel && outer.version == NEWT) {
        column_negate(*inner_rel_p, inner.version, inner.idx, *ram.overflow_rel,
                      outer.version, outer.idx, ram.cached_indices, meta_var,
                      pop_outer);
    } else {
        column_negate(*inner_rel_p, inner.version, inner.idx,
                      *ram.rels[outer.rel], outer.version, outer.idx,
                      ram.cached_indices, meta_var, pop_outer);
    }
}

std::string NegateOperator::to_string() {
    return "negate_op(" + inner.to_string() + ", " + outer.to_string() +
           ", \"" + meta_var + "\")";
}

void NegateMulti::execute(RelationalAlgebraMachine &ram) {
    auto inner_ptr = ram.get_rel(inner.name);
    if (inner_ptr->full_size == 0) {
        return;
    }

    auto negate_cache_size = ram.cached_indices[meta_vars[0]]->size();
    device_ranges_t matched_ranges(negate_cache_size);
    auto &diff_indices_default =
        ram.cached_indices[meta_vars[inner_ptr->default_index_column]];
    auto default_col_outer_info =
        outer_columns[inner_ptr->default_index_column];
    auto default_col_outer_rel = ram.get_rel(default_col_outer_info.rel);
    auto default_col_other_raw_begin = default_col_outer_rel->get_raw_data_ptrs(
        default_col_outer_info.version, default_col_outer_info.idx);

    if (diff_indices_default->size() == 0) {
        return;
    }
    auto &inner_default_col =
        inner_ptr->get_versioned_columns(FULL)[inner_ptr->default_index_column];
    inner_default_col.map_find(
        thrust::make_permutation_iterator(default_col_other_raw_begin,
                                          diff_indices_default->begin()),
        thrust::make_permutation_iterator(default_col_other_raw_begin,
                                          diff_indices_default->end()),
        matched_ranges.begin());

    DEVICE_VECTOR<internal_data_type *> all_col_fulls_ptrs(inner_ptr->arity);
    for (int i = 0; i < inner_ptr->arity; i++) {
        all_col_fulls_ptrs[i] = inner_ptr->data[i].RAW_PTR;
    }

    device_bitmap_t dup_other_flags(negate_cache_size, false);
    // TODO: macro to gen different arities version
    if (inner_ptr->arity == 2) {
        auto outer_start_iter = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::make_permutation_iterator(
                ram.get_rel(outer_columns[0].rel)
                    ->get_raw_data_ptrs(outer_columns[0].version,
                                        outer_columns[0].idx),
                ram.cached_indices[meta_vars[0]]->begin()),
            thrust::make_permutation_iterator(
                ram.get_rel(outer_columns[1].rel)
                    ->get_raw_data_ptrs(outer_columns[1].version,
                                        outer_columns[1].idx),
                ram.cached_indices[meta_vars[1]]->begin())));
        thrust::transform(
            EXE_POLICY,
            outer_start_iter, outer_start_iter + negate_cache_size,
            matched_ranges.begin(), dup_other_flags.begin(),
            [all_col_fulls_ptrs = all_col_fulls_ptrs.RAW_PTR,
             all_col_full_idx_ptrs =
                 inner_ptr->full_columns[inner_ptr->default_index_column]
                     .sorted_indices.RAW_PTR,
             default_index_column = inner_ptr->default_index_column,
             arity = inner_ptr->arity] LAMBDA_TAG(auto outer_tuple,
                                                  auto range) -> bool {
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
                    int mid = (left + right) / 2;
                    auto full_index =
                        all_col_full_idx_ptrs[full_pos_start + mid];
                    if (all_col_fulls_ptrs[0][full_index] ==
                            thrust::get<0>(outer_tuple) &&
                        all_col_fulls_ptrs[1][full_index] ==
                            thrust::get<1>(outer_tuple)) {
                        return true;
                    } else if (all_col_fulls_ptrs[0][full_index] <
                                   thrust::get<0>(outer_tuple) ||
                               (all_col_fulls_ptrs[0][full_index] ==
                                    thrust::get<0>(outer_tuple) &&
                                all_col_fulls_ptrs[1][full_index] <
                                    thrust::get<1>(outer_tuple))) {
                        left = mid + 1;
                    } else {
                        right = mid - 1;
                    }
                }
                return false;
            });
    } else {
        // not implemented
    }

    auto dup_size = thrust::count(EXE_POLICY, dup_other_flags.begin(),
                                  dup_other_flags.end(), true);
    if (dup_size != 0) {
        for (auto mv : meta_vars) {
            auto new_other_end = thrust::remove_if(
                EXE_POLICY, ram.cached_indices[mv]->begin(),
                ram.cached_indices[mv]->end(), dup_other_flags.begin(),
                thrust::identity<bool>());
            ram.cached_indices[mv]->resize(new_other_end -
                                           ram.cached_indices[mv]->begin());
        }
    }
}

std::string NegateMulti::to_string() {
    return "negate_multi(" + inner.to_string() + "... ";
}
} // namespace ram

} // namespace vflog


#include "ram.cuh"

#include "utils.cuh"
#include <cstdint>
#include <string>
#include <thrust/remove.h>
#include <thrust/transform.h>
#include <vector>

namespace vflog {

__device__ int
clustered_tuple_compare(int inner_indices_size,
                        internal_data_type **inner_clustered_data,
                        internal_data_type **outer_clustered_data,
                        unsigned int start_pos, unsigned int mid) {
    for (int i = 0; i < inner_indices_size; i++) {
        if (inner_clustered_data[i][start_pos + mid] <
            outer_clustered_data[i][start_pos]) {
            return -1;
            ;
        } else if (inner_clustered_data[i][start_pos + mid] >
                   outer_clustered_data[i][start_pos]) {
            return 1;
        }
    }
    return 0;
}

void multi_arities_join(ram::RelationalAlgebraMachine &ram,
                        std::vector<column_t> inner_columns,
                        std::vector<column_t> outer_columns,
                        std::string inner_reg,
                        std::vector<std::string> outer_meta_vars,
                        bool pop_outer) {
    auto inner_rel_name = inner_columns[0].rel;
    auto &inner = ram.rels[inner_rel_name];
    auto inner_ver = inner_columns[0].version;

    std::vector<int> inner_indices;
    for (auto &col : inner_columns) {
        inner_indices.push_back(col.idx);
    }
    // Get the indices of the inner and outer relations
    auto &outer_tuple_indices = ram.cached_indices[outer_meta_vars[0]];
    auto &inner_tuple_indices = ram.register_map[inner_reg];
    if (outer_tuple_indices->size() == 0) {
        // clear all other in cache
        for (auto &meta : ram.cached_indices) {
            meta.second->resize(0);
        }
        inner_tuple_indices->resize(0);
        return;
    }

    // gather the outer values
    int outer_prime_idx = outer_columns[0].idx;
    auto &outer_prime_column =
        ram.rels[outer_columns[0].rel]->get_versioned_columns(
            outer_columns[0].version)[outer_prime_idx];
    int inner_prime_idx = inner_indices[0];
    auto &inner_prime_column =
        inner->get_versioned_columns(inner_ver)[inner_prime_idx];
    if (inner->get_versioned_size(inner_ver) == 0) {
        return;
    }

    device_ranges_t matched_ranges(outer_tuple_indices->size());
    auto outer_prime_raw_begin =
        ram.rels[outer_columns[0].rel]->data[outer_prime_idx].begin() +
        outer_prime_column.raw_offset;
    inner_prime_column.map_find(
        thrust::make_permutation_iterator(outer_prime_raw_begin,
                                          outer_tuple_indices->begin()),
        thrust::make_permutation_iterator(outer_prime_raw_begin,
                                          outer_tuple_indices->end()),
        matched_ranges.begin());

    if (outer_columns.size() == 1) {
        // materialize the matched ranges
        materialize_matched_range(
            inner_prime_column, outer_prime_column, matched_ranges,
            ram.cached_indices, outer_meta_vars[0], inner_tuple_indices,
            pop_outer);
        return;
    }
    // check if all joined column match
    DEVICE_VECTOR<internal_data_type *> clustered_inner_col_ptrs(
        inner_indices.size());
    for (int i = 0; i < inner_indices.size(); i++) {
        clustered_inner_col_ptrs[i] = inner->data[inner_indices[i]].RAW_PTR +
                                      inner_prime_column.raw_offset;
    }
    DEVICE_VECTOR<internal_data_type *> clustered_outer_col_ptrs(
        outer_columns.size());
    for (int i = 0; i < outer_columns.size(); i++) {
        vflog::VerticalColumn &outer_column =
            ram.rels[outer_columns[i].rel]->get_versioned_columns(
                outer_columns[i].version)[outer_columns[i].idx];
        clustered_outer_col_ptrs[i] =
            ram.rels[outer_columns[i].rel]->data[outer_columns[i].idx].RAW_PTR +
            outer_column.raw_offset;
    }

    // filter the matched ranges
    thrust::transform(
        EXE_POLICY, matched_ranges.begin(), matched_ranges.end(),
        matched_ranges.begin(),
        [inner_clustered_data = clustered_inner_col_ptrs.RAW_PTR,
         outer_clustered_data = clustered_outer_col_ptrs.RAW_PTR,
         inner_indices_size =
             inner_indices.size()] LAMBDA_TAG(unsigned long range) {
            if (range == UINT32_MAX) {
                return range;
            }
            unsigned int start_pos = static_cast<unsigned int>(range >> 32);
            unsigned int size = static_cast<unsigned int>(range);
            // binary search to match
            int left = 0;
            int right = size - 1;
            while (left < right) {
                int mid = (left + right) / 2;
                int cmp_res = clustered_tuple_compare(
                    inner_indices_size, inner_clustered_data,
                    outer_clustered_data, start_pos, mid);
                if (cmp_res == 0) {
                    auto eq_pos_start = mid;
                    auto eq_pos_end = mid;
                    while (clustered_tuple_compare(
                               inner_indices_size, inner_clustered_data,
                               outer_clustered_data, start_pos,
                               eq_pos_end) == 0) {
                        eq_pos_end++;
                    }
                    return (static_cast<unsigned long>(start_pos + eq_pos_start)
                            << 32) |
                           (eq_pos_start - eq_pos_start);
                }
                if (cmp_res < 0) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            return static_cast<unsigned long>(UINT32_MAX);
        });

    // filter out the invalid ranges
    auto new_end = thrust::remove(EXE_POLICY, matched_ranges.begin(),
                                  matched_ranges.end(), UINT32_MAX);
    matched_ranges.resize(new_end - matched_ranges.begin());

    // materialize the matched ranges
    materialize_matched_range(
        inner_prime_column, outer_prime_column, matched_ranges,
        ram.cached_indices, outer_meta_vars[0], inner_tuple_indices, pop_outer);
}

} // namespace vflog

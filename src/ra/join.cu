
#include "ra.cuh"

#include <exception>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>

namespace fvlog {
void RelationalJoin::execute(RelationalEnvironment &env) {
    auto &rel_slices = env.slices;
    // sanity check before join
    // all rel_slice must have the same size
    int slice_tuple_size = -1;

    // create a map to store the relation and the column index pair
    HOST_VECTOR<std::pair<device_data_ptr, device_data_ptr>,
                hisa::device_data_t>
        join_result_history;

    // get the outer-most relation
    auto outer_relation = input_columns[0].relation;
    auto outer_version = input_columns[0].version;
    auto outer_idx = input_columns[0].column_idx;
    auto &outer_joined_column =
        outer_relation->get_column(outer_version, outer_idx);

    // create outer most relation result slice
    rel_slices[outer_relation->get_name()] =
        slice(outer_relation, outer_version);
    auto &outer_slice = rel_slices[outer_relation->get_name()];

    auto &outer_map = outer_joined_column.unique_v_map;
    hisa::device_data_t outer_values(outer_map->size());
    // init outer values to all unique values
    outer_map->retrieve_all(outer_values.data(),
                            thrust::make_discard_iterator());
    // outer_slice.bitmap.resize(outer_joined_column.size(), false);

    // hisa::device_bitmap_t matched_value_flags(outer_values.size(), false);
    hisa::device_bitmap_t matched_value_flags_tmp(outer_values.size(), false);
    // filter the outer most relation
    for (size_t i = 1; i < input_columns.size(); i++) {
        auto inner_relation = input_columns[i].relation;
        auto inner_version = input_columns[i].version;
        auto inner_joined_column = input_columns[i].column_idx;
        auto inner_size = inner_relation->get_size(inner_version);
        auto &inner_column =
            inner_relation->get_column(inner_version, inner_joined_column);
        auto &inner_map = inner_column.unique_v_map;
        inner_map->contains(outer_values.begin(), outer_values.end(),
                            matched_value_flags_tmp.data());
        // remove the outer values by stencil in matched_value_flags_tmp
        auto filtered_value_end = thrust::remove_if(
            DEFAULT_DEVICE_POLICY, outer_values.begin(), outer_values.end(),
            matched_value_flags_tmp.begin(), thrust::logical_not<bool>());
        outer_values.resize(filtered_value_end - outer_values.begin());
        matched_value_flags_tmp.resize(outer_values.size());
    }

    // update the environment slices
    hisa::device_ranges_t matched_ranges(outer_values.size());
    for (auto &input_col : input_columns) {
        auto &relation = input_col.relation;
        auto &version = input_col.version;
        auto &column_idx = input_col.column_idx;
        auto &column = relation->get_column(version, column_idx);
        auto &unique_map = column.unique_v_map;
        hisa::device_bitmap_t matched_flags(column.size(), false);
        unique_map->find(outer_values.begin(), outer_values.end(),
                         matched_ranges.begin());
        // TODO: this is inefficient, we can use a co-rank like
        // parallel algorithm to get the matched flags
        thrust::for_each(
            DEFAULT_DEVICE_POLICY, matched_ranges.begin(), matched_ranges.end(),
            [flags_raw = matched_flags.data().get(),
             column_size = column.size()] __device__(hisa::comp_range_t range) {
                uint32_t offset = (uint32_t)(range >> 32);
                uint32_t length = (uint32_t)(range & 0xFFFFFFFF);
                for (size_t i = 0; i < length; i++) {
                    flags_raw[offset + i] = true;
                }
            });
        // check if its already in the environment
        if (rel_slices.find(relation->get_name()) == rel_slices.end()) {
            rel_slices[relation->get_name()] = slice(relation, version);
            rel_slices[relation->get_name()].move_indices(matched_flags);
        } else {
            auto &slice = rel_slices[relation->get_name()];
            // and with original bitmap
            thrust::transform(DEFAULT_DEVICE_POLICY, slice.bitmap.begin(),
                              slice.bitmap.end(), matched_flags.begin(),
                              slice.bitmap.begin(),
                              thrust::logical_and<bool>());
        }
    }
}

} // namespace fvlog


#pragma once

#include "hisa.cuh"
#include "utils.cuh"
#include <string>

namespace vflog {

void column_join(multi_hisa &inner, RelationVersion inner_ver, size_t inner_idx,
                 multi_hisa &outer, RelationVersion outer_ver, size_t outer_idx,
                 host_buf_ref_t &cached_indices, std::string meta_var,
                 std::shared_ptr<device_indices_t> matched_indices,
                 bool pop_outer = false);

namespace ram {
struct RelationalAlgebraMachine;
}

void column_join(ram::RelationalAlgebraMachine &ram, multi_hisa &inner,
                 RelationVersion inner_ver, size_t inner_idx, multi_hisa &outer,
                 RelationVersion outer_ver, size_t outer_idx,
                 std::string inner_meta_var, std::string outer_meta_var,
                 bool pop_outer);

void column_negate(multi_hisa &inner, RelationVersion inner_ver,
                   size_t inner_idx, multi_hisa &outer,
                   RelationVersion outer_ver, size_t outer_idx,
                   host_buf_ref_t &cached_indices, std::string meta_var,
                   bool pop_outer = false);

void column_copy(multi_hisa &src, RelationVersion src_ver, size_t src_idx,
                 multi_hisa &dst, RelationVersion dst_ver, size_t dst_idx,
                 std::shared_ptr<device_indices_t> &indices);

void column_copy_all(multi_hisa &src, RelationVersion src_version,
                     size_t src_idx, multi_hisa &dst,
                     RelationVersion dst_version, size_t dst_idx,
                     bool append = false);
void column_copy_indices(multi_hisa &src, RelationVersion src_version,
                         size_t src_idx, multi_hisa &dst,
                         RelationVersion dst_version, size_t dst_idx,
                         std::shared_ptr<device_indices_t> &indices,
                         bool append = false);

void multi_arities_join(ram::RelationalAlgebraMachine &ram,
                        std::vector<column_t> inner_columns,
                        std::vector<column_t> outer_columns,
                        std::string inner_reg,
                        std::vector<std::string> outer_meta_vars,
                        bool pop_outer);

void relational_copy(multi_hisa &src, RelationVersion src_version,
                     multi_hisa &dst, RelationVersion dst_version,
                     host_buf_ref_t &cached_indices,
                     std::vector<std::string> input_meta_vars,
                     std::vector<std::string> output_meta_vars);

void materialize_matched_range(
    vflog::VerticalColumn &inner_column, device_ranges_t &matched_ranges,
    host_buf_ref_t &cached_indices, std::string outer_meta_var,
    std::shared_ptr<device_indices_t> matched_indices, bool pop_outer = false);

} // namespace vflog

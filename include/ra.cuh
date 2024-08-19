

#include "hisa.cuh"
#include <algorithm>
#include <iostream>
#include <vector>

namespace vflog {

enum class RAOperatorType {
    JOIN,
    INDEX,
    PROJECT,
    FILTER,
    NEGATE,
    AGGREGATE,
    ARITHMETIC,
};

struct RelationalAlgebraOperator {
    using ID_t = uint32_t;
    ID_t id;
    RAOperatorType type;
    bool debug_flag = false;

    virtual void execute() = 0;
};

/**
 * @brief Relational join operator for free join
 * Join operator takes k input relation column as input
 * It will be applied to the re_slices
 * After execution, the result will be updated in the rel_slices
 */
struct JoinOperator : public RelationalAlgebraOperator {

    void execute() override {}
};

void column_join(multi_hisa &inner, RelationVersion inner_ver, size_t inner_idx,
                 multi_hisa &outer, RelationVersion outer_ver, size_t outer_idx,
                 host_buf_ref_t &cached_indices, std::string meta_var,
                 std::shared_ptr<device_indices_t> matched_indices,
                 bool pop_outer = false);

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

void relational_copy(multi_hisa &src, RelationVersion src_version,
                     multi_hisa &dst, RelationVersion dst_version,
                     host_buf_ref_t &cached_indices,
                     std::vector<std::string> input_meta_vars,
                     std::vector<std::string> output_meta_vars);

} // namespace vflog

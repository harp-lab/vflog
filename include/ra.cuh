

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

void column_join(multi_hisa &inner, RelationVersion inner_version,
                 size_t inner_column_idx, multi_hisa &outer,
                 RelationVersion outer_version, size_t outer_column_idx,
                 device_indices_t &outer_tuple_indices,
                 device_indices_t &matched_indices,
                 DEVICE_VECTOR<bool> &unmatched);
void column_join(multi_hisa &inner, RelationVersion inner_ver, size_t inner_idx,
                 multi_hisa &outer, RelationVersion outer_ver, size_t outer_idx,
                 host_buf_ref_t &cached_indices, std::string meta_var,
                 std::shared_ptr<device_indices_t> matched_indices,
                 bool pop_outer = false);

void column_copy(multi_hisa &src, RelationVersion src_version, size_t src_idx,
                 multi_hisa &dst, RelationVersion dst_version, size_t dst_idx,
                 device_indices_t &indices);
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
                         bool append = false);

} // namespace vflog

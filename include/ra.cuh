
#pragma once

#include "hisa.cuh"
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <thrust/sequence.h>

namespace vflog {

struct RelationalAlgebraMachine;

enum class RAMInstructionType {
    JOIN,
    INDEX,
    PROJECT,
    FILTER,
    NEGATE,
    AGGREGATE,
    ARITHMETIC,
    DIFF,
    CUSTOM,
    CACHE_UPDATE,
    CACHE_INIT,
    CACHE_CLEAR,
    ALLOC_NEW,
    RECORD_NEW_SIZE,
    PERSISTENT,
    PRINT_SIZE,
    FIXPOINT
};

struct RAMInstruction {
    using ID_t = uint32_t;
    ID_t id;
    RAMInstructionType type;
    bool debug_flag = false;

    virtual void execute(RelationalAlgebraMachine &ram) = 0;
};

using rel_ptr = std::shared_ptr<multi_hisa>;

struct column_t {
    std::string rel;
    size_t idx;
    RelationVersion version;

    int frozen_idx = -1;

    column_t(std::string rel, size_t idx, RelationVersion version)
        : rel(rel), idx(idx), version(version) {}
    column_t(std::string rel, size_t idx, RelationVersion version, int frozen_idx)
        : rel(rel), idx(idx), version(version), frozen_idx(frozen_idx) {}

    bool is_frozen() { return frozen_idx != -1; }
};

struct rel_t {
    std::string name;

    rel_t(std::string name) : name(name) {}
};

struct PrintSize : RAMInstruction {
    rel_t rel;

    PrintSize(rel_t rel) : rel(rel) { type = RAMInstructionType::PRINT_SIZE; }

    void execute(RelationalAlgebraMachine &ram) override;
};

struct CacheInit : RAMInstruction {
    std::string meta_var;
    rel_t rel;
    RelationVersion version;

    CacheInit(std::string meta_var, rel_t rel, RelationVersion version)
        : meta_var(meta_var), rel(rel), version(version) {
        type = RAMInstructionType::CACHE_INIT;
    }

    void execute(RelationalAlgebraMachine &ram) override;
};

struct CacheUpdate : RAMInstruction {
    std::string meta_var;
    std::string existing_register;
    std::shared_ptr<device_indices_t> indices = nullptr;

    CacheUpdate(std::string meta_var, std::shared_ptr<device_indices_t> indices)
        : meta_var(meta_var), indices(indices) {
        type = RAMInstructionType::CACHE_UPDATE;
    }

    CacheUpdate(std::string meta_var, std::string existing_register)
        : meta_var(meta_var), existing_register(existing_register) {
        type = RAMInstructionType::CACHE_UPDATE;
    }

    void execute(RelationalAlgebraMachine &ram) override;
};

struct CacheClear : RAMInstruction {

    CacheClear() { type = RAMInstructionType::CACHE_CLEAR; }

    void execute(RelationalAlgebraMachine &ram) override;
};

struct PrepareMaterialization : RAMInstruction {
    rel_t rel;
    std::string meta_var;

    PrepareMaterialization(rel_t rel, std::string meta_var)
        : rel(rel), meta_var(meta_var) {
        type = RAMInstructionType::ALLOC_NEW;
    }

    void execute(RelationalAlgebraMachine &ram) override;
};

struct EndMaterialization : RAMInstruction {
    rel_t rel;
    std::string meta_var;

    EndMaterialization(rel_t rel, std::string meta_var)
        : rel(rel), meta_var(meta_var) {
        type = RAMInstructionType::RECORD_NEW_SIZE;
    }

    void execute(RelationalAlgebraMachine &ram) override;
};

struct Persistent : RAMInstruction {
    rel_t rel;
    device_indices_t _indices_default;

    Persistent(rel_t rel) : rel(rel) { type = RAMInstructionType::PERSISTENT; }

    void execute(RelationalAlgebraMachine &ram) override;
};

struct JoinOperator : public RAMInstruction {
    column_t inner;
    column_t outer;
    std::string outer_meta_var; // the meta
    std::string result_register = "";

    std::shared_ptr<device_indices_t> matched_indices = nullptr;
    bool pop_outer = false;

    JoinOperator(column_t inner, column_t outer, std::string outer_meta_var,
                 std::shared_ptr<device_indices_t> matched_indices,
                 bool pop_outer = false)
        : inner(inner), outer(outer), outer_meta_var(outer_meta_var),
          matched_indices(matched_indices), pop_outer(pop_outer) {
        type = RAMInstructionType::JOIN;
    }

    JoinOperator(column_t inner, column_t outer, std::string outer_meta_var,
                 std::string result_register, bool pop_outer = false)
        : inner(inner), outer(outer), outer_meta_var(outer_meta_var),
          result_register(result_register), pop_outer(pop_outer) {
        type = RAMInstructionType::JOIN;
    }

    void execute(RelationalAlgebraMachine &ram) override;
};

struct NegateOperator : public RAMInstruction {
    column_t inner;
    column_t outer;
    std::string meta_var;
    bool pop_outer = false;

    NegateOperator(column_t inner, column_t outer, std::string meta_var,
                   bool pop_outer = false)
        : inner(inner), outer(outer), meta_var(meta_var), pop_outer(pop_outer) {
        type = RAMInstructionType::NEGATE;
    }

    void execute(RelationalAlgebraMachine &ram) override;
};

struct ProjectOperator : public RAMInstruction {
    column_t src;
    column_t dst;
    bool id_flag = false;
    std::string meta_var;

    ProjectOperator(column_t src, column_t dst, std::string meta_var,
                    bool id_flag = false)
        : src(src), dst(dst), meta_var(meta_var), id_flag(id_flag) {
        type = RAMInstructionType::PROJECT;
    }

    void execute(RelationalAlgebraMachine &ram) override;
};

struct DiffOperator : public RAMInstruction {
    rel_t src;
    rel_t dst;
    std::shared_ptr<device_indices_t> diff_indices;

    DiffOperator(rel_t src, rel_t dst,
                 std::shared_ptr<device_indices_t> diff_indices)
        : src(src), dst(dst), diff_indices(diff_indices) {
        type = RAMInstructionType::DIFF;
    }

    void execute(RelationalAlgebraMachine &ram) override;
};

struct CustomizedOperator : public RAMInstruction {
    std::function<void(RelationalAlgebraMachine &)> func;

    CustomizedOperator(std::function<void(RelationalAlgebraMachine &)> func)
        : func(func) {
        type = RAMInstructionType::CUSTOM;
    }

    void execute(RelationalAlgebraMachine &ram) override { func(ram); }
};

struct FixpointOperator : public RAMInstruction {
    std::vector<std::shared_ptr<RAMInstruction>> operators;
    std::vector<rel_t> rels;
    device_indices_t _indices_default;
    int max_iter = -1;
    std::map<int, double> stats;

    FixpointOperator(std::vector<std::shared_ptr<RAMInstruction>> operators,
                     std::vector<rel_t> rels, int max_iter = -1)
        : operators(operators), rels(rels) {
        type = RAMInstructionType::FIXPOINT;
    }

    void execute(RelationalAlgebraMachine &ram) override;
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

// >>>>>>>>>>>>>>>>>>>>> Exposed RAM instruction API >>>>>>>>>>>>>>>>>>>>>>>>>>>

// print the size of a relation
inline std::shared_ptr<PrintSize> print_size(rel_t rel) {
    return std::make_shared<PrintSize>(rel);
}

// join to column, and materialize the what's in cache
inline std::shared_ptr<JoinOperator>
join_op(column_t inner, column_t outer, std::string outer_meta_var,
        std::shared_ptr<device_indices_t> matched_indices,
        bool pop_outer = false) {
    return std::make_shared<JoinOperator>(inner, outer, outer_meta_var,
                                          matched_indices, pop_outer);
}
inline std::shared_ptr<JoinOperator> join_op(column_t inner, column_t outer,
                                             std::string outer_meta_var,
                                             std::string result_register,
                                             bool pop_outer = false) {
    return std::make_shared<JoinOperator>(inner, outer, outer_meta_var,
                                          result_register, pop_outer);
}

inline std::shared_ptr<NegateOperator> negate_op(column_t inner, column_t outer,
                                                 std::string meta_var,
                                                 bool pop_outer = false) {
    return std::make_shared<NegateOperator>(inner, outer, meta_var, pop_outer);
}

// project src to dst, based on the indices cached in meta_var
inline std::shared_ptr<ProjectOperator> project_op(column_t src, column_t dst,
                                                   std::string meta_var,
                                                   bool id_flag = false) {
    return std::make_shared<ProjectOperator>(src, dst, meta_var, id_flag);
}

// remove dst tuple which is in src
inline std::shared_ptr<DiffOperator>
diff_op(rel_t src, rel_t dst, std::shared_ptr<device_indices_t> diff_indices,
        bool debug_flag = false) {
    auto op = std::make_shared<DiffOperator>(src, dst, diff_indices);
    op->debug_flag = debug_flag;
    return op;
}

// let user nest C++ code in the query
inline std::shared_ptr<CustomizedOperator>
custom_op(std::function<void(RelationalAlgebraMachine &)> func,
          bool debug_flag = false) {
    auto op = std::make_shared<CustomizedOperator>(func);
    op->debug_flag = debug_flag;
    return op;
}

// init a cached variable to id of all tuples in a versioned relation
inline std::shared_ptr<CacheInit> cache_init(std::string meta_var, rel_t rel,
                                             RelationVersion version,
                                             bool debug_flag = false) {
    auto op = std::make_shared<CacheInit>(meta_var, rel, version);
    op->debug_flag = debug_flag;
    return op;
}

// update a cached variable to a set of indices
inline std::shared_ptr<CacheUpdate>
cache_update(std::string meta_var, std::shared_ptr<device_indices_t> indices,
             bool debug_flag = false) {
    auto op = std::make_shared<CacheUpdate>(meta_var, indices);
    op->debug_flag = debug_flag;
    return op;
}
inline std::shared_ptr<CacheUpdate> cache_update(std::string meta_var,
                                                 std::string existing_register,
                                                 bool debug_flag = false) {
    auto op = std::make_shared<CacheUpdate>(meta_var, existing_register);
    op->debug_flag = debug_flag;
    return op;
}

// clear all cached idxs
inline std::shared_ptr<CacheClear> cache_clear(bool debug_flag = false) {
    auto op = std::make_shared<CacheClear>();
    op->debug_flag = debug_flag;
    return op;
}

// prepare a versioned relation for materialization
// if a relation overflow, it will be split into multiple relations
inline std::shared_ptr<PrepareMaterialization>
prepare_materialization(rel_t rel, std::string meta_var,
                        bool debug_flag = false) {
    auto op = std::make_shared<PrepareMaterialization>(rel, meta_var);
    op->debug_flag = debug_flag;
    return op;
}

// record the size of the materialized relation
inline std::shared_ptr<EndMaterialization>
end_materialization(rel_t rel, std::string meta_var, bool debug_flag = false) {
    auto op = std::make_shared<EndMaterialization>(rel, meta_var);
    op->debug_flag = debug_flag;
    return op;
}

// persist the materialized relation
inline std::shared_ptr<Persistent> persistent(rel_t rel,
                                              bool debug_flag = false) {
    auto op = std::make_shared<Persistent>(rel);
    op->debug_flag = debug_flag;
    return op;
}

inline std::shared_ptr<FixpointOperator>
fixpoint_op(std::vector<std::shared_ptr<RAMInstruction>> operators,
            std::vector<rel_t> rels, bool debug_flag = false) {
    auto op = std::make_shared<FixpointOperator>(operators, rels);
    op->debug_flag = debug_flag;
    return op;
}

} // namespace vflog

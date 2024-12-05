
#pragma once

#include "ra.cuh"
#include "utils.cuh"
#include <map>
#include <memory>
#include <string>
#include <sys/types.h>
#include <vector>

#include <thrust/sequence.h>

namespace vflog {

namespace ram {

struct RelationalAlgebraMachine;

enum class RAMInstructionType {
    MULIT_ARITY_JOIN,
    JOIN,
    ID_JOIN,
    FREE_JOIN,
    INDEX,
    PROJECT,
    PROJECT_ID,
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
    FIXPOINT,
    STATIC_BATCH,
    DECLARE,
    EXT_REGISTER,
    FILL,
    SET_COLUMN_STRATEGY,
    DEFAULT_COLUMN,
    LOAD_FILE,
    FACT,
    HASH_DIGEST,
    CARTEISAN,
};

struct RAMInstruction {
    using ID_t = uint32_t;
    ID_t id;
    RAMInstructionType type;
    bool debug_flag = false;

    virtual void execute(RelationalAlgebraMachine &ram) = 0;

    virtual std::string to_string() = 0;
};

// using RAMProgram = std::vector<std::shared_ptr<RAMInstruction>>;

struct RAMProgram {
    std::vector<std::shared_ptr<RAMInstruction>> instructions;

    RAMProgram() = default;

    RAMProgram(std::vector<std::shared_ptr<RAMInstruction>> instructions)
        : instructions(instructions) {}

    void add_instruction(std::shared_ptr<RAMInstruction> instruction) {
        instructions.push_back(instruction);
    }

    void add_instructions(
        std::vector<std::shared_ptr<RAMInstruction>> instructions) {
        for (auto &instruction : instructions) {
            add_instruction(instruction);
        }
    }

    void execute(RelationalAlgebraMachine &ram) {
        for (auto &instruction : instructions) {
            instruction->execute(ram);
        }
    }

    std::string to_string() {
        std::string str = "{\n";
        for (auto &instruction : instructions) {
            str += instruction->to_string() + ",\n";
        }
        str += "}";
        return str;
    }
};

struct Declaration : RAMInstruction {
    std::string name;
    int arity;
    char *data_path;

    Declaration(std::string name, int arity, char *data_path = nullptr)
        : name(name), arity(arity), data_path(data_path) {
        type = RAMInstructionType::DECLARE;
    }

    void execute(RelationalAlgebraMachine &ram) override;
    std::string to_string() override;
};

struct ExtendRegister : RAMInstruction {
    std::string name;

    ExtendRegister(std::string name) : name(name) {
        type = RAMInstructionType::EXT_REGISTER;
    }

    void execute(RelationalAlgebraMachine &ram) override;

    std::string to_string() override {
        return "extend_register(\"" + name + "\")";
    }
};

using rel_ptr = std::shared_ptr<multi_hisa>;

struct rel_t {
    std::string name;

    rel_t(std::string name) : name(name) {}

    std::string to_string() {
        std::string str = "rel_t(\"" + name + "\")";
        return str;
    }
};

struct PrintSize : RAMInstruction {
    rel_t rel;

    PrintSize(rel_t rel) : rel(rel) { type = RAMInstructionType::PRINT_SIZE; }

    void execute(RelationalAlgebraMachine &ram) override;
    std::string to_string() override;
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
    std::string to_string() override;
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
    std::string to_string() override;
};

struct CacheClear : RAMInstruction {

    CacheClear() { type = RAMInstructionType::CACHE_CLEAR; }

    void execute(RelationalAlgebraMachine &ram) override;
    std::string to_string() override { return "cache_clear()"; }
};

struct PrepareMaterialization : RAMInstruction {
    rel_t rel;
    std::string meta_var;

    PrepareMaterialization(rel_t rel, std::string meta_var)
        : rel(rel), meta_var(meta_var) {
        type = RAMInstructionType::ALLOC_NEW;
    }

    void execute(RelationalAlgebraMachine &ram) override;
    std::string to_string() override;
};

struct EndMaterialization : RAMInstruction {
    rel_t rel;
    std::string meta_var;

    EndMaterialization(rel_t rel, std::string meta_var)
        : rel(rel), meta_var(meta_var) {
        type = RAMInstructionType::RECORD_NEW_SIZE;
    }

    void execute(RelationalAlgebraMachine &ram) override;
    std::string to_string() override;
};

struct Persistent : RAMInstruction {
    rel_t rel;
    device_indices_t _indices_default;

    Persistent(rel_t rel) : rel(rel) { type = RAMInstructionType::PERSISTENT; }

    void execute(RelationalAlgebraMachine &ram) override;

    std::string to_string() override {
        return "persistent(" + rel.to_string() + ")";
    }
};

struct JoinOperator : public RAMInstruction {
    column_t inner;
    column_t outer;
    std::string outer_meta_var; // the meta
    std::string inner_meta_var = "";
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

    JoinOperator(column_t inner, column_t outer, std::string inner_meta_var,
                 std::string outer_meta_var, std::string result_register,
                 bool pop_outer = false)
        : inner(inner), outer(outer), inner_meta_var(inner_meta_var),
          outer_meta_var(outer_meta_var), result_register(result_register),
          pop_outer(pop_outer) {
        type = RAMInstructionType::JOIN;
    }

    void execute(RelationalAlgebraMachine &ram) override;
    std::string to_string() override;
};

struct CartesianOperator : public RAMInstruction {
    rel_t rel1;
    RelationVersion version1;
    rel_t rel2;
    RelationVersion version2;
    std::string result_register1;
    std::string result_register2;

    CartesianOperator(rel_t rel1, RelationVersion version1, rel_t rel2,
                      RelationVersion version2, std::string result_register1,
                      std::string result_register2)
        : rel1(rel1), version1(version1), rel2(rel2), version2(version2),
          result_register1(result_register1),
          result_register2(result_register2) {
        type = RAMInstructionType::CARTEISAN;
    }

    void execute(RelationalAlgebraMachine &ram) override;
    std::string to_string() override;
};

struct IdJoinOperator : public RAMInstruction {
    column_t inner;
    column_t outer;
    std::string outer_meta_var;
    std::string inner_meta_var;
    std::string result_register;

    IdJoinOperator(column_t inner, column_t outer, std::string outer_meta_var,
                   std::string result_register)
        : inner(inner), outer(outer), outer_meta_var(outer_meta_var),
          result_register(result_register) {
        type = RAMInstructionType::ID_JOIN;
    }

    void execute(RelationalAlgebraMachine &ram) override;
    std::string to_string() override;
};

struct MultiArityJoinOperator : public RAMInstruction {
    std::vector<column_t> inner_columns;
    std::vector<column_t> outer_columns;
    std::string inner_reg;
    std::vector<std::string> outer_meta_vars;
    bool pop_outer = false;

    MultiArityJoinOperator(std::vector<column_t> inner_columns,
                           std::vector<column_t> outer_columns,
                           std::string inner_reg,
                           std::vector<std::string> outer_meta_vars,
                           bool pop_outer = false)
        : inner_columns(inner_columns), outer_columns(outer_columns),
          inner_reg(inner_reg), outer_meta_vars(outer_meta_vars),
          pop_outer(pop_outer) {
        type = RAMInstructionType::MULIT_ARITY_JOIN;
    }

    void execute(RelationalAlgebraMachine &ram) override;

    std::string to_string() override;
};

struct FreeJoinOperator : public RAMInstruction {
    std::vector<column_t> columns;
    std::vector<std::string> meta_vars;

    FreeJoinOperator(std::vector<column_t> columns,
                     std::vector<std::string> meta_vars)
        : columns(columns), meta_vars(meta_vars) {
        type = RAMInstructionType::FREE_JOIN;
    }

    void execute(RelationalAlgebraMachine &ram) override;
    std::string to_string() override;
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
    std::string to_string() override;
};

struct NegateMulti : public RAMInstruction {
    rel_t inner;
    std::vector<column_t> outer_columns;
    std::vector<std::string> meta_vars;
    // bool pop_outer = false;

    NegateMulti(rel_t inner, std::vector<column_t> outer_columns,
                std::vector<std::string> meta_vars)
        : inner(inner), outer_columns(outer_columns), meta_vars(meta_vars) {
        type = RAMInstructionType::NEGATE;
    }

    void execute(RelationalAlgebraMachine &ram) override;
    std::string to_string() override;
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
    std::string to_string() override;
};

struct ProjectIdOperator : public RAMInstruction {
    column_t dst;
    std::string meta_var;

    ProjectIdOperator(column_t dst, std::string meta_var)
        : dst(dst), meta_var(meta_var) {
        type = RAMInstructionType::PROJECT_ID;
    }

    void execute(RelationalAlgebraMachine &ram) override;
    std::string to_string() override;
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
    std::string to_string() override {
        return "diff_op(" + src.to_string() + ", " + dst.to_string() + ")";
    }
};

struct CustomizedOperator : public RAMInstruction {
    std::function<void(RelationalAlgebraMachine &)> func;

    CustomizedOperator(std::function<void(RelationalAlgebraMachine &)> func)
        : func(func) {
        type = RAMInstructionType::CUSTOM;
    }

    void execute(RelationalAlgebraMachine &ram) override { func(ram); }
    std::string to_string() override { return "custom_op()"; }
};

struct FixpointOperator : public RAMInstruction {
    std::vector<std::shared_ptr<RAMInstruction>> operators;
    std::vector<rel_t> rels;
    device_indices_t _indices_default;
    int max_iter = -1;
    std::map<int, double> stats;
    int iter = 0;

    FixpointOperator(std::vector<std::shared_ptr<RAMInstruction>> operators,
                     std::vector<rel_t> rels, int max_iter = -1)
        : operators(operators), rels(rels) {
        type = RAMInstructionType::FIXPOINT;
    }

    void execute(RelationalAlgebraMachine &ram) override;
    std::string to_string() override;

    bool step(RelationalAlgebraMachine &ram);
};

struct StaticBatch : public RAMInstruction {
    std::vector<std::shared_ptr<RAMInstruction>> operators;
    std::map<int, double> stats;

    StaticBatch(std::vector<std::shared_ptr<RAMInstruction>> operators)
        : operators(operators) {
        type = RAMInstructionType::STATIC_BATCH;
    }

    void execute(RelationalAlgebraMachine &ram) override;
    std::string to_string() override;
};

struct FillOperator : public RAMInstruction {
    column_t dst;
    uint32_t value;

    FillOperator(column_t dst, uint32_t value) : dst(dst), value(value) {
        type = RAMInstructionType::FILL;
    }

    void execute(RelationalAlgebraMachine &ram) override;
    std::string to_string() override;
};

struct SetColumnStrategy : RAMInstruction {
    column_t column;
    IndexStrategy strategy;

    SetColumnStrategy(column_t column, IndexStrategy strategy)
        : column(column), strategy(strategy) {
        type = RAMInstructionType::SET_COLUMN_STRATEGY;
    }

    void execute(RelationalAlgebraMachine &ram) override;
    std::string to_string() override;
};

struct Index : RAMInstruction {
    rel_t rel;
    std::vector<int> columns;
    bool build_immediately = false;

    Index(rel_t rel, std::vector<int> columns, bool build_immediately)
        : rel(rel), columns(columns), build_immediately(build_immediately) {
        type = RAMInstructionType::INDEX;
    }

    void execute(RelationalAlgebraMachine &ram) override;
    std::string to_string() override;
};

struct DefaultColumn : RAMInstruction {
    rel_t rel;
    size_t column_idx;

    DefaultColumn(rel_t rel, size_t column_idx)
        : rel(rel), column_idx(column_idx) {
        type = RAMInstructionType::DEFAULT_COLUMN;
    }

    void execute(RelationalAlgebraMachine &ram) override;
    std::string to_string() override;
};

struct LoadFile : RAMInstruction {
    rel_t rel;
    const char *file_path;
    int arity;

    LoadFile(rel_t rel, const char *file_path, int arity)
        : rel(rel), file_path(file_path), arity(arity) {
        type = RAMInstructionType::LOAD_FILE;
    }

    void execute(RelationalAlgebraMachine &ram) override;
    std::string to_string() override;
};

struct Fact : RAMInstruction {
    rel_t rel;
    std::vector<std::vector<uint32_t>> data;

    Fact(rel_t rel, std::vector<std::vector<uint32_t>> data)
        : rel(rel), data(data) {
        type = RAMInstructionType::FACT;
    }

    void execute(RelationalAlgebraMachine &ram) override;
    std::string to_string() override;
};

struct HashDigest : RAMInstruction {
    std::vector<std::string> meta_vars;
    std::vector<column_t> columns;
    std::string result_register;

    HashDigest(std::vector<std::string> meta_vars,
               std::vector<column_t> columns, std::string result_register)
        : meta_vars(meta_vars), columns(columns),
          result_register(result_register) {
        type = RAMInstructionType::HASH_DIGEST;
    }

    void execute(RelationalAlgebraMachine &ram) override;
    std::string to_string() override;
};

struct ProjectHostOperator : RAMInstruction {
    uint32_t **host_src;
    size_t src_size;
    rel_t dst;

    ProjectHostOperator(uint32_t **host_src, size_t src_size, rel_t dst)
        : host_src(host_src), src_size(src_size), dst(dst) {
        type = RAMInstructionType::PROJECT;
    }

    void execute(RelationalAlgebraMachine &ram) override;
    std::string to_string() override;
};

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
inline std::shared_ptr<JoinOperator> join_op2(column_t inner, column_t outer,
                                              std::string inner_meta_var,
                                              std::string outer_meta_var,
                                              std::string result_register,
                                              bool pop_outer = false) {
    return std::make_shared<JoinOperator>(inner, outer, inner_meta_var,
                                          outer_meta_var, result_register,
                                          pop_outer);
}

inline std::shared_ptr<IdJoinOperator> join_id_op(column_t inner,
                                                  column_t outer,
                                                  std::string outer_meta_var,
                                                  std::string result_register) {
    // return std::make_shared<IdJoinOperator>(inner, outer, outer_meta_var,
    // result_register);
    // TODO: fix this
    return nullptr;
}

inline std::shared_ptr<MultiArityJoinOperator>
multi_arity_join_op(std::vector<column_t> inner_columns,
                    std::vector<column_t> outer_columns, std::string inner_reg,
                    std::vector<std::string> outer_meta_vars,
                    bool pop_outer = false) {
    return std::make_shared<MultiArityJoinOperator>(
        inner_columns, outer_columns, inner_reg, outer_meta_vars, pop_outer);
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

// project the id of src to dst, based on the indices cached in meta_var
inline std::shared_ptr<ProjectIdOperator> project_id_op(column_t dst,
                                                        std::string meta_var) {
    return std::make_shared<ProjectIdOperator>(dst, meta_var);
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

inline std::shared_ptr<Declaration> declare(std::string name, int arity,
                                            char *data_path = nullptr,
                                            bool debug_flag = false) {
    auto op = std::make_shared<Declaration>(name, arity, data_path);
    op->debug_flag = debug_flag;
    return op;
}

inline std::shared_ptr<ExtendRegister>
extend_register(std::string name, bool debug_flag = false) {
    auto op = std::make_shared<ExtendRegister>(name);
    op->debug_flag = debug_flag;
    return op;
}

inline std::shared_ptr<FillOperator> fill_op(column_t dst, uint32_t value,
                                             bool debug_flag = false) {
    auto op = std::make_shared<FillOperator>(dst, value);
    op->debug_flag = debug_flag;
    return op;
}

inline std::shared_ptr<StaticBatch>
static_batch(std::vector<std::shared_ptr<RAMInstruction>> operators,
             bool debug_flag = false) {
    auto op = std::make_shared<StaticBatch>(operators);
    op->debug_flag = debug_flag;
    return op;
}

inline std::shared_ptr<SetColumnStrategy>
set_column_strategy(column_t column, IndexStrategy strategy,
                    bool debug_flag = false) {
    auto op = std::make_shared<SetColumnStrategy>(column, strategy);
    op->debug_flag = debug_flag;
    return op;
}

inline std::shared_ptr<DefaultColumn>
default_column(rel_t rel, size_t column_idx, bool debug_flag = false) {
    auto op = std::make_shared<DefaultColumn>(rel, column_idx);
    op->debug_flag = debug_flag;
    return op;
}

inline std::shared_ptr<LoadFile> load_file(rel_t rel, const char *file_path,
                                           int arity, bool debug_flag = false) {
    auto op = std::make_shared<LoadFile>(rel, file_path, arity);
    op->debug_flag = debug_flag;
    return op;
}

inline std::shared_ptr<Fact> fact(rel_t rel,
                                  std::vector<std::vector<uint32_t>> data,
                                  bool debug_flag = false) {
    auto op = std::make_shared<Fact>(rel, data);
    op->debug_flag = debug_flag;
    return op;
}

inline std::shared_ptr<Index> index_op(rel_t rel, std::vector<int> columns,
                                       bool build_immediately = false,
                                       bool debug_flag = false) {
    auto op = std::make_shared<Index>(rel, columns, build_immediately);
    op->debug_flag = debug_flag;
    return op;
}

inline std::shared_ptr<HashDigest>
hash_digest(std::vector<std::string> meta_vars, std::vector<column_t> columns,
            std::string result_register, bool debug_flag = false) {
    auto op = std::make_shared<HashDigest>(meta_vars, columns, result_register);
    op->debug_flag = debug_flag;
    return op;
}

inline std::shared_ptr<NegateMulti>
negate_multi(rel_t inner, std::vector<column_t> outer_columns,
             std::vector<std::string> meta_vars, bool debug_flag = false) {
    auto op = std::make_shared<NegateMulti>(inner, outer_columns, meta_vars);
    op->debug_flag = debug_flag;
    return op;
}

inline std::shared_ptr<CartesianOperator>
cartesian_op(rel_t rel1, RelationVersion version1, rel_t rel2,
             RelationVersion version2, std::string result_register1,
             std::string result_register2, bool debug_flag = false) {
    auto op = std::make_shared<CartesianOperator>(
        rel1, version1, rel2, version2, result_register1, result_register2);
    op->debug_flag = debug_flag;
    return op;
}

inline std::shared_ptr<ProjectHostOperator>
project_host_op(uint32_t **host_src, size_t src_size, rel_t dst,
                bool debug_flag = false) {
    auto op = std::make_shared<ProjectHostOperator>(host_src, src_size, dst);
    op->debug_flag = debug_flag;
    return op;
}

} // namespace ram

} // namespace vflog

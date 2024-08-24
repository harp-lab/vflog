
#pragma once

#include "ra.cuh"
#include <string>
#include <vector>

namespace vflog {

struct RelationalAlgebraMachine {
    std::vector<std::shared_ptr<RAMInstruction>> operators;
    std::map<std::string, rel_ptr> rels;
    std::shared_ptr<d_buffer> global_buffer;
    host_buf_ref_t cached_indices;
    host_buf_ref_t register_map;

    int iter_counter = 0;

    // a relation will be split into multiple relations if it exceeds max_tuples
    // during materialization, if overflow happens, the new relation will be
    // stored in overflow_rel, in each iteration, only 1 relation can be split.
    rel_ptr overflow_rel = nullptr;
    std::string overflow_rel_name;

    // switch to disable split of relations, usually used when need operation on NEWT
    bool disable_split = false;

    std::map<std::string, std::vector<rel_ptr>> frozen_rels;
    std::map<std::string, uint32_t> rel_arities;

    RelationalAlgebraMachine() = default;

    void
    add_operator(std::vector<std::shared_ptr<RAMInstruction>> new_operators);

    void add_rel(std::string name, rel_ptr rel) { rels[name] = rel; }

    void extend_register(std::string name) {
        register_map[name] = std::make_shared<device_indices_t>();
    }

    std::shared_ptr<multi_hisa> create_rel(std::string name, size_t arity,
                                           char *data_path = nullptr);

    void execute();

    std::shared_ptr<device_indices_t> get_register(std::string name) {
        return register_map[name];
    }

    // split a relation, store old data into frozen_rels
    void split_relation(std::string rel_name);

    bool has_frozen(std::string rel_name) {
        return frozen_rels.find(rel_name) != frozen_rels.end();
    }

    rel_ptr get_frozen(std::string rel_name, int idx) {
        if (frozen_rels.find(rel_name) == frozen_rels.end()) {
            return nullptr;
        }
        if (idx >= frozen_rels[rel_name].size()) {
            return nullptr;
        }
        return frozen_rels[rel_name][idx];
    }

    int get_frozen_size(std::string rel_name) {
        if (frozen_rels.find(rel_name) == frozen_rels.end()) {
            return 0;
        }
        return frozen_rels[rel_name].size();
    }

    void flush_overflow() {
        rels[overflow_rel_name] = overflow_rel;
        overflow_rel = nullptr;
        overflow_rel_name = "";
    }

    bool has_overflow() { return overflow_rel != nullptr; }

    void reset_iter_counter() { iter_counter = 0; }
    void inc_iter_counter() { iter_counter += 1; }
};

} // namespace vflog

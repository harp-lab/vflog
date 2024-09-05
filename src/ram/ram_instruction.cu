
#include "fmt/ostream.h"
#include "io.cuh"
#include "ra.cuh"
#include "ram.cuh"
#include "utils.cuh"
#include <cstdint>
#include <iostream>
#include <string>

namespace vflog::ram {

void CacheInit::execute(RelationalAlgebraMachine &ram) {
    auto rel_p = ram.rels[rel.name];
    auto &cached_indices = ram.cached_indices;
    cached_indices[meta_var]->resize(rel_p->get_versioned_size(version));
    thrust::sequence(EXE_POLICY, cached_indices[meta_var]->begin(),
                     cached_indices[meta_var]->end());
}

std::string CacheInit::to_string() {
    // cache_init("c_edge1", rel_t("edge"), FULL),
    return "cache_init(\"" + meta_var + "\", " + rel.to_string() + ", " +
           version_to_string(version) + ")";
}

void CacheUpdate::execute(RelationalAlgebraMachine &ram) {
    auto &cached_indices = ram.cached_indices;
    if (indices == nullptr) {
        cached_indices[meta_var] = ram.get_register(existing_register);
    } else {
        cached_indices[meta_var] = indices;
    }
}

std::string CacheUpdate::to_string() {
    // cache_update("c_edge1", "r_x"),
    return "cache_update(\"" + meta_var + "\", \"" + existing_register + "\")";
}

void CacheClear::execute(RelationalAlgebraMachine &ram) {
    auto &cached_indices = ram.cached_indices;
    for (auto &meta : cached_indices) {
        meta.second->resize(0);
    }
    cached_indices.clear();
}

void PrepareMaterialization::execute(RelationalAlgebraMachine &ram) {
    auto rel_p = ram.rels[rel.name];
    auto &cached_indices = ram.cached_indices;
    if (cached_indices[meta_var]->size() == 0) {
        return;
    }
    // check if overflow happens
    if (ram.overflow_rel == nullptr && ram.disable_split == false) {
        if (rel_p->split_mode == SPLIT_SIZE &&
            rel_p->total_tuples + cached_indices[meta_var]->size() >
                rel_p->max_tuples) {
            // check if current newt size is empty
            if (rel_p->newt_size != 0) {
                // TODO: unimplemented
                throw std::runtime_error(
                    "Overflow happens, but newt is not empty");
            }
            // split the relation
            ram.split_relation(rel.name);
            ram.overflow_rel->allocate_newt(cached_indices[meta_var]->size());
            return;
        }
        if (rel_p->split_mode == SPLIT_ITER &&
            ram.iter_counter >= rel_p->split_iter_count) {
            // check if current newt size is empty
            if (rel_p->newt_size != 0) {
                //  TODO: unimplemented
                throw std::runtime_error(
                    "Overflow happens, but newt is not empty");
            }
            // split the relation
            ram.split_relation(rel.name);
            ram.overflow_rel->allocate_newt(cached_indices[meta_var]->size());
            return;
        }
    }
    rel_p->allocate_newt(cached_indices[meta_var]->size());
}

std::string PrepareMaterialization::to_string() {
    return "prepare_materialization(" + rel.to_string() + ", \"" + meta_var +
           "\")";
}

void EndMaterialization::execute(RelationalAlgebraMachine &ram) {
    if (ram.cached_indices[meta_var]->size() == 0) {
        return;
    }
    if (ram.overflow_rel_name == rel.name) {
        ram.overflow_rel->newt_size = ram.cached_indices[meta_var]->size();
        ram.overflow_rel->total_tuples += ram.overflow_rel->newt_size;
        return;
    }
    auto rel_p = ram.rels[rel.name];
    auto &cached_indices = ram.cached_indices;
    rel_p->newt_size = cached_indices[meta_var]->size();
    rel_p->total_tuples += rel_p->newt_size;
}

std::string EndMaterialization::to_string() {
    return "end_materialization(" + rel.to_string() + ", \"" + meta_var + "\")";
}

void Persistent::execute(RelationalAlgebraMachine &ram) {
    if (rel.name == ram.overflow_rel_name) {
        // deduplicate will all frozen relations
        int idx = 0;
        // ram.overflow_rel->newt_self_deduplicate();
        for (auto &frozen : ram.frozen_rels[rel.name]) {
            std::cout << "Diff with frozen " << idx << std::endl;
            frozen->diff(*ram.overflow_rel, NEWT, _indices_default);
            idx++;
        }
        ram.overflow_rel->newt_self_deduplicate();
        ram.overflow_rel->persist_newt();
        return;
    }
    auto rel_p = ram.rels[rel.name];
    // deduplicate will all frozen relations
    rel_p->newt_self_deduplicate();
    for (auto &frozen : ram.frozen_rels[rel.name]) {
        frozen->diff(*rel_p, NEWT, _indices_default);
    }
    rel_p->persist_newt();
}

void PrintSize::execute(RelationalAlgebraMachine &ram) {
    uint32_t delta_size = ram.rels[rel.name]->get_versioned_size(DELTA);
    uint32_t newt_size = ram.rels[rel.name]->get_versioned_size(NEWT);
    uint32_t full_size = ram.rels[rel.name]->get_versioned_size(FULL);
    if (ram.has_frozen(rel.name)) {
        for (auto &frozen : ram.frozen_rels[rel.name]) {
            full_size += frozen->get_versioned_size(FULL);
        }
    }
    std::cout << rel.name << " Relation has FULL " << full_size << " DELTA "
              << delta_size << " NEWT " << newt_size << std::endl;
}

std::string PrintSize::to_string() {
    return "print_size(" + rel.to_string() + ")";
}

void DiffOperator::execute(RelationalAlgebraMachine &ram) {
    auto src_t = ram.rels[src.name];
    auto dst_t = ram.rels[dst.name];
    src_t->diff(*dst_t, NEWT, *diff_indices);
    if (ram.has_frozen(src.name)) {
        for (auto &frozen : ram.frozen_rels[src.name]) {
            src_t->diff(*frozen, NEWT, *diff_indices);
        }
    }
}

void ProjectOperator::execute(RelationalAlgebraMachine &ram) {
    auto src_ptr = ram.rels[src.rel];
    if (src.version == FULL && src.is_frozen()) {
        src_ptr = ram.get_frozen(src.rel, src.frozen_idx);
        if (src_ptr == nullptr) {
            // frozen relation not generated yet
            return;
        }
    }
    if (src_ptr->get_versioned_size(src.version) == 0) {
        return;
    }
    if (ram.overflow_rel_name == dst.rel) {
        std::cout << "<<<<< " << std::endl;
        column_copy(*src_ptr, src.version, src.idx, *ram.overflow_rel,
                    dst.version, dst.idx, ram.cached_indices[meta_var]);
    } else {
        column_copy(*src_ptr, src.version, src.idx, *ram.rels[dst.rel],
                    dst.version, dst.idx, ram.cached_indices[meta_var]);
    }
}

std::string ProjectOperator::to_string() {
    return "project_op(" + src.to_string() + ", " + dst.to_string() + ", \"" +
           meta_var + "\")";
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

void MultiArityJoinOperator::execute(RelationalAlgebraMachine &ram) {
    // TODO: support frozen relations
    multi_arities_join(ram, inner_columns, outer_columns, inner_reg,
                       outer_meta_vars, pop_outer);
}

std::string MultiArityJoinOperator::to_string() {
    std::string res = "multi_arity_join(";
    res += "{";
    for (auto &col : inner_columns) {
        res += col.to_string() + ", ";
    }
    res += "}, {";
    for (auto &col : outer_columns) {
        res += col.to_string() + ", ";
    }
    res += "}, \"" + inner_reg + "\", {";
    for (auto &meta : outer_meta_vars) {
        res += "\"" + meta + "\", ";
    }
    res += "}, " + std::to_string(pop_outer) + ")";
    return res;
}

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

void Declaration::execute(RelationalAlgebraMachine &ram) {
    if (data_path == nullptr) {
        ram.create_rel(name, arity);
    } else {
        ram.create_rel(name, arity, data_path);
    }
}

std::string Declaration::to_string() {
    if (data_path == nullptr) {
        return "create_rel(\"" + name + "\", " + std::to_string(arity) + ")";
    } else {
        return "create_rel(\"" + name + "\", " + std::to_string(arity) +
               ", \"" + std::string(data_path) + "\")";
    }
}

void ExtendRegister::execute(RelationalAlgebraMachine &ram) {
    ram.extend_register(name);
}

void FillOperator::execute(RelationalAlgebraMachine &ram) {
    auto dst_rel_p = ram.rels[dst.rel];
    auto newt_head_ptr = dst_rel_p->get_raw_data_ptrs(NEWT, dst.idx);
    auto total_size = dst_rel_p->data[dst.idx].size();
    thrust::fill(EXE_POLICY, newt_head_ptr,
                 dst_rel_p->data[dst.idx].data().get() + total_size, value);
}

std::string FillOperator::to_string() {
    return "fill_op(" + dst.to_string() + ", " + std::to_string(value) + ")";
}

void StaticBatch::execute(RelationalAlgebraMachine &ram) {
    for (auto &op : operators) {
        op->execute(ram);
    }
}

std::string StaticBatch::to_string() {
    std::string res = "\nstatic_batch({\n";
    for (auto &op : operators) {
        res += op->to_string() + ", \n";
    }
    res += "})";
    return res;
}

void SetColumnStrategy::execute(RelationalAlgebraMachine &ram) {
    auto rel_p = ram.rels[column.rel];
    rel_p->set_index_startegy(column.idx, column.version, strategy);
    // rel_p->set_index_startegy(column.idx, DELTA, strategy);
}

std::string SetColumnStrategy::to_string() {
    return "set_column_strategy(" + column.to_string() + ", " +
           index_strategy_to_string(strategy) + ")";
}

void Index::execute(RelationalAlgebraMachine &ram) {
    auto rel_p = ram.rels[rel.name];
    if (columns.size() == 1) {
        rel_p->set_index_startegy(columns[0], FULL, IndexStrategy::EAGER);
    } else {
        rel_p->add_clustered_index(columns);
    }
}

std::string Index::to_string() {
    if (columns.size() == 1) {
        return "index(" + rel.to_string() + ", " + std::to_string(columns[0]) +
               ")";
    } else {
        std::string res = "index(" + rel.to_string() + ", {";
        for (auto &col : columns) {
            res += std::to_string(col) + ", ";
        }
        res += "})";
        return res;
    }
}

void DefaultColumn::execute(RelationalAlgebraMachine &ram) {
    auto rel_p = ram.rels[rel.name];
    rel_p->set_default_index_column(column_idx);
}

std::string DefaultColumn::to_string() {
    return "default_column(" + rel.to_string() + ", " +
           std::to_string(column_idx) + ")";
}

void LoadFile::execute(RelationalAlgebraMachine &ram) {
    auto rel_p = ram.rels[rel.name];
    read_kary_relation(file_path, *rel_p, arity);
    rel_p->newt_self_deduplicate();
    rel_p->persist_newt();
    std::cout << "Loaded " << rel.name << " from " << file_path << std::endl;
    std::cout << "Relation " << rel.name << " has " << rel_p->full_size
              << " tuples" << std::endl;
}

std::string LoadFile::to_string() {
    if (file_path == nullptr) {
        return "load_file(\"" + rel.name + "\", " + rel.to_string() + ")";
    }
    return "load_file(\"" + std::string(file_path) + "\", " + rel.to_string() +
           ")";
}

} // namespace vflog::ram

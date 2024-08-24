
#include "ram.cuh"
#include <cstdint>

namespace vflog {

void CacheInit::execute(RelationalAlgebraMachine &ram) {
    auto rel_p = ram.rels[rel.name];
    auto &cached_indices = ram.cached_indices;
    cached_indices[meta_var]->resize(rel_p->get_versioned_size(version));
    thrust::sequence(EXE_POLICY, cached_indices[meta_var]->begin(),
                     cached_indices[meta_var]->end());
}

void CacheUpdate::execute(RelationalAlgebraMachine &ram) {
    auto &cached_indices = ram.cached_indices;
    if (indices == nullptr) {
        cached_indices[meta_var] = ram.get_register(existing_register);
    } else {
        cached_indices[meta_var] = indices;
    }
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
                throw std::runtime_error("Overflow happens, but newt is not empty");
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
                throw std::runtime_error("Overflow happens, but newt is not empty");
            }
            // split the relation
            ram.split_relation(rel.name);
            ram.overflow_rel->allocate_newt(cached_indices[meta_var]->size());
            return;
        }
    }
    rel_p->allocate_newt(cached_indices[meta_var]->size());
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

void Persistent::execute(RelationalAlgebraMachine &ram) {
    if (rel.name == ram.overflow_rel_name) {
        // deduplicate will all frozen relations
        int idx = 0;
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

} // namespace vflog

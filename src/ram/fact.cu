
#include "ram.cuh"
#include "ram_instruction.cuh"

namespace vflog::ram {

void Fact::execute(RelationalAlgebraMachine &ram) {
    auto new_size = data.size();
    auto rel_ptr = ram.get_rel(rel.name);

    if (rel_ptr == nullptr) {
        throw std::runtime_error("Relation not found: " + rel.name);
    }

    rel_ptr->allocate_newt(new_size);
    auto arity = rel_ptr->arity;
    auto full_size = rel_ptr->full_size;
    for (size_t i = 0; i < new_size; i++) {
        for (size_t j = 0; j < arity; j++) {
            auto &raw_data = rel_ptr->data[j];
            raw_data[full_size + i] = data[i][j];
        }
    }

    // persist the relation
    rel_ptr->newt_self_deduplicate();
    rel_ptr->persist_newt();
}

std::string Fact::to_string() {
    std::string ret = "Fact(" + rel.name + ", [";
    for (size_t i = 0; i < data.size(); i++) {
        ret += "[";
        for (size_t j = 0; j < data[i].size(); j++) {
            ret += std::to_string(data[i][j]);
            if (j != data[i].size() - 1) {
                ret += ", ";
            }
        }
        ret += "]";
        if (i != data.size() - 1) {
            ret += ", ";
        }
    }
    ret += "])";
    return ret;
}

} // namespace vflog::ram

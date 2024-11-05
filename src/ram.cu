
#include "ram.cuh"

namespace vflog::ram {

void RelationalAlgebraMachine::add_program(std::shared_ptr<RAMProgram> program) {
    for (auto &op : program->instructions) {
        operators.push_back(op);
    }
}

void RelationalAlgebraMachine::add_operator(
    std::vector<std::shared_ptr<RAMInstruction>> new_operators) {
    for (auto &op : new_operators) {
        operators.push_back(op);
    }
}

std::shared_ptr<multi_hisa>
RelationalAlgebraMachine::create_rel(std::string name, size_t arity,
                                     const char *data_path) {
    if (data_path == nullptr) {
        rels[name] = std::make_shared<multi_hisa>(name, arity, global_buffer);
    } else {
        rels[name] =
            std::make_shared<multi_hisa>(name, arity, data_path, global_buffer);
    }
    auto total_rel_size = rels.size();
    rels[name]->uid = total_rel_size;
    rel_arities[name] = arity;
    return rels[name];
}

void RelationalAlgebraMachine::execute() {
    std::cout << "Executing " << operators.size() << " operators" << std::endl;
    auto iter = 0;
    for (auto &op : operators) {
        // std::cout << "Executing operator " << iter << std::endl;
        // std::cout << op->to_string() << std::endl;
        op->execute(*this);
        iter += 1;
    }
    cached_indices.clear();
}

void RelationalAlgebraMachine::split_relation(std::string rel_name) {
    int frozen_size = 0;
    auto rel = rels[rel_name];
    if (frozen_rels.find(rel_name) == frozen_rels.end()) {
        frozen_rels[rel_name] = std::vector<rel_ptr>();
        frozen_size = 0;
    } else {
        frozen_size = frozen_rels[rel_name].size();
    }
    // allocate a new relation name based on frozen_size
    auto frozen_rel_name = rel_name + "@frozen_" + std::to_string(frozen_size);
    frozen_rels[rel_name].push_back(rel);
    std::cout << "Splitting relation " << rel_name << " into " << frozen_rel_name
              << std::endl;

    overflow_rel =
        std::make_shared<multi_hisa>(rel_name, rel_arities[rel_name], global_buffer);
    overflow_rel->copy_meta(*rel);
    overflow_rel_name = rel_name;
    rels[frozen_rel_name] = rel;
}

} // namespace vflog

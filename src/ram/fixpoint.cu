
#include "ram.cuh"
#include "ram_instruction.cuh"
#include <cstddef>
#include <memory>

namespace vflog::ram {

void FixpointOperator::execute(RelationalAlgebraMachine &ram) {
    int iter = 0;
    KernelTimer timer;
    ram.reset_iter_counter();
    for (size_t i = 0; i < operators.size(); i++) {
        std::cout << "SCC Operator " << i << " " << operators[i]->to_string()
                  << std::endl;
    }
    while (true) {
        std::cout << "Iteration " << iter << std::endl;
        if (this->step(ram)) {
            break;
        }
    }
    double total_time = 0;
    std::cout << "Total iterations: " << iter << std::endl;
    for (auto &stat : stats) {
        total_time += stat.second;
        std::cout << "Operator " << stat.first << " took " << stat.second << "s"
                  << std::endl;
    }
    std::cout << "Total time: " << total_time << "s" << std::endl;
}

bool FixpointOperator::step(RelationalAlgebraMachine &ram) {
    // std::cout << "Iteration " << iter << std::endl;
    int i = 0;
    KernelTimer timer;
    // check if each relation has reach max split
    for (auto &rel : rels) {
        auto rel_p = ram.rels[rel.name];
        if (ram.get_frozen_size(rel.name) >= rel_p->max_splits) {
            // disable split
            rel_p->stop_split();
        }
    }

    for (auto &op : operators) {
        std::cout << "Executing operator " << i << " " << op->to_string()
                  << std::endl;
        timer.start_timer();
        op->execute(ram);
        timer.stop_timer();
        auto elapsed = timer.get_spent_time();
        if (stats.find(i) == stats.end()) {
            stats[i] = elapsed;
        } else {
            stats[i] += elapsed;
        }
        i += 1;
    }
    bool flag = true;
    for (auto &rel : rels) {
        auto rel_p = ram.rels[rel.name];
        if (rel_p->get_versioned_size(DELTA) != 0) {
            flag = false;
        }
        if (ram.has_overflow()) {
            ram.flush_overflow();
        }
        ram.rels[rel.name]->inc_iter();
    }
    if (max_iter != -1 && iter >= max_iter) {
        return true;
    }
    if (flag) {
        return true;
    }
    iter += 1;
    ram.inc_iter_counter();
    ram.cached_indices.clear();
    return false;
}

std::string FixpointOperator::to_string() {
    std::string ret = "\nfixpoint_op({\n";
    for (auto &op : operators) {
        ret += op->to_string() + ",\n";
    }
    ret += "}, {";
    for (auto &rel : rels) {
        ret += rel.to_string() + ",";
    }
    ret += "})";
    return ret;
}

} // namespace vflog::ram

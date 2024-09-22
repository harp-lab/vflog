
#include "mir.cuh"
#include "ram_instruction.cuh"
#include "utils.cuh"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vflog::mir {

std::vector<std::shared_ptr<ram::RAMInstruction>>
RAMGenPass::allocate_registers() {
    // look all rules count the max clause size
    std::vector<std::shared_ptr<ram::RAMInstruction>> instrs;
    int max_clause_size = 0;
    for (auto node : mir_program->lines()) {
        if (node->type == MIRNodeType::SCC) {
            // cast to MIRScc
            auto scc = std::dynamic_pointer_cast<MIRScc>(node);
            for (auto line : scc->lines()) {
                if (line->type == MIRNodeType::RULE) {
                    // cast to MIRRule
                    auto rule = std::dynamic_pointer_cast<MIRRule>(line);
                    int clause_size = rule->body.size();

                    if (clause_size > max_clause_size) {
                        max_clause_size = clause_size;
                    }
                }
            }
        }
    }

    // allocate registers
    for (int i = 0; i < max_clause_size; i++) {
        instrs.push_back(ram::extend_register("r" + std::to_string(i)));
    }
    return instrs;
}

void RAMGenPass::compile() {
    auto reg_instr = allocate_registers();
    auto decl_instr = declare_relations();

    std::vector<std::shared_ptr<ram::RAMInstruction>> comp_instr;
    for (auto line : mir_program->lines()) {
        if (line->type == MIRNodeType::SCC) {
            // cast to MIRScc
            auto scc = std::dynamic_pointer_cast<MIRScc>(line);
            auto fixpoint_op = compile_scc(scc);
            comp_instr.push_back(fixpoint_op);
        }
        if (line->type == MIRNodeType::CUSTOM) {
            auto custom_node = std::dynamic_pointer_cast<MIRCustomNode>(line);
            comp_instr.push_back(ram::custom_op(custom_node->func));
        }
    }

    ram_instructions->add_instructions(decl_instr);

    for (auto p : joined_rel_columns) {
        std::string rel_name = p.first;
        std::vector<int> columns = p.second;
        if (columns[0] != 0) {
            ram_instructions->add_instruction(
                ram::default_column(ram::rel_t(rel_name), columns[0]));
        }
        for (auto col_idx : columns) {
            if (col_idx != 0) {
                ram_instructions->add_instruction(ram::set_column_strategy(
                    column_t(rel_name, col_idx, FULL), EAGER));
            }
        }
    }
    ram_instructions->add_instructions(io_instructions);

    ram_instructions->add_instructions(reg_instr);
    ram_instructions->add_instructions(comp_instr);
}

} // namespace vflog::mir

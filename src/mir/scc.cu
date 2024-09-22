#include "mir.cuh"
#include "ram_instruction.cuh"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace vflog::mir {

std::shared_ptr<ram::RAMInstruction>
RAMGenPass::compile_scc(std::shared_ptr<MIRScc> scc) {
    // check all relation name in a scc
    std::vector<std::string> updated_rel_names;
    for (auto line : scc->lines()) {
        if (line->type == MIRNodeType::RULE) {
            // cast to MIRRule
            auto rule = std::dynamic_pointer_cast<MIRRule>(line);
            if (rule->head->type == MIRNodeType::SINGLE_HEAD) {
                auto head_clause =
                    std::dynamic_pointer_cast<MIRSingleHeadClause>(rule->head);
                if (std::find(updated_rel_names.begin(),
                              updated_rel_names.end(), head_clause->rel_name) ==
                    updated_rel_names.end()) {
                    updated_rel_names.push_back(head_clause->rel_name);
                }
            } else if (rule->head->type == MIRNodeType::CONJ_HEAD) {
                auto head_clause =
                    std::dynamic_pointer_cast<MIRConjHeadClause>(rule->head);
                auto head_names = head_clause->get_all_rel_names();
                for (auto name : head_names) {
                    if (std::find(updated_rel_names.begin(),
                                  updated_rel_names.end(),
                                  name) == updated_rel_names.end()) {
                        updated_rel_names.push_back(name);
                    }
                }
            } else {
                throw std::runtime_error("unsupported head clause type.");
            }
        }
    }
    // std::vector
    std::vector<ram::rel_t> updated_rels;
    for (auto rel_name : updated_rel_names) {
        updated_rels.push_back(ram::rel_t(rel_name));
    }

    std::vector<std::shared_ptr<ram::RAMInstruction>> fixpoint_instrs;

    // compile all rules in a scc
    for (auto line : scc->lines()) {
        if (line->type == MIRNodeType::RULE) {
            // cast to MIRRule
            auto rule = std::dynamic_pointer_cast<MIRRule>(line);
            auto rule_instrs =
                compile_rule(rule, updated_rel_names, scc->recursive);
            fixpoint_instrs.insert(fixpoint_instrs.end(), rule_instrs.begin(),
                                   rule_instrs.end());
        }
        if (line->type == MIRNodeType::FACT) {
            // cast to MIRFact
            auto fact = std::dynamic_pointer_cast<MIRFact>(line);
            auto fact_instr = compile_fact(fact);
            fixpoint_instrs.push_back(fact_instr);
        }
    }

    fixpoint_instrs.push_back(ram::cache_clear());
    // persistent all relations need to be updated
    for (auto rel : updated_rels) {
        fixpoint_instrs.push_back(ram::persistent(ram::rel_t(rel.name)));
        fixpoint_instrs.push_back(ram::print_size(ram::rel_t(rel.name)));
    }

    if (scc->recursive) {
        return ram::fixpoint_op(fixpoint_instrs, updated_rels);
    } else {
        return ram::static_batch(fixpoint_instrs);
    }
}
} // namespace vflog::mir

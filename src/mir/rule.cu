#include "mir.cuh"
#include "ram_instruction.cuh"
#include <memory>
#include <string>
#include <vector>

namespace vflog::mir {
std::shared_ptr<ram::RAMInstruction>
RAMGenPass::compile_fact(std::shared_ptr<MIRFact> fact) {
    auto rel = ram::rel_t(fact->rel_name);
    auto instr = ram::fact(rel, fact->data);
    return instr;
}

std::vector<std::shared_ptr<ram::RAMInstruction>>
RAMGenPass::compile_rule(std::shared_ptr<MIRRule> rule,
                         std::vector<std::string> updated_rel_names,
                         bool is_recursive) {
    std::vector<std::shared_ptr<ram::RAMInstruction>> rule_instrs;

    // check body clause size
    int clause_size = 0;
    for (auto body_clause : rule->body) {
        if (body_clause->type == MIRNodeType::BODY_CLAUSE) {
            clause_size++;
        }
    }

    if (clause_size == 1) {
        // copy
        return compile_copy_rules(rule, is_recursive);
    } else {
        // join
        return compile_join_rules(rule, is_recursive);
    }

    // if clause size > 1, need to join

    return rule_instrs;
}
} // namespace vflog::mir

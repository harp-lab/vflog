#include "mir.cuh"
#include "ram_instruction.cuh"
#include "utils.cuh"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vflog::mir {
std::vector<std::shared_ptr<ram::RAMInstruction>>
RAMGenPass::compile_copy_rules(std::shared_ptr<MIRRule> rule,
                               bool is_recursive) {
    // copy rule
    if (rule->body[0]->type != MIRNodeType::BODY_CLAUSE) {
        throw std::runtime_error(
            "put the relation in the 1st body clause when copy");
    }
    std::vector<std::shared_ptr<ram::RAMInstruction>> compiled_instrs;
    auto copy_rule_for_single_head = [&](std::shared_ptr<MIRSingleHeadClause>
                                             head_clause) {
        auto body_clause =
            std::dynamic_pointer_cast<MIRBodyClause>(rule->body[0]);
        RelationVersion ver = is_recursive ? DELTA : FULL;
        // map from dst column to src column
        std::map<int, int> reordered_column_map;
        std::map<int, uint32_t> const_column_map;
        for (int i = 0; i < body_clause->args.size(); i++) {
            auto arg = body_clause->args[i];
            if (arg->type == MIRNodeType::METAVAR) {
                auto var = std::dynamic_pointer_cast<MIRMetavar>(arg);
                // find the position of the variable in the head clause
                bool macthed = false;
                for (int j = 0; j < head_clause->args.size(); j++) {
                    auto head_arg = head_clause->args[j];
                    if (head_arg->type == MIRNodeType::METAVAR) {
                        auto head_var =
                            std::dynamic_pointer_cast<MIRMetavar>(head_arg);
                        if (head_var->name == var->name) {
                            macthed = true;
                            reordered_column_map[j] = i;
                        }
                    }
                }
                if (!macthed) {
                    throw std::runtime_error(
                        "variable in body clause not found in head clause.");
                }
            }
        }

        // check if there is const need to be copied
        for (int i = 0; i < head_clause->args.size(); i++) {
            auto head_arg = head_clause->args[i];
            if (head_arg->type == MIRNodeType::NUMBER) {
                auto num = std::dynamic_pointer_cast<MIRNumber>(head_arg);
                const_column_map[i] = i2d(num->value);
            }
        }
        compiled_instrs.push_back(
            ram::cache_update(body_clause->rel_name, "r0"));
        compiled_instrs.push_back(ram::cache_init(
            body_clause->rel_name, ram::rel_t(body_clause->rel_name), ver));
        compiled_instrs.push_back(ram::prepare_materialization(
            ram::rel_t(head_clause->rel_name), body_clause->rel_name));
        // fill all const
        for (auto const_pair : const_column_map) {
            compiled_instrs.push_back(ram::fill_op(
                column_t(body_clause->rel_name, const_pair.first, ver),
                const_pair.second));
        }
        // project all variables
        for (auto reordered_pair : reordered_column_map) {
            compiled_instrs.push_back(ram::project_op(
                column_t(body_clause->rel_name, reordered_pair.second, ver),
                column_t(head_clause->rel_name, reordered_pair.first, NEWT),
                body_clause->rel_name));
        }

        for (int i = 0; i < head_clause->args.size(); i++) {
            auto head_arg = head_clause->args[i];
            if (head_arg->type == MIRNodeType::ID) {
                // auto id = std::dynamic_pointer_cast<MIRId>(head_arg);
                compiled_instrs.push_back(ram::project_id_op(
                    column_t(head_clause->rel_name, i, NEWT),
                    body_clause->rel_name));
            }
        }

        // handle the rest of clauses
        for (int i = 1; i < rule->body.size(); i++) {
            auto other_body_node = rule->body[i];
            if (other_body_node->type == MIRNodeType::CUSTOM) {
                auto custom_node =
                    std::dynamic_pointer_cast<MIRCustomNode>(other_body_node);
                compiled_instrs.push_back(ram::custom_op(custom_node->func));
            } else {
                throw std::runtime_error(
                    "unsupported body clause type in copy rule.");
            }
        }
        compiled_instrs.push_back(ram::end_materialization(
            ram::rel_t(head_clause->rel_name), body_clause->rel_name));
    };

    if (rule->head->type == SINGLE_HEAD) {
        auto single_head_clause =
            std::dynamic_pointer_cast<MIRSingleHeadClause>(rule->head);
        copy_rule_for_single_head(single_head_clause);
    } else if (rule->head->type == CONJ_HEAD) {
        auto conj_head_clause =
            std::dynamic_pointer_cast<MIRConjHeadClause>(rule->head);
        for (auto head_clause : conj_head_clause->heads) {
            copy_rule_for_single_head(
                std::dynamic_pointer_cast<MIRSingleHeadClause>(head_clause));
        }
    }
    compiled_instrs.push_back(ram::cache_clear());
    return compiled_instrs;
}
} // namespace vflog::mir

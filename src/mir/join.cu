#include "mir.cuh"
#include "ram_instruction.cuh"
#include "utils.cuh"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vflog::mir {

void RAMGenPass::add_join_column(std::string rel_name, int column) {
    if (joined_rel_columns.find(rel_name) == joined_rel_columns.end()) {
        joined_rel_columns[rel_name] = std::vector<int>();
        joined_rel_columns[rel_name].push_back(column);
    } else {
        auto &columns = joined_rel_columns[rel_name];
        if (std::find(columns.begin(), columns.end(), column) ==
            columns.end()) {
            columns.push_back(column);
        }
    }
}

std::vector<std::shared_ptr<ram::RAMInstruction>>
RAMGenPass::compile_materialization(std::shared_ptr<MIRRule> rule,
                                    bool is_recursive) {
    std::vector<std::shared_ptr<ram::RAMInstruction>> rule_instrs;
    std::function<void(std::shared_ptr<MIRNode>)> materialize_one_head =
        [&](std::shared_ptr<MIRNode> head) {
            if (head->type == MIRNodeType::SINGLE_HEAD) {
                auto head_clause =
                    std::dynamic_pointer_cast<MIRSingleHeadClause>(head);
                rule_instrs.push_back(ram::prepare_materialization(
                    ram::rel_t(head_clause->rel_name),
                    "c" + std::to_string(rule->body.size() - 1)));
                // project each column in head clause
                for (int i = 0; i < head_clause->args.size(); i++) {
                    auto arg_node = head_clause->args[i];
                    if (arg_node->type == MIRNodeType::METAVAR) {
                        auto var_node =
                            std::dynamic_pointer_cast<MIRMetavar>(arg_node);
                        for (int j = 0; j < rule->body.size(); j++) {
                            auto body_node = rule->body[j];
                            if (body_node->type == MIRNodeType::BODY_CLAUSE) {
                                auto body_clause =
                                    std::dynamic_pointer_cast<MIRBodyClause>(
                                        body_node);
                                bool matched = false;
                                for (int k = 0; k < body_clause->args.size();
                                     k++) {
                                    auto arg = body_clause->args[k];
                                    if (arg->type == MIRNodeType::METAVAR) {
                                        auto var = std::dynamic_pointer_cast<
                                            MIRMetavar>(arg);
                                        if (var->name == var_node->name) {
                                            RelationVersion ver =
                                                (is_recursive && j == 0) ? DELTA
                                                                         : FULL;
                                            rule_instrs.push_back(
                                                ram::project_op(
                                                    column_t(
                                                        body_clause->rel_name,
                                                        k, ver),
                                                    column_t(
                                                        head_clause->rel_name,
                                                        i, NEWT),
                                                    "c" + std::to_string(j)));
                                            matched = true;
                                            break;
                                        }
                                    }
                                }
                                if (matched) {
                                    break;
                                }
                            }
                        }
                    }
                    if (arg_node->type == MIRNodeType::NUMBER) {
                        auto num_node =
                            std::dynamic_pointer_cast<MIRNumber>(arg_node);
                        rule_instrs.push_back(ram::fill_op(
                            column_t(head_clause->rel_name, i, NEWT),
                            i2d(num_node->value)));
                    }
                    if (arg_node->type == MIRNodeType::ID) {
                        auto id_node =
                            std::dynamic_pointer_cast<MIRId>(arg_node);
                        for (int j = rule->body.size() - 1; j >= 0; j--) {
                            if (rule->body[j]->type ==
                                MIRNodeType::BODY_CLAUSE) {
                                auto body_clause =
                                    std::dynamic_pointer_cast<MIRBodyClause>(
                                        rule->body[j]);
                                if (body_clause->with_id &&
                                    body_clause->id_var_name ==
                                        id_node->var_name) {
                                    rule_instrs.push_back(ram::project_id_op(
                                        column_t(head_clause->rel_name, i,
                                                 NEWT),
                                        "c" + std::to_string(j)));
                                    break;
                                }
                            }
                        }
                    }
                }
                rule_instrs.push_back(ram::end_materialization(
                    ram::rel_t(head_clause->rel_name),
                    "c" + std::to_string(rule->body.size() - 1)));
            } else if (head->type == MIRNodeType::CONJ_HEAD) {
                auto conj_head_clause =
                    std::dynamic_pointer_cast<MIRConjHeadClause>(head);
                for (auto head_clause : conj_head_clause->heads) {
                    materialize_one_head(head_clause);
                }
            }
        };
    materialize_one_head(rule->head);
}

std::vector<std::shared_ptr<ram::RAMInstruction>>
RAMGenPass::compile_join_rules(std::shared_ptr<MIRRule> rule,
                               bool is_recursive) {
    std::vector<std::shared_ptr<ram::RAMInstruction>> rule_instrs;
    // we always consider the first body clause as the delta if recursive
    for (int i = 0; i < rule->body.size(); i++) {
        auto cur_node = rule->body[i];
        if (cur_node->type == MIRNodeType::BODY_CLAUSE) {
            auto cur_body_clause =
                std::dynamic_pointer_cast<MIRBodyClause>(cur_node);
            // std::cout << "body clause: " << i << " " <<
            // cur_body_clause->rel_name
            //           << std::endl;
            if (i == 0) {
                rule_instrs.push_back(
                    ram::cache_update("c" + std::to_string(i), "r0"));
                RelationVersion ver = is_recursive ? DELTA : FULL;
                rule_instrs.push_back(ram::cache_init(
                    "c" + std::to_string(i),
                    ram::rel_t(cur_body_clause->rel_name), ver));
            } else {
                // find joined column in all clauses before it
                std::map<int, int> reordered_column_map;
                for (int j = 0; j < i; j++) {
                    auto prev_node = rule->body[j];
                    bool joined = false;
                    if (prev_node->type == MIRNodeType::BODY_CLAUSE) {
                        auto prev_body_clause =
                            std::dynamic_pointer_cast<MIRBodyClause>(prev_node);
                        if (prev_body_clause->with_id) {
                            // check if id of this prev clause is used in cur
                            for (int k = 0; k < cur_body_clause->args.size();
                                 k++) {
                                auto arg = cur_body_clause->args[k];
                                if (arg->type == MIRNodeType::ID) {
                                    auto id_node =
                                        std::dynamic_pointer_cast<MIRId>(arg);
                                    if (id_node->var_name ==
                                        prev_body_clause->id_var_name) {
                                        // join
                                        joined = true;
                                        add_join_column(
                                            cur_body_clause->rel_name, k);
                                        RelationVersion outer_ver =
                                            (is_recursive && j == 0) ? DELTA
                                                                     : FULL;
                                        // TODO: add id join
                                        rule_instrs.push_back(ram::join_id_op(
                                            column_t(cur_body_clause->rel_name,
                                                     k, FULL),
                                            column_t(prev_body_clause->rel_name,
                                                     UINT32_MAX, outer_ver),
                                            "c" + std::to_string(j),
                                            "r" + std::to_string(i)));
                                        rule_instrs.push_back(ram::cache_update(
                                            "c" + std::to_string(i),
                                            "r" + std::to_string(i)));
                                    }
                                }
                            }
                        }
                        for (int k = 0; k < prev_body_clause->args.size();
                             k++) {
                            auto arg = prev_body_clause->args[k];
                            if (arg->type == MIRNodeType::METAVAR) {
                                auto var =
                                    std::dynamic_pointer_cast<MIRMetavar>(arg);
                                // skip wild card
                                if (var->name == "_") {
                                    continue;
                                }
                                // find the position of the variable in current
                                // clause
                                for (int l = 0;
                                     l < cur_body_clause->args.size(); l++) {
                                    auto cur_arg = cur_body_clause->args[l];
                                    if (cur_arg->type == MIRNodeType::METAVAR) {
                                        auto cur_var =
                                            std::dynamic_pointer_cast<
                                                MIRMetavar>(cur_arg);
                                        if (cur_var->name == var->name) {
                                            // join
                                            joined = true;
                                            if (j != 0) {
                                                add_join_column(
                                                    prev_body_clause->rel_name,
                                                    k);
                                            }
                                            add_join_column(
                                                cur_body_clause->rel_name, l);
                                            RelationVersion outer_ver =
                                                (is_recursive && j == 0) ? DELTA
                                                                         : FULL;
                                            rule_instrs.push_back(ram::join_op(
                                                column_t(
                                                    cur_body_clause->rel_name,
                                                    l, FULL),
                                                column_t(
                                                    prev_body_clause->rel_name,
                                                    k, outer_ver),
                                                "c" + std::to_string(j),
                                                "r" + std::to_string(i)));
                                            rule_instrs.push_back(
                                                ram::cache_update(
                                                    "c" + std::to_string(i),
                                                    "r" + std::to_string(i)));
                                        }
                                    }
                                }
                            } else {
                                throw std::runtime_error(
                                    "transformed MIR join need to be const "
                                    "free.");
                            }
                        }
                        if (!joined) {
                            throw std::runtime_error(
                                "catesian product is not supported.");
                        }
                    }
                }
            }
        }
        if (cur_node->type == MIRNodeType::CUSTOM) {
            auto custom_node =
                std::dynamic_pointer_cast<MIRCustomNode>(cur_node);
            rule_instrs.push_back(ram::custom_op(custom_node->func));
        }
    }

    // materialize
    auto materialization_instrs = compile_materialization(rule, is_recursive);
    for (auto instr : materialization_instrs) {
        rule_instrs.push_back(instr);
    }
    // rule_instrs.push_back(ram::cache_clear());

    return rule_instrs;
}
} // namespace vflog::mir

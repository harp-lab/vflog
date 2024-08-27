
#include "mir.cuh"
#include "ram_instruction.cuh"
#include "utils.cuh"
#include <algorithm>
#include <cstdint>
#include <memory>
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

std::vector<std::shared_ptr<ram::RAMInstruction>>
RAMGenPass::declare_relations() {
    // looking for all lines, if its decl, add to ram_instructions
    std::vector<std::shared_ptr<ram::RAMInstruction>> instrs;
    for (auto line : mir_program->lines()) {
        if (line->type == MIRNodeType::DECL) {
            // cast to MIRDecl
            auto mdecl = std::dynamic_pointer_cast<MIRDecl>(line);
            auto decl_instr =
                ram::declare(mdecl->rel_name, mdecl->arg_names.size());
            if (mdecl->data_path != nullptr) {
                io_instructions.push_back(
                    ram::load_file(ram::rel_t(mdecl->rel_name),
                                   mdecl->data_path, mdecl->arg_names.size()));
            }
            instrs.push_back(decl_instr);
        }
    }
    return instrs;
}

std::shared_ptr<ram::RAMInstruction>
RAMGenPass::compile_scc(std::shared_ptr<MIRScc> scc) {
    // check all relation name in a scc
    std::vector<std::string> updated_rel_names;
    for (auto line : scc->lines()) {
        if (line->type == MIRNodeType::RULE) {
            // cast to MIRRule
            auto rule = std::dynamic_pointer_cast<MIRRule>(line);
            // check if head is in updated_rels
            auto head_clause =
                std::dynamic_pointer_cast<MIRHeadClause>(rule->head);
            auto head_rel_name = head_clause->rel_name;
            if (std::find(updated_rel_names.begin(), updated_rel_names.end(),
                          head_rel_name) == updated_rel_names.end()) {
                updated_rel_names.push_back(head_rel_name);
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

std::vector<std::shared_ptr<ram::RAMInstruction>>
RAMGenPass::compile_rule(std::shared_ptr<MIRRule> rule,
                         std::vector<std::string> updated_rel_names,
                         bool is_recursive) {
    std::vector<std::shared_ptr<ram::RAMInstruction>> rule_instrs;

    auto head_clause = std::dynamic_pointer_cast<MIRHeadClause>(rule->head);

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

std::vector<std::shared_ptr<ram::RAMInstruction>>
RAMGenPass::compile_copy_rules(std::shared_ptr<MIRRule> rule,
                               bool is_recursive) {
    auto head_clause = std::dynamic_pointer_cast<MIRHeadClause>(rule->head);
    // copy rule
    if (rule->body[0]->type != MIRNodeType::BODY_CLAUSE) {
        throw std::runtime_error(
            "put the relation in the 1st body clause when copy");
    }
    auto body_clause = std::dynamic_pointer_cast<MIRBodyClause>(rule->body[0]);
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
    std::vector<std::shared_ptr<ram::RAMInstruction>> compiled_instrs = {
        ram::cache_update(body_clause->rel_name, "r0"),
        ram::cache_init(body_clause->rel_name,
                        ram::rel_t(body_clause->rel_name), ver),
        ram::prepare_materialization(ram::rel_t(head_clause->rel_name),
                                     body_clause->rel_name)
    };
    // fill all const
    for (auto const_pair : const_column_map) {
        compiled_instrs.push_back(ram::fill_op(
            ram::column_t(body_clause->rel_name, const_pair.first, ver),
            const_pair.second));
    }
    // project all variables
    for (auto reordered_pair : reordered_column_map) {
        compiled_instrs.push_back(ram::project_op(
            ram::column_t(body_clause->rel_name, reordered_pair.second, ver),
            ram::column_t(head_clause->rel_name, reordered_pair.first, NEWT),
            body_clause->rel_name));
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

    return compiled_instrs;
}

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
RAMGenPass::compile_join_rules(std::shared_ptr<MIRRule> rule,
                               bool is_recursive) {
    std::vector<std::shared_ptr<ram::RAMInstruction>> rule_instrs;

    auto head_clause = std::dynamic_pointer_cast<MIRHeadClause>(rule->head);

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
                                                ram::column_t(
                                                    cur_body_clause->rel_name,
                                                    l, FULL),
                                                ram::column_t(
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
    rule_instrs.push_back(ram::prepare_materialization(
        ram::rel_t(head_clause->rel_name),
        "c" + std::to_string(rule->body.size() - 1)));
    // project each column in head clause
    for (int i = 0; i < head_clause->args.size(); i++) {
        auto arg_node = head_clause->args[i];
        if (arg_node->type == MIRNodeType::METAVAR) {
            auto var_node = std::dynamic_pointer_cast<MIRMetavar>(arg_node);
            for (int j = 0; j < rule->body.size(); j++) {
                auto body_node = rule->body[j];
                if (body_node->type == MIRNodeType::BODY_CLAUSE) {
                    auto body_clause =
                        std::dynamic_pointer_cast<MIRBodyClause>(body_node);
                    bool matched = false;
                    for (int k = 0; k < body_clause->args.size(); k++) {
                        auto arg = body_clause->args[k];
                        if (arg->type == MIRNodeType::METAVAR) {
                            auto var =
                                std::dynamic_pointer_cast<MIRMetavar>(arg);
                            if (var->name == var_node->name) {
                                RelationVersion ver =
                                    (is_recursive && j == 0) ? DELTA : FULL;
                                rule_instrs.push_back(ram::project_op(
                                    ram::column_t(body_clause->rel_name, k,
                                                  ver),
                                    ram::column_t(head_clause->rel_name, i,
                                                  NEWT),
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
            auto num_node = std::dynamic_pointer_cast<MIRNumber>(arg_node);
            rule_instrs.push_back(
                ram::fill_op(ram::column_t(head_clause->rel_name, i, NEWT),
                             i2d(num_node->value)));
        }
    }
    rule_instrs.push_back(
        ram::end_materialization(ram::rel_t(head_clause->rel_name),
                                 "c" + std::to_string(rule->body.size() - 1)));
    rule_instrs.push_back(ram::cache_clear());

    return rule_instrs;
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
                    ram::column_t(rel_name, col_idx, FULL), EAGER));
            }
        }
    }
    ram_instructions->add_instructions(io_instructions);

    ram_instructions->add_instructions(reg_instr);
    ram_instructions->add_instructions(comp_instr);
}

} // namespace vflog::mir

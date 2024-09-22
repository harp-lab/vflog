#include "mir.cuh"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace vflog::mir {

std::vector<std::string> MIRSingleHeadClause::get_all_meta_vars() {
    std::vector<std::string> meta_vars;
    for (auto arg : args) {
        if (arg->type == MIRNodeType::METAVAR) {
            auto var = std::dynamic_pointer_cast<MIRMetavar>(arg);
            meta_vars.push_back(var->name);
        }
    }
    return meta_vars;
}

std::vector<std::string> MIRSingleHeadClause::get_all_rel_names() {
    std::vector<std::string> rel_names;
    rel_names.push_back(rel_name);
    return rel_names;
}

std::vector<std::string> MIRConjHeadClause::get_all_meta_vars() {
    std::vector<std::string> meta_vars;
    for (auto &head : heads) {
        if (head->type == MIRNodeType::SINGLE_HEAD) {
            auto single_head =
                std::dynamic_pointer_cast<MIRSingleHeadClause>(head);
            auto vars = single_head->get_all_meta_vars();
            for (auto var : vars) {
                if (std::find(meta_vars.begin(), meta_vars.end(), var) ==
                    meta_vars.end()) {
                    meta_vars.push_back(var);
                }
            }
        } else if (head->type == MIRNodeType::CONJ_HEAD) {
            auto conj_head = std::dynamic_pointer_cast<MIRConjHeadClause>(head);
            auto vars = conj_head->get_all_meta_vars();
            for (auto var : vars) {
                if (std::find(meta_vars.begin(), meta_vars.end(), var) ==
                    meta_vars.end()) {
                    meta_vars.push_back(var);
                }
            }
        }
    }
    return meta_vars;
}

std::vector<std::string> MIRConjHeadClause::get_all_rel_names() {
    std::vector<std::string> rel_names;
    for (auto head : heads) {
        if (head->type == MIRNodeType::SINGLE_HEAD) {
            auto single_head =
                std::dynamic_pointer_cast<MIRSingleHeadClause>(head);
            auto names = single_head->get_all_rel_names();
            for (auto name : names) {
                if (std::find(rel_names.begin(), rel_names.end(), name) ==
                    rel_names.end()) {
                    rel_names.push_back(name);
                }
            }
        } else if (head->type == MIRNodeType::CONJ_HEAD) {
            auto conj_head = std::dynamic_pointer_cast<MIRConjHeadClause>(head);
            auto names = conj_head->get_all_rel_names();
            for (auto name : names) {
                if (std::find(rel_names.begin(), rel_names.end(), name) ==
                    rel_names.end()) {
                    rel_names.push_back(name);
                }
            }
        }
    }
    return rel_names;
}

} // namespace vflog::mir

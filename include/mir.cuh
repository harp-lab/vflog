// middle level IR for datalog
// this is almost vanilla Datalog, but recursive rules are separated

#pragma once

#include "ram.cuh"
#include <cstdint>
#include <memory>
#include <vector>

namespace vflog {

namespace mir {

enum MIRNodeType {
    DECL,
    HEAD_CLAUSE,
    BODY_CLAUSE,
    RULE,
    METAVAR,
    STRING,
    NUMBER,
    CUSTOM,
    SCC,
    PROGRAM
};

struct MIRNode {
    MIRNodeType type;
    int id;
    // std::string name;
    int token_from;
    int token_to;
    std::vector<std::string> args;
    std::vector<std::shared_ptr<MIRNode>> children;

    MIRNode(){};

    virtual ~MIRNode(){};
};

struct MIRProgram : public MIRNode {
    std::vector<std::shared_ptr<MIRNode>> nodes;

    MIRProgram(std::vector<std::shared_ptr<MIRNode>> nodes) {
        this->type = PROGRAM;
        this->nodes = nodes;
        for (auto &node : nodes) {
            this->children.push_back(node);
        }
    }

    std::vector<std::shared_ptr<MIRNode>> lines() { return nodes; }
};

struct MIRScc : public MIRNode {
    std::vector<std::shared_ptr<MIRNode>> nodes;
    bool recursive;

    MIRScc(std::vector<std::shared_ptr<MIRNode>> nodes, bool recursive) {
        this->type = SCC;
        this->nodes = nodes;
        this->recursive = recursive;
        for (auto &node : nodes) {
            this->children.push_back(node);
        }
    }

    std::vector<std::shared_ptr<MIRNode>> lines() { return nodes; }
};

struct MIRDecl : public MIRNode {
    std::string rel_name;
    std::vector<std::string> arg_names;
    int arity;
    int max_split = 0;
    uint32_t split_size = UINT32_MAX;
    char *data_path = nullptr;

    MIRDecl(std::string rel_name, std::vector<std::string> arg_names,
            int max_split = 0, uint32_t split_size = UINT32_MAX,
            char *data_path = nullptr) {
        this->type = DECL;
        this->rel_name = rel_name;
        this->arg_names = arg_names;
        this->arity = arg_names.size();
        this->max_split = max_split;
        this->split_size = split_size;
        this->data_path = data_path;
    }
};

struct MIRBodyClause : public MIRNode {
    std::string rel_name;
    std::vector<std::shared_ptr<MIRNode>> args;
    bool negated;

    MIRBodyClause(std::string rel_name,
                  std::vector<std::shared_ptr<MIRNode>> args,
                  bool negated = false) {
        this->type = BODY_CLAUSE;
        this->rel_name = rel_name;
        this->args = args;
        this->negated = negated;
        for (auto &node : args) {
            this->children.push_back(node);
        }
    }
};

struct MIRRule : public MIRNode {
    std::shared_ptr<MIRNode> head;
    std::vector<std::shared_ptr<MIRNode>> body;

    MIRRule(std::shared_ptr<MIRNode> head,
            std::vector<std::shared_ptr<MIRNode>> body) {
        this->type = RULE;
        this->head = head;
        this->body = body;
        this->children.push_back(head);
        for (auto &node : body) {
            this->children.push_back(node);
        }
    }

    std::shared_ptr<MIRNode> get_head() { return head; }
    std::vector<std::shared_ptr<MIRNode>> get_body() { return body; }
};



struct MIRHeadClause : public MIRNode {
    std::string rel_name;
    std::vector<std::shared_ptr<MIRNode>> args;

    MIRHeadClause(std::string rel_name,
                  std::vector<std::shared_ptr<MIRNode>> args) {
        this->type = HEAD_CLAUSE;
        this->rel_name = rel_name;
        this->args = args;
        for (auto &node : args) {
            this->children.push_back(node);
        }
    }
};

struct MIRMetavar : public MIRNode {
    std::string name;

    MIRMetavar(std::string name) {
        this->type = METAVAR;
        this->name = name;
    }
};

struct MIRString : public MIRNode {
    std::string value;

    MIRString(std::string value) {
        this->type = STRING;
        this->value = value;
    }
};

struct MIRNumber : public MIRNode {
    int value;

    MIRNumber(int value) {
        this->type = NUMBER;
        this->value = value;
    }
};

struct MIRCustomNode : public MIRNode {

    std::string related_rel_name;
    std::function<void(ram::RelationalAlgebraMachine &)> func;

    MIRCustomNode(std::function<void(ram::RelationalAlgebraMachine &)> func,
                  std::string related_rel_name = "") {
        this->type = CUSTOM;
        this->func = func;
        this->related_rel_name = related_rel_name;
    }
};

inline std::shared_ptr<MIRProgram>
program(std::vector<std::shared_ptr<MIRNode>> nodes) {
    return std::make_shared<MIRProgram>(nodes);
}

inline std::shared_ptr<MIRScc> scc(std::vector<std::shared_ptr<MIRNode>> nodes,
                                   bool recursive) {
    return std::make_shared<MIRScc>(nodes, recursive);
}

inline std::shared_ptr<MIRDecl> declare(std::string rel_name,
                                        std::vector<std::string> arg_names,
                                        char *data_path = nullptr,
                                        int max_split = 0,
                                        uint32_t split_size = UINT32_MAX) {
    return std::make_shared<MIRDecl>(rel_name, arg_names, max_split, split_size,
                                     data_path);
}

inline std::shared_ptr<MIRRule>
rule(std::shared_ptr<MIRNode> head,
     std::vector<std::shared_ptr<MIRNode>> body) {
    return std::make_shared<MIRRule>(head, body);
}

inline std::shared_ptr<MIRBodyClause>
body(std::string rel_name, std::vector<std::shared_ptr<MIRNode>> args,
     bool negated = false) {
    return std::make_shared<MIRBodyClause>(rel_name, args, negated);
}

inline std::shared_ptr<MIRHeadClause>
head(std::string rel_name, std::vector<std::shared_ptr<MIRNode>> args) {
    return std::make_shared<MIRHeadClause>(rel_name, args);
}

inline std::shared_ptr<MIRMetavar> var(std::string name) {
    return std::make_shared<MIRMetavar>(name);
}

inline std::shared_ptr<MIRString> string(std::string value) {
    return std::make_shared<MIRString>(value);
}

inline std::shared_ptr<MIRNumber> number(int value) {
    return std::make_shared<MIRNumber>(value);
}

inline std::shared_ptr<MIRCustomNode>
custom(std::function<void(ram::RelationalAlgebraMachine &)> func,
       std::string related_rel_name = "") {
    return std::make_shared<MIRCustomNode>(func, related_rel_name);
}

// compile MIR to RAM
struct RAMGenPass {

    std::shared_ptr<ram::RAMProgram> ram_instructions;

    std::shared_ptr<MIRProgram> mir_program;

    std::map<std::string, std::vector<int>> joined_rel_columns;

    std::vector<std::shared_ptr<ram::RAMInstruction>> io_instructions;

    void compile();

    RAMGenPass(std::shared_ptr<mir::MIRProgram> mir_program)
        : mir_program(mir_program) {
        ram_instructions = std::make_shared<ram::RAMProgram>();
    }

    void get_ram_instructions(std::shared_ptr<ram::RAMProgram> &instructions) {
        instructions = this->ram_instructions;
    }

    std::vector<std::shared_ptr<ram::RAMInstruction>> allocate_registers();

    std::vector<std::shared_ptr<ram::RAMInstruction>> declare_relations();

    std::shared_ptr<ram::RAMInstruction>
    compile_scc(std::shared_ptr<MIRScc> scc);

    std::vector<std::shared_ptr<ram::RAMInstruction>>
    compile_rule(std::shared_ptr<MIRRule> rule,
                 std::vector<std::string> updated_rel_names, bool is_recursive);

    std::vector<std::shared_ptr<ram::RAMInstruction>>
    compile_copy_rules(std::shared_ptr<MIRRule> rule, bool is_recursive);

    std::vector<std::shared_ptr<ram::RAMInstruction>>
    compile_join_rules(std::shared_ptr<MIRRule> rule, bool is_recursive);

    void add_join_column(std::string rel_name, int column);
};

} // namespace mir

} // namespace vflog

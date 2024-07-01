
#include "relation.cuh"

#include <iostream>
#include <vector>
#include <algorithm>

namespace fvlog {

enum class RAOperatorType {
    JOIN,
    INDEX,
    PROJECT,
    FILTER,
    NEGATE,
    AGGREGATE,
    ARITHMETIC,
};

struct RelationalAlgebraOperator {
    using ID_t = uint32_t;
    ID_t id;
    RAOperatorType type;
    bool debug_flag = false;

    virtual void execute(RelationalEnvironment &env) = 0;
};

struct ColumnRAInfo {
    relational_ptr relation;
    RelationVersion version;
    size_t column_idx;
};

/**
 * @brief Relational join operator for free join
 * Join operator takes k input relation column as input
 * It will be applied to the re_slices
 * After execution, the result will be updated in the rel_slices
 */
struct RelationalJoin : public RelationalAlgebraOperator {
    using ID_t = uint32_t;

    HOST_VECTOR<ColumnRAInfo> input_columns;
    relational_ptr output_relation;
    HOST_VECTOR<ColumnRAInfo> output_columns;

    RelationalJoin(ID_t id, HOST_VECTOR<ColumnRAInfo> ins, relational_ptr out,
                   HOST_VECTOR<ColumnRAInfo> column) {
        this->id = id;
        this->type = RAOperatorType::JOIN;
        this->input_columns = ins;
        this->output_relation = out;
        this->output_columns = column;
    }

    void execute(RelationalEnvironment &env) override;

    /**
     * @brief Get all join input relation names
     */
    HOST_VECTOR<std::string> get_input_relation_names() const {
        HOST_VECTOR<std::string> names;
        for (auto &col : input_columns) {
            names.push_back(col.relation->get_name());
        }
        return names;
    }

    /**
     * @brief Get all input relation name need in out
     */
    HOST_VECTOR<std::string> get_output_relation_names() const {
        HOST_VECTOR<std::string> names;
        for (auto &col : output_columns) {
            names.push_back(col.relation->get_name());
        }
        // remove duplicate
        std::sort(names.begin(), names.end());
        names.erase(std::unique(names.begin(), names.end()), names.end());
        return names;
    }

    /**
     * @brief check if a relation name is used in output
     */
    bool is_used_in_out(const std::string &name) const {
        for (auto &col : output_columns) {
            if (col.relation->get_name() == name) {
                return true;
            }
        }
        return false;
    }
};

/**
 * @brief Relational index operator
 * Index operator takes a relation, a version and a column index as input
 * Force rebuild the index for the column
 */
struct RelationalIndex : public RelationalAlgebraOperator {
    using ID_t = uint32_t;

    ColumnRAInfo column;

    RelationalIndex(ID_t id, ColumnRAInfo column) : column(column) {
        this->id = id;
        this->type = RAOperatorType::INDEX;
    }

    void execute(RelationalEnvironment &env) override;
};

/**
 * @brief Relational project operator
 * Project operator takes a relation, a version and a column index as input
 * Project the column to the result slice
 */
struct RelationalProject : public RelationalAlgebraOperator {
    using ID_t = uint32_t;

    ColumnRAInfo input_column;
    ColumnRAInfo output_column;

    RelationalProject(ID_t id, ColumnRAInfo in, ColumnRAInfo out) {
        this->id = id;
        this->type = RAOperatorType::PROJECT;
        this->input_column = in;
        this->output_column = out;
    }

    void execute(RelationalEnvironment &env) override;
};

} // namespace fvlog

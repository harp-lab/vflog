
#include "relation.cuh"

#include <iostream>
#include <vector>

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

    RelationalJoin(ID_t id, std::vector<ColumnRAInfo> columns)
        : input_columns(columns) {
        this->id = id;
        this->type = RAOperatorType::JOIN;
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
};

} // namespace fvlog

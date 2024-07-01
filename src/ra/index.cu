#include "ra.cuh"

namespace fvlog {

void RelationalIndex::execute(RelationalEnvironment &env) {
    auto &rel = column.relation;
    auto &col = rel->get_column(column.version, column.column_idx);
    rel->get_hisa_data()->force_column_index(column.version, column.column_idx);
}

} // namespace fvlog

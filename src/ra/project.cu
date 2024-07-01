
#include "ra.cuh"

namespace fvlog {

void RelationalProject::execute(RelationalEnvironment &env) {

    auto &in_rel = input_column.relation;
    auto &out_rel = output_column.relation;

    auto &in_col =
        in_rel->get_column(input_column.version, input_column.column_idx);
    auto &out_col =
        out_rel->get_column(output_column.version, output_column.column_idx);
    auto in_tuple_size = in_rel->get_size(input_column.version);

    // find in environment
    if (env.slices.find(in_rel->get_name()) == env.slices.end()) {
        throw std::runtime_error("Cannot find input relation in environment");
    } else {
        auto &in_bitmap = env.slices[in_rel->get_name()].bitmap;
        // TODO: materialize base on all matched relation
        
    }
}

} // namespace fvlog

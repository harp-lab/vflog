
#include "relation.cuh"

namespace fvlog {
void relation::load_data(const std::vector<std::vector<int>> &data,
                         bool dup_flag) {
    // make sure before loading relation is empty
    assert(cpu_columns[0].full_size == 0);
    // load data to cpu first
    for (size_t i = 0; i < arity; i++) {
        cpu_columns[i].raw_data = data[i];
        cpu_columns[i].newt_size = data[i].size();
    }
    // load data to hisa
    for (size_t i = 0; i < arity; i++) {
        hisa_data->load_column_cpu(cpu_columns[i], i);
    }
    // deduplicate
    if (dup_flag) {
        hisa_data->deduplicate();
    }
    hisa_data->persist_newt();
}

void relation::allocate_newt(size_t size) { hisa_data->allocate_newt(size); }

device_data_ptr relation::get_newt_head(int i) {
    return hisa_data->data[i].data() + hisa_data->newt_columns[i].raw_offset;
}

} // namespace fvlog

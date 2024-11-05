#include "io.cuh"
#include "mir.cuh"
#include "ram_instruction.cuh"
#include "utils.cuh"
#include "vflog.cuh"

#include <iostream>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <thrust/sequence.h>

void cspa_ram(char *data_path) {
    std::string assign_path = std::string(data_path) + "/assign.facts";
    std::string dereference_path =
        std::string(data_path) + "/dereference.facts";

    auto ram = vflog::ram::RelationalAlgebraMachine();

    auto assign = ram.create_rel("assign", 2);
    ram.rels["assign"]->set_index_startegy(0, FULL, vflog::IndexStrategy::EAGER);
    ram.rels["assign"]->set_index_startegy(1, FULL, vflog::IndexStrategy::EAGER);
    auto dereference =
        ram.create_rel("dereference", 2, dereference_path.c_str());

    auto value_flow = ram.create_rel("value_flow", 2);
    ram.rels["value_flow"]->set_index_startegy(0, FULL, vflog::IndexStrategy::EAGER);
    ram.rels["value_flow"]->set_index_startegy(1, FULL, vflog::IndexStrategy::EAGER);
    auto memory_alias = ram.create_rel("memory_alias", 2);
    ram.rels["memory_alias"]->set_index_startegy(1, FULL, vflog::IndexStrategy::EAGER);
    ram.rels["memory_alias"]->set_index_startegy(1, FULL, vflog::IndexStrategy::EAGER);
    auto value_alias = ram.create_rel("value_alias", 2);
//     ram.rels["value_alias"]->set_index_startegy(0, FULL, vflog::IndexStrategy::EAGER);
//     ram.rels["value_alias"]->set_index_startegy(1, FULL, vflog::IndexStrategy::EAGER);

    ram.extend_register("r_a");
    ram.extend_register("r_b");
    ram.extend_register("r_c");

    using namespace vflog::ram;
    ram.add_operator({
        load_file(rel_t("assign"), assign_path.c_str(), 2),

        // ValueFlow(y, x) :- Assign(y, x).
        cache_update("c_assign", "r_a"),
        cache_init("c_assign", rel_t("assign"), FULL),
        prepare_materialization(rel_t("value_flow"), "c_assign"),
        project_op(vflog::column_t("assign", 0, FULL),
                   vflog::column_t("value_flow", 0, NEWT), "c_assign"),
        project_op(vflog::column_t("assign", 1, FULL),
                   vflog::column_t("value_flow", 1, NEWT), "c_assign"),
        end_materialization(rel_t("value_flow"), "c_assign"),
        // ValueFlow(x, x) :- Assign(x, y).
        prepare_materialization(rel_t("value_flow"), "c_assign"),
        project_op(vflog::column_t("assign", 0, FULL),
                   vflog::column_t("value_flow", 0, NEWT), "c_assign"),
        project_op(vflog::column_t("assign", 0, FULL),
                   vflog::column_t("value_flow", 1, NEWT), "c_assign"),
        end_materialization(rel_t("value_flow"), "c_assign"),
        // ValueFlow(x, x) :- Assign(y, x).
        prepare_materialization(rel_t("value_flow"), "c_assign"),
        project_op(vflog::column_t("assign", 1, FULL),
                   vflog::column_t("value_flow", 0, NEWT), "c_assign"),
        project_op(vflog::column_t("assign", 1, FULL),
                   vflog::column_t("value_flow", 1, NEWT), "c_assign"),
        end_materialization(rel_t("value_flow"), "c_assign"),

        // MemoryAlias(x, x) :- Assign(y, x).
        prepare_materialization(rel_t("memory_alias"), "c_assign"),
        project_op(vflog::column_t("assign", 1, FULL),
                   vflog::column_t("memory_alias", 0, NEWT), "c_assign"),
        project_op(vflog::column_t("assign", 1, FULL),
                   vflog::column_t("memory_alias", 1, NEWT), "c_assign"),
        end_materialization(rel_t("memory_alias"), "c_assign"),
        // MemoryAlias(x, x) :- Assign(x, y).
        prepare_materialization(rel_t("memory_alias"), "c_assign"),
        project_op(vflog::column_t("assign", 0, FULL),
                   vflog::column_t("memory_alias", 0, NEWT), "c_assign"),
        project_op(vflog::column_t("assign", 0, FULL),
                   vflog::column_t("memory_alias", 1, NEWT), "c_assign"),
        end_materialization(rel_t("memory_alias"), "c_assign"),        
        cache_clear(),
        persistent(rel_t("value_flow")),
        persistent(rel_t("memory_alias")),
        print_size(rel_t("value_flow")),
        print_size(rel_t("memory_alias")),

        fixpoint_op(
            {
                // ValueFlow(x, y) :- ValueFlow(x, z), ValueFlow(z, y).
                // ValueFlow(x, z) is delta, ValueFlow(z, y) is full.
                cache_update("c_value_flow1", "r_a"),
                cache_init("c_value_flow1", rel_t("value_flow"), DELTA),
                join_op(vflog::column_t("value_flow", 0, FULL),
                        vflog::column_t("value_flow", 1, DELTA),
                        "c_value_flow1", "r_b"),
                cache_update("c_value_flow2", "r_b"),
                prepare_materialization(rel_t("value_flow"), "c_value_flow2"),
                project_op(vflog::column_t("value_flow", 0, DELTA),
                           vflog::column_t("value_flow", 0, NEWT),
                           "c_value_flow1"),
                project_op(vflog::column_t("value_flow", 1, FULL),
                           vflog::column_t("value_flow", 1, NEWT),
                           "c_value_flow2"),
                end_materialization(rel_t("value_flow"), "c_value_flow2"),
                cache_clear(),
                // ValueFlow(x, z) is full, ValueFlow(z, y) is delta.
                cache_update("c_value_flow2", "r_a"),
                cache_init("c_value_flow2", rel_t("value_flow"), DELTA),
                join_op(vflog::column_t("value_flow", 1, FULL),
                        vflog::column_t("value_flow", 0, DELTA),
                        "c_value_flow2", "r_b"),
                cache_update("c_value_flow1", "r_b"),
                prepare_materialization(rel_t("value_flow"), "c_value_flow1"),
                project_op(vflog::column_t("value_flow", 0, FULL),
                           vflog::column_t("value_flow", 0, NEWT),
                           "c_value_flow1"),
                project_op(vflog::column_t("value_flow", 1, DELTA),
                           vflog::column_t("value_flow", 1, NEWT),
                           "c_value_flow2"),
                end_materialization(rel_t("value_flow"), "c_value_flow1"),
                cache_clear(),

                // join_va_vf_vf: ValueAlias(x, y) :-
                //                    ValueFlow(z, x),ValueFlow(z, y).
                // ValueFlow(z, x) is delta, ValueFlow(z, y) is full.
                cache_update("c_value_flow1", "r_a"),
                cache_init("c_value_flow1", rel_t("value_flow"), DELTA),
                join_op(vflog::column_t("value_flow", 0, FULL),
                        vflog::column_t("value_flow", 0, DELTA),
                        "c_value_flow1", "r_b"),
                cache_update("c_value_flow2", "r_b"),
                prepare_materialization(rel_t("value_alias"), "c_value_flow2"),
                project_op(vflog::column_t("value_flow", 1, DELTA),
                           vflog::column_t("value_alias", 0, NEWT),
                           "c_value_flow1"),
                project_op(vflog::column_t("value_flow", 1, FULL),
                           vflog::column_t("value_alias", 1, NEWT),
                           "c_value_flow2"),
                end_materialization(rel_t("value_alias"), "c_value_flow2"),
                
                prepare_materialization(rel_t("value_alias"), "c_value_flow2"),
                project_op(vflog::column_t("value_flow", 1, DELTA),
                           vflog::column_t("value_alias", 1, NEWT),
                           "c_value_flow1"),
                project_op(vflog::column_t("value_flow", 1, FULL),
                           vflog::column_t("value_alias", 0, NEWT),
                           "c_value_flow2"),
                end_materialization(rel_t("value_alias"), "c_value_flow2"),

                cache_clear(),

                // ValueFlow(x, y) :- Assign(x, z), MemoryAlias(z, y).
                cache_update("c_memory_alias", "r_a"),
                cache_init("c_memory_alias", rel_t("memory_alias"), DELTA),
                join_op(vflog::column_t("assign", 1, FULL),
                        vflog::column_t("memory_alias", 0, DELTA),
                        "c_memory_alias", "r_b"),
                cache_update("c_assign", "r_b"),
                prepare_materialization(rel_t("value_flow"), "c_assign"),
                project_op(vflog::column_t("assign", 0, FULL),
                           vflog::column_t("value_flow", 0, NEWT), "c_assign"),
                project_op(vflog::column_t("memory_alias", 1, DELTA),
                           vflog::column_t("value_flow", 1, NEWT),
                           "c_memory_alias"),
                end_materialization(rel_t("value_flow"), "c_assign"),
                cache_clear(),

                // MemoryAlias(x, w) :- Dereference(y, x),
                //                      ValueAlias(y, z), Dereference(z, w).
                cache_update("c_value_alias", "r_a"),
                cache_init("c_value_alias", rel_t("value_alias"), DELTA),
                join_op(vflog::column_t("dereference", 0, FULL),
                        vflog::column_t("value_alias", 0, DELTA),
                        "c_value_alias", "r_b"),
                cache_update("c_dereference1", "r_b"),
                join_op(vflog::column_t("dereference", 0, FULL),
                        vflog::column_t("value_alias", 1, DELTA),
                        "c_value_alias", "r_c"),
                cache_update("c_dereference2", "r_c"),
                prepare_materialization(rel_t("memory_alias"),
                                        "c_dereference2"),
                project_op(vflog::column_t("dereference", 1, FULL),
                           vflog::column_t("memory_alias", 0, NEWT),
                           "c_dereference1"),
                project_op(vflog::column_t("dereference", 1, FULL),
                           vflog::column_t("memory_alias", 1, NEWT),
                           "c_dereference2"),
                end_materialization(rel_t("memory_alias"), "c_dereference2"),
                cache_clear(),

                // ValueAlias(x, y) :- ValueFlow(z, x), MemoryAlias(z, w),
                //                     ValueFlow(w, y).
                // ValueFlow(z, x) is delta
                cache_update("c_value_flow1", "r_a"),
                cache_init("c_value_flow1", rel_t("value_flow"), DELTA),
                join_op(vflog::column_t("memory_alias", 0, FULL),
                        vflog::column_t("value_flow", 0, DELTA),
                        "c_value_flow1", "r_b"),
                cache_update("c_memory_alias", "r_b"),
                join_op(vflog::column_t("value_flow", 0, FULL),
                        vflog::column_t("memory_alias", 1, FULL),
                        "c_memory_alias", "r_c"),
                cache_update("c_value_flow2", "r_c"),
                prepare_materialization(rel_t("value_alias"), "c_value_flow2"),
                project_op(vflog::column_t("value_flow", 1, DELTA),
                           vflog::column_t("value_alias", 0, NEWT),
                           "c_value_flow1"),
                project_op(vflog::column_t("value_flow", 1, FULL),
                           vflog::column_t("value_alias", 1, NEWT),
                           "c_value_flow2"),
                end_materialization(rel_t("value_alias"), "c_value_flow2"),
                cache_clear(),
                // MemoryAlias(z, w) is delta
                cache_update("c_memory_alias", "r_a"),
                cache_init("c_memory_alias", rel_t("memory_alias"), DELTA),
                join_op(vflog::column_t("value_flow", 0, FULL),
                        vflog::column_t("memory_alias", 0, DELTA),
                        "c_memory_alias", "r_b"),
                cache_update("c_value_flow1", "r_b"),
                join_op(vflog::column_t("value_flow", 0, FULL),
                        vflog::column_t("memory_alias", 1, DELTA),
                        "c_memory_alias", "r_c"),
                cache_update("c_value_flow2", "r_c"),
                prepare_materialization(rel_t("value_alias"), "c_value_flow2"),
                project_op(vflog::column_t("value_flow", 1, FULL),
                           vflog::column_t("value_alias", 0, NEWT),
                           "c_value_flow1"),
                project_op(vflog::column_t("value_flow", 1, FULL),
                           vflog::column_t("value_alias", 1, NEWT),
                           "c_value_flow2"),
                end_materialization(rel_t("value_alias"), "c_value_flow2"),
                cache_clear(),
                // ValueFlow(w, y) is delta
                cache_update("c_value_flow2", "r_a"),
                cache_init("c_value_flow2", rel_t("value_flow"), DELTA),
                join_op(vflog::column_t("memory_alias", 1, FULL),
                        vflog::column_t("value_flow", 0, DELTA),
                        "c_value_flow2", "r_b"),
                cache_update("c_memory_alias", "r_b"),
                join_op(vflog::column_t("value_flow", 0, FULL),
                        vflog::column_t("memory_alias", 0, FULL),
                        "c_memory_alias", "r_c"),
                cache_update("c_value_flow1", "r_c"),
                prepare_materialization(rel_t("value_alias"), "c_value_flow1"),
                project_op(vflog::column_t("value_flow", 1, FULL),
                           vflog::column_t("value_alias", 0, NEWT),
                           "c_value_flow1"),
                project_op(vflog::column_t("value_flow", 1, DELTA),
                           vflog::column_t("value_alias", 1, NEWT),
                           "c_value_flow2"),
                end_materialization(rel_t("value_alias"), "c_value_flow1"),
                cache_clear(),

                persistent(rel_t("value_alias")),
                persistent(rel_t("memory_alias")),
                persistent(rel_t("value_flow")),
            },
            {rel_t("value_flow"), rel_t("memory_alias"), rel_t("value_alias"),}),
        print_size(rel_t("value_alias")),
        print_size(rel_t("memory_alias")),
        print_size(rel_t("value_flow")),
    });

    ram.execute();
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <data_path>" << std::endl;
        return 1;
    }

    ENABLE_RMM_POOL_MEMORY_RESOURCE

    char *data_path = argv[1];
    cspa_ram(data_path);
}


#include "ram.cuh"
#include "ram_instruction.cuh"
#include "utils.cuh"
#include "vflog.cuh"

#include <iostream>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <string>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>

void polonius_ram(char *data_path) {
    auto ram = vflog::ram::RelationalAlgebraMachine();

    std::string child_path_path = std::string(data_path) + "/child_path.facts";
    ram.create_rel("child_path", 2, child_path_path.c_str());
    ram.create_rel("ancestor_path", 2);
    std::string path_moved_at_base_path =
        std::string(data_path) + "/path_moved_at_base.facts";
    ram.create_rel("path_moved_at_base", 2, path_moved_at_base_path.c_str());
    ram.create_rel("path_moved_at", 2);
    std::string path_assigned_at_base_path =
        std::string(data_path) + "/path_assigned_at_base.facts";
    ram.create_rel("path_assigned_at_base", 2,
                   path_assigned_at_base_path.c_str());
    ram.create_rel("path_assigned_at", 2);
    std::string path_accessed_at_base_path =
        std::string(data_path) + "/path_accessed_at_base.facts";
    ram.create_rel("path_accessed_at_base", 2,
                   path_accessed_at_base_path.c_str());
    ram.create_rel("path_accessed_at", 2);
    std::string path_is_var_path =
        std::string(data_path) + "/path_is_var.facts";
    ram.create_rel("path_is_var", 2, path_is_var_path.c_str());
    ram.create_rel("path_begins_with_var", 2);
    ram.create_rel("path_maybe_initialized_on_exit", 2);
    ram.create_rel("path_maybe_uninitialized_on_exit", 2);
    std::string cfg_edge_path = std::string(data_path) + "/cfg_edge.facts";
    ram.create_rel("cfg_edge", 2, cfg_edge_path.c_str());
    ram.create_rel("var_maybe_partly_initialized_on_exit", 2);
    ram.get_rel("var_maybe_partly_initialized_on_exit")
        ->set_index_startegy(1, FULL, vflog::IndexStrategy::EAGER);
    ram.create_rel("cfg_node", 1);
    std::string universal_region_path =
        std::string(data_path) + "/universal_region.facts";
    ram.create_rel("universal_region", 1, universal_region_path.c_str());
    ram.create_rel("origin_live_on_entry", 2);
    ram.create_rel("var_live_on_entry", 2);
    std::string var_used_at_path =
        std::string(data_path) + "/var_used_at.facts";
    // ram.create_rel("var_used_at", 2, var_used_at_path.c_str());
    ram.create_rel("var_used_at", 2);
    ram.create_rel("var_maybe_partly_initialized_on_entry", 2);

    ram.extend_register("r_1");
    ram.extend_register("r_2");
    ram.extend_register("r_3");

    thrust::device_vector<bool> containes_tmp;
    using namespace vflog::ram;
    ram.add_operator({
        custom_op([](RelationalAlgebraMachine &ram_) {
            std::cout << "ancestor_path(x, y) :- child_path(x, y)."
                      << std::endl;
        }),
        cache_update("c_child_path", "r_1"),
        cache_init("c_child_path", rel_t("child_path"), FULL),
        prepare_materialization(rel_t("ancestor_path"), "c_child_path"),
        project_op(vflog::column_t("child_path", 0, FULL),
                   vflog::column_t("ancestor_path", 0, NEWT), "c_child_path"),
        project_op(vflog::column_t("child_path", 1, FULL),
                   vflog::column_t("ancestor_path", 1, NEWT), "c_child_path"),
        end_materialization(rel_t("ancestor_path"), "c_child_path"),
        cache_clear(),
        custom_op([](RelationalAlgebraMachine &ram_) {
            std::cout << "path_moved_at(x, y) :- path_moved_at_base(x, y)."
                      << std::endl;
        }),
        cache_update("c_path_moved_at_base", "r_1"),
        cache_init("c_path_moved_at_base", rel_t("path_moved_at_base"), FULL),
        prepare_materialization(rel_t("path_moved_at"), "c_path_moved_at_base"),
        project_op(vflog::column_t("path_moved_at_base", 0, FULL),
                   vflog::column_t("path_moved_at", 0, NEWT),
                   "c_path_moved_at_base"),
        project_op(vflog::column_t("path_moved_at_base", 1, FULL),
                   vflog::column_t("path_moved_at", 1, NEWT),
                   "c_path_moved_at_base"),
        end_materialization(rel_t("path_moved_at"), "c_path_moved_at_base"),
        cache_clear(),
        custom_op([](RelationalAlgebraMachine &ram_) {
            std::cout
                << "path_assigned_at(x, y) :- path_assigned_at_base(x, y)."
                << std::endl;
        }),
        cache_update("c_path_assigned_at_base", "r_1"),
        cache_init("c_path_assigned_at_base", rel_t("path_assigned_at_base"),
                   FULL),
        prepare_materialization(rel_t("path_assigned_at"),
                                "c_path_assigned_at_base"),
        project_op(vflog::column_t("path_assigned_at_base", 0, FULL),
                   vflog::column_t("path_assigned_at", 0, NEWT),
                   "c_path_assigned_at_base"),
        project_op(vflog::column_t("path_assigned_at_base", 1, FULL),
                   vflog::column_t("path_assigned_at", 1, NEWT),
                   "c_path_assigned_at_base"),
        end_materialization(rel_t("path_assigned_at"),
                            "c_path_assigned_at_base"),
        cache_clear(),
        custom_op([](RelationalAlgebraMachine &ram_) {
            std::cout
                << "path_accessed_at(x, y) :- path_accessed_at_base(x, y)."
                << std::endl;
        }),
        cache_update("c_path_accessed_at_base", "r_1"),
        cache_init("c_path_accessed_at_base", rel_t("path_accessed_at_base"),
                   FULL),
        prepare_materialization(rel_t("path_accessed_at"),
                                "c_path_accessed_at_base"),
        project_op(vflog::column_t("path_accessed_at_base", 0, FULL),
                   vflog::column_t("path_accessed_at", 0, NEWT),
                   "c_path_accessed_at_base"),
        project_op(vflog::column_t("path_accessed_at_base", 1, FULL),
                   vflog::column_t("path_accessed_at", 1, NEWT),
                   "c_path_accessed_at_base"),
        end_materialization(rel_t("path_accessed_at"),
                            "c_path_accessed_at_base"),
        cache_clear(),
        custom_op([](RelationalAlgebraMachine &ram_) {
            std::cout << "path_begins_with_var(x, y) :- path_is_var(x, y)."
                      << std::endl;
        }),
        cache_update("c_path_is_var", "r_1"),
        cache_init("c_path_is_var", rel_t("path_is_var"), FULL),
        prepare_materialization(rel_t("path_begins_with_var"), "c_path_is_var"),
        project_op(vflog::column_t("path_is_var", 0, FULL),
                   vflog::column_t("path_begins_with_var", 0, NEWT),
                   "c_path_is_var"),
        project_op(vflog::column_t("path_is_var", 0, FULL),
                   vflog::column_t("path_begins_with_var", 1, NEWT),
                   "c_path_is_var"),
        end_materialization(rel_t("path_begins_with_var"), "c_path_is_var"),
        cache_clear(),
        persistent(rel_t("ancestor_path")),
        persistent(rel_t("path_moved_at")),
        persistent(rel_t("path_assigned_at")),
        persistent(rel_t("path_accessed_at")),
        persistent(rel_t("path_begins_with_var")),

        print_size(rel_t("ancestor_path")),
        print_size(rel_t("path_moved_at")),
        print_size(rel_t("path_assigned_at")),
        print_size(rel_t("path_accessed_at")),
        print_size(rel_t("path_begins_with_var")),

        custom_op([](RelationalAlgebraMachine &ram_) {
            std::cout << "ancestor_path(Grandparent, Child) :-\n "
                         "    ancestor_path(Parent, Child),\n "
                         "    child_path(Parent, Grandparent)."
                      << std::endl;
        }),
        fixpoint_op(
            {
                cache_update("c_ancestor_path", "r_1"),
                cache_init("c_ancestor_path", rel_t("ancestor_path"), DELTA),
                join_op(vflog::column_t("child_path", 0, FULL),
                        vflog::column_t("ancestor_path", 0, DELTA),
                        "c_ancestor_path", "r_2"),
                cache_update("c_child_path", "r_2"),
                prepare_materialization(rel_t("ancestor_path"),
                                        "c_ancestor_path"),
                project_op(vflog::column_t("ancestor_path", 1, DELTA),
                           vflog::column_t("ancestor_path", 1, NEWT),
                           "c_ancestor_path"),
                project_op(vflog::column_t("child_path", 1, FULL),
                           vflog::column_t("ancestor_path", 0, NEWT),
                           "c_child_path"),
                end_materialization(rel_t("ancestor_path"), "c_ancestor_path"),
                cache_clear(),
                persistent(rel_t("ancestor_path")),
            },
            {rel_t("ancestor_path")}),
        print_size(rel_t("ancestor_path")),

        custom_op([](RelationalAlgebraMachine &ram_) {
            std::cout << "path_moved_at(Child, Point) :-\n "
                         "    path_moved_at(Parent, Point),\n "
                         "    ancestor_path(Parent, Child)."
                      << std::endl;
        }),

        fixpoint_op(
            {
                cache_update("c_path_moved_at", "r_1"),
                cache_init("c_path_moved_at", rel_t("path_moved_at"), DELTA),
                join_op(vflog::column_t("ancestor_path", 0, FULL),
                        vflog::column_t("path_moved_at", 0, DELTA),
                        "c_path_moved_at", "r_2"),
                cache_update("c_ancestor_path", "r_2"),
                prepare_materialization(rel_t("path_moved_at"),
                                        "c_ancestor_path"),
                project_op(vflog::column_t("ancestor_path", 1, FULL),
                           vflog::column_t("path_moved_at", 0, NEWT),
                           "c_ancestor_path"),
                project_op(vflog::column_t("path_moved_at", 1, DELTA),
                           vflog::column_t("path_moved_at", 1, NEWT),
                           "c_path_moved_at"),
                end_materialization(rel_t("path_moved_at"), "c_ancestor_path"),
                cache_clear(),
                persistent(rel_t("path_moved_at")),
            },
            {rel_t("path_moved_at")}),
        print_size(rel_t("path_moved_at")),

        custom_op([](RelationalAlgebraMachine &ram_) {
            std::cout << "path_assigned_at(Child, Point) :-\n "
                         "    path_assigned_at(Parent, Point),\n "
                         "    ancestor_path(Parent, Child)."
                      << std::endl;
        }),
        fixpoint_op(
            {
                cache_update("c_path_assigned_at", "r_1"),
                cache_init("c_path_assigned_at", rel_t("path_assigned_at"),
                           DELTA),
                join_op(vflog::column_t("ancestor_path", 0, FULL),
                        vflog::column_t("path_assigned_at", 0, DELTA),
                        "c_path_assigned_at", "r_2"),
                cache_update("c_ancestor_path", "r_2"),
                prepare_materialization(rel_t("path_assigned_at"),
                                        "c_ancestor_path"),
                project_op(vflog::column_t("ancestor_path", 1, FULL),
                           vflog::column_t("path_assigned_at", 0, NEWT),
                           "c_ancestor_path"),
                project_op(vflog::column_t("path_assigned_at", 1, DELTA),
                           vflog::column_t("path_assigned_at", 1, NEWT),
                           "c_path_assigned_at"),
                end_materialization(rel_t("path_assigned_at"),
                                    "c_ancestor_path"),
                cache_clear(),
                persistent(rel_t("path_assigned_at")),
            },
            {rel_t("path_assigned_at")}),
        print_size(rel_t("path_assigned_at")),

        custom_op([](RelationalAlgebraMachine &ram_) {
            std::cout << "path_accessed_at(Child, Point) :-\n "
                         "    path_accessed_at(Parent, Point),\n "
                         "    ancestor_path(Parent, Child)."
                      << std::endl;
        }),
        fixpoint_op(
            {
                cache_update("c_path_accessed_at", "r_1"),
                cache_init("c_path_accessed_at", rel_t("path_accessed_at"),
                           DELTA),
                join_op(vflog::column_t("ancestor_path", 0, FULL),
                        vflog::column_t("path_accessed_at", 0, DELTA),
                        "c_path_accessed_at", "r_2"),
                cache_update("c_ancestor_path", "r_2"),
                prepare_materialization(rel_t("path_accessed_at"),
                                        "c_ancestor_path"),
                project_op(vflog::column_t("ancestor_path", 1, FULL),
                           vflog::column_t("path_accessed_at", 0, NEWT),
                           "c_ancestor_path"),
                project_op(vflog::column_t("path_accessed_at", 1, DELTA),
                           vflog::column_t("path_accessed_at", 1, NEWT),
                           "c_path_accessed_at"),
                end_materialization(rel_t("path_accessed_at"),
                                    "c_ancestor_path"),
                cache_clear(),
                persistent(rel_t("path_accessed_at")),
            },
            {rel_t("path_accessed_at")}),
        print_size(rel_t("path_accessed_at")),

        custom_op([](RelationalAlgebraMachine &ram_) {
            std::cout << "path_begins_with_var(Child, Var) :-\n "
                         "    path_begins_with_var(Parent, Var),\n "
                         "    ancestor_path(Parent, Child)."
                      << std::endl;
        }),
        fixpoint_op(
            {
                cache_update("c_path_begins_with_var", "r_1"),
                cache_init("c_path_begins_with_var",
                           rel_t("path_begins_with_var"), DELTA),
                join_op(vflog::column_t("ancestor_path", 0, FULL),
                        vflog::column_t("path_begins_with_var", 0, DELTA),
                        "c_path_begins_with_var", "r_2"),
                cache_update("c_ancestor_path", "r_2"),
                prepare_materialization(rel_t("path_begins_with_var"),
                                        "c_ancestor_path"),
                project_op(vflog::column_t("ancestor_path", 1, FULL),
                           vflog::column_t("path_begins_with_var", 0, NEWT),
                           "c_ancestor_path"),
                project_op(vflog::column_t("path_begins_with_var", 1, DELTA),
                           vflog::column_t("path_begins_with_var", 1, NEWT),
                           "c_path_begins_with_var"),
                end_materialization(rel_t("path_begins_with_var"),
                                    "c_ancestor_path"),
                cache_clear(),
                persistent(rel_t("path_begins_with_var")),
            },
            {rel_t("path_begins_with_var")}),
        print_size(rel_t("path_begins_with_var")),

        // Step 2: Compute path initialization and deinitialization across
        // the CFG.
        custom_op([](RelationalAlgebraMachine &ram_) {
            std::cout << "path_maybe_initialized_on_exit(path, point) :-\n "
                         "    path_assigned_at(path, point)."
                      << std::endl;
        }),
        cache_update("c_path_assigned_at", "r_1"),
        cache_init("c_path_assigned_at", rel_t("path_assigned_at"), FULL),
        prepare_materialization(rel_t("path_maybe_initialized_on_exit"),
                                "c_path_assigned_at"),
        project_op(vflog::column_t("path_assigned_at", 0, FULL),
                   vflog::column_t("path_maybe_initialized_on_exit", 0, NEWT),
                   "c_path_assigned_at"),
        project_op(vflog::column_t("path_assigned_at", 1, FULL),
                   vflog::column_t("path_maybe_initialized_on_exit", 1, NEWT),
                   "c_path_assigned_at"),
        end_materialization(rel_t("path_maybe_initialized_on_exit"),
                            "c_path_assigned_at"),
        cache_clear(),
        persistent(rel_t("path_maybe_initialized_on_exit")),
        print_size(rel_t("path_maybe_initialized_on_exit")),

        custom_op([](RelationalAlgebraMachine &ram_) {
            std::cout << "path_maybe_uninitialized_on_exit(path, point) :-\n "
                         "    path_moved_at(path, point)."
                      << std::endl;
        }),
        cache_update("c_path_moved_at", "r_1"),
        cache_init("c_path_moved_at", rel_t("path_moved_at"), FULL),
        prepare_materialization(rel_t("path_maybe_uninitialized_on_exit"),
                                "c_path_moved_at"),
        project_op(vflog::column_t("path_moved_at", 0, FULL),
                   vflog::column_t("path_maybe_uninitialized_on_exit", 0, NEWT),
                   "c_path_moved_at"),
        project_op(vflog::column_t("path_moved_at", 1, FULL),
                   vflog::column_t("path_maybe_uninitialized_on_exit", 1, NEWT),
                   "c_path_moved_at"),
        end_materialization(rel_t("path_maybe_uninitialized_on_exit"),
                            "c_path_moved_at"),
        cache_clear(),
        persistent(rel_t("path_maybe_uninitialized_on_exit")),
        print_size(rel_t("path_maybe_uninitialized_on_exit")),

        custom_op([](RelationalAlgebraMachine &ram_) {
            std::cout
                << "path_maybe_uninitialized_on_exit(path, point2) :-\n "
                   "    path_maybe_uninitialized_on_exit(path, point1),\n "
                   "    cfg_edge(point1, point2),\n "
                   "    !path_assigned_at(path, point2)."
                << std::endl;
        }),
        fixpoint_op(
            {
                cache_update("c_path_maybe_uninitialized_on_exit", "r_1"),
                cache_init("c_path_maybe_uninitialized_on_exit",
                           rel_t("path_maybe_uninitialized_on_exit"), DELTA),
                join_op(vflog::column_t("cfg_edge", 0, FULL),
                        vflog::column_t("path_maybe_uninitialized_on_exit", 1,
                                        DELTA),
                        "c_path_maybe_uninitialized_on_exit", "r_2"),
                cache_update("c_cfg_edge", "r_2"),
                negate_multi(
                    rel_t("path_assigned_at"),
                    {vflog::column_t("path_maybe_uninitialized_on_exit", 0,
                                     DELTA),
                     vflog::column_t("cfg_edge", 1, FULL)},
                    {"c_path_maybe_uninitialized_on_exit", "c_cfg_edge"}),
                prepare_materialization(
                    rel_t("path_maybe_uninitialized_on_exit"),
                    "c_path_maybe_uninitialized_on_exit"),
                project_op(vflog::column_t("path_maybe_uninitialized_on_exit",
                                           0, DELTA),
                           vflog::column_t("path_maybe_uninitialized_on_exit",
                                           0, NEWT),
                           "c_path_maybe_uninitialized_on_exit"),
                project_op(vflog::column_t("cfg_edge", 1, FULL),
                           vflog::column_t("path_maybe_uninitialized_on_exit",
                                           1, NEWT),
                           "c_cfg_edge"),
                end_materialization(rel_t("path_maybe_uninitialized_on_exit"),
                                    "c_path_maybe_uninitialized_on_exit"),
                cache_clear(),
                persistent(rel_t("path_maybe_uninitialized_on_exit")),
                // print_size(rel_t("path_maybe_uninitialized_on_exit")),
            },
            {rel_t("path_maybe_uninitialized_on_exit")}),

        custom_op([](RelationalAlgebraMachine &ram_) {
            std::cout << "path_maybe_initialized_on_exit(path, point2) :-\n "
                         "    path_maybe_initialized_on_exit(path, point1),\n "
                         "    cfg_edge(point1, point2),\n "
                         "    !path_moved_at(path, point2).\n"
                      << std::endl;
        }),
        fixpoint_op(
            {
                cache_update("c_path_maybe_initialized_on_exit", "r_1"),
                cache_init("c_path_maybe_initialized_on_exit",
                           rel_t("path_maybe_initialized_on_exit"), DELTA),
                join_op(
                    vflog::column_t("cfg_edge", 0, FULL),
                    vflog::column_t("path_maybe_initialized_on_exit", 1, DELTA),
                    "c_path_maybe_initialized_on_exit", "r_2"),
                cache_update("c_cfg_edge", "r_2"),
                negate_multi(
                    rel_t("path_moved_at"),
                    {vflog::column_t("path_maybe_initialized_on_exit", 0,
                                     DELTA),
                     vflog::column_t("cfg_edge", 1, FULL)},
                    {"c_path_maybe_initialized_on_exit", "c_cfg_edge"}),
                prepare_materialization(rel_t("path_maybe_initialized_on_exit"),
                                        "c_path_maybe_initialized_on_exit"),
                project_op(
                    vflog::column_t("path_maybe_initialized_on_exit", 0, DELTA),
                    vflog::column_t("path_maybe_initialized_on_exit", 0, NEWT),
                    "c_path_maybe_initialized_on_exit"),
                project_op(
                    vflog::column_t("cfg_edge", 1, FULL),
                    vflog::column_t("path_maybe_initialized_on_exit", 1, NEWT),
                    "c_cfg_edge"),
                end_materialization(rel_t("path_maybe_initialized_on_exit"),
                                    "c_path_maybe_initialized_on_exit"),
                cache_clear(),
                persistent(rel_t("path_maybe_initialized_on_exit")),
            },
            {rel_t("path_maybe_initialized_on_exit")}),
        print_size(rel_t("path_maybe_initialized_on_exit")),
        custom_op([](RelationalAlgebraMachine &ram_) {
            std::cout
                << "var_maybe_partly_initialized_on_exit(var, point) :-\n "
                   "    path_maybe_initialized_on_exit(path, point),\n "
                   "    path_begins_with_var(path, var)."
                << std::endl;
        }),
        cache_update("c_path_maybe_initialized_on_exit", "r_1"),
        cache_init("c_path_maybe_initialized_on_exit",
                   rel_t("path_maybe_initialized_on_exit"), FULL),
        join_op(vflog::column_t("path_begins_with_var", 0, FULL),
                vflog::column_t("path_maybe_initialized_on_exit", 0, FULL),
                "c_path_maybe_initialized_on_exit", "r_2"),
        cache_update("c_path_begins_with_var", "r_2"),
        prepare_materialization(rel_t("var_maybe_partly_initialized_on_exit"),
                                "c_path_begins_with_var"),
        project_op(
            vflog::column_t("path_begins_with_var", 1, FULL),
            vflog::column_t("var_maybe_partly_initialized_on_exit", 0, NEWT),
            "c_path_begins_with_var"),
        project_op(
            vflog::column_t("path_maybe_initialized_on_exit", 1, FULL),
            vflog::column_t("var_maybe_partly_initialized_on_exit", 1, NEWT),
            "c_path_maybe_initialized_on_exit"),
        end_materialization(rel_t("var_maybe_partly_initialized_on_exit"),
                            "c_path_begins_with_var"),
        cache_clear(),
        persistent(rel_t("var_maybe_partly_initialized_on_exit")),
        print_size(rel_t("var_maybe_partly_initialized_on_exit")),

        print_size(rel_t("path_maybe_uninitialized_on_exit")),
        print_size(rel_t("path_maybe_initialized_on_exit")),

        custom_op([](RelationalAlgebraMachine &ram_) {
            std::cout << "cfg_node(point1) :- cfg_edge(point1, _)."
                      << std::endl;
        }),
        cache_update("c_cfg_edge", "r_1"),
        cache_init("c_cfg_edge", rel_t("cfg_edge"), FULL),
        prepare_materialization(rel_t("cfg_node"), "c_cfg_edge"),
        project_op(vflog::column_t("cfg_edge", 0, FULL),
                   vflog::column_t("cfg_node", 0, NEWT), "c_cfg_edge"),
        end_materialization(rel_t("cfg_node"), "c_cfg_edge"),
        cache_clear(),
        custom_op([](RelationalAlgebraMachine &ram_) {
            std::cout << "cfg_node(point2) :- cfg_edge(_, point2)."
                      << std::endl;
        }),
        cache_update("c_cfg_edge", "r_1"),
        cache_init("c_cfg_edge", rel_t("cfg_edge"), FULL),
        prepare_materialization(rel_t("cfg_node"), "c_cfg_edge"),
        project_op(vflog::column_t("cfg_edge", 1, FULL),
                   vflog::column_t("cfg_node", 0, NEWT), "c_cfg_edge"),
        end_materialization(rel_t("cfg_node"), "c_cfg_edge"),
        cache_clear(),
        persistent(rel_t("cfg_node")),
        print_size(rel_t("cfg_node")),

        custom_op([](RelationalAlgebraMachine &ram_) {
            std::cout << "origin_live_on_entry(origin, point) :-\n "
                         "    cfg_node(point),\n "
                         "    universal_region(origin)."
                      << std::endl;
        }),
        print_size(rel_t("universal_region")),
        print_size(rel_t("cfg_node")),
        cartesian_op(rel_t("universal_region"), FULL, rel_t("cfg_node"), FULL,
                     "r_1", "r_2"),
        cache_update("c_universal_region", "r_1"),
        cache_update("c_cfg_node", "r_2"),
        prepare_materialization(rel_t("origin_live_on_entry"),
                                "c_universal_region"),
        project_op(vflog::column_t("universal_region", 0, FULL),
                   vflog::column_t("origin_live_on_entry", 0, NEWT),
                   "c_universal_region"),
        project_op(vflog::column_t("cfg_node", 0, FULL),
                   vflog::column_t("origin_live_on_entry", 1, NEWT),
                   "c_cfg_node"),
        end_materialization(rel_t("origin_live_on_entry"),
                            "c_universal_region"),
        cache_clear(),
        persistent(rel_t("origin_live_on_entry")),
        print_size(rel_t("origin_live_on_entry")),

        // custom_op([](RelationalAlgebraMachine &ram_) {
        //     std::cout << "var_live_on_entry(var, point) :-\n"
        //               << "    var_used_at(var, point)." << std::endl;
        // }),
        // cache_update("c_var_used_at", "r_1"),
        // cache_init("c_var_used_at", rel_t("var_used_at"), FULL),
        // prepare_materialization(rel_t("var_live_on_entry"), "c_var_used_at"),
        // project_op(vflog::column_t("var_used_at", 0, FULL),
        //            vflog::column_t("var_live_on_entry", 0, NEWT),
        //            "c_var_used_at"),
        // project_op(vflog::column_t("var_used_at", 1, FULL),
        //            vflog::column_t("var_live_on_entry", 1, NEWT),
        //            "c_var_used_at"),
        // end_materialization(rel_t("var_live_on_entry"), "c_var_used_at"),
        // cache_clear(),
        // persistent(rel_t("var_live_on_entry")),
        // print_size(rel_t("var_live_on_entry")),

        custom_op([](RelationalAlgebraMachine &ram_) {
            std::cout
                << "var_maybe_partly_initialized_on_entry(var, point2) :-\n"
                   "    var_maybe_partly_initialized_on_exit(var, point1),\n"
                   "    cfg_edge(point1, point2)."
                << std::endl;
        }),
        cache_update("c_var_maybe_partly_initialized_on_exit", "r_1"),
        cache_init("c_var_maybe_partly_initialized_on_exit",
                   rel_t("var_maybe_partly_initialized_on_exit"), FULL),
        join_op(
            vflog::column_t("cfg_edge", 0, FULL),
            vflog::column_t("var_maybe_partly_initialized_on_exit", 1, FULL),
            "c_var_maybe_partly_initialized_on_exit", "r_2"),
        cache_update("c_cfg_edge", "r_2"),
        prepare_materialization(rel_t("var_maybe_partly_initialized_on_entry"),
                                "c_var_maybe_partly_initialized_on_exit"),
        project_op(
            vflog::column_t("var_maybe_partly_initialized_on_exit", 0, FULL),
            vflog::column_t("var_maybe_partly_initialized_on_entry", 0, NEWT),
            "c_var_maybe_partly_initialized_on_exit"),
        project_op(
            vflog::column_t("cfg_edge", 1, FULL),
            vflog::column_t("var_maybe_partly_initialized_on_entry", 1, NEWT),
            "c_cfg_edge"),
        end_materialization(rel_t("var_maybe_partly_initialized_on_entry"),
                            "c_var_maybe_partly_initialized_on_exit"),
        cache_clear(),
        persistent(rel_t("var_maybe_partly_initialized_on_entry")),
        print_size(rel_t("var_maybe_partly_initialized_on_entry")),
    });

    KernelTimer timer;
    timer.start_timer();
    ram.execute();
    timer.stop_timer();
    std::cout << "Time: " << timer.get_spent_time() << std::endl;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <data_path>" << std::endl;
        return 1;
    }

    ENABLE_RMM_POOL_MEMORY_RESOURCE

    char *data_path = argv[1];
    polonius_ram(data_path);
}

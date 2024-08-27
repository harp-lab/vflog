
#include "mir.cuh"
#include "vflog.cuh"

#include <iostream>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <thrust/sequence.h>

void tc_ram(char *data_path, int splits_mode) {
    using namespace vflog::ram;
    auto ram = RelationalAlgebraMachine();

    auto edge = ram.create_rel("edge", 2, data_path);
    auto path = ram.create_rel("path", 2);
    if (splits_mode == 1) {
        path->split_when(327'680'000, 4);
    } else if (splits_mode == 2) {
        path->split_when(163'840'000, 4);
    } else if (splits_mode == 3) {
        path->split_when(327'680'000 * 2, 2);
    } else if (splits_mode > 10) {
        path->split_iter(splits_mode, 4);
    }

    auto input_indices_ptr = std::make_shared<vflog::device_indices_t>();
    auto tmp_id0 = std::make_shared<vflog::device_indices_t>();
    auto tmp_id1 = std::make_shared<vflog::device_indices_t>();
    auto tmp_id2 = std::make_shared<vflog::device_indices_t>();

    ram.add_operator({
        // path(a, b) :- edge(a, b).
        cache_update("edge", input_indices_ptr),
        cache_init("edge", rel_t("edge"), FULL),
        prepare_materialization(rel_t("path"), "edge"),
        project_op(column_t("edge", 0, FULL), column_t("path", 0, FULL),
                   "edge"),
        project_op(column_t("edge", 1, FULL), column_t("path", 1, FULL),
                   "edge"),
        end_materialization(rel_t("path"), "edge"),
        persistent(rel_t("path")),
        cache_clear(),
    });

    auto gen_join1 =
        [&](int frozen_idx) -> std::vector<std::shared_ptr<RAMInstruction>> {
        return {
            // path(a, c) :- path(a, b), edge(b, c).
            cache_update("path", tmp_id0),
            cache_init("path", rel_t("path"), DELTA),
            join_op(column_t("edge", 0, FULL, frozen_idx),
                    column_t("path", 1, DELTA), "path", tmp_id1),
            cache_update("edge", tmp_id1),
            prepare_materialization(rel_t("path"), "path"),
            project_op(column_t("path", 0, DELTA), column_t("path", 0, NEWT),
                       "path"),
            project_op(column_t("edge", 1, FULL, frozen_idx),
                       column_t("path", 1, NEWT), "edge"),
            end_materialization(rel_t("path"), "path"),
            cache_clear(),
        };
    };

    std::vector<std::shared_ptr<RAMInstruction>> fixpoint_instructions;
    auto splited_instr = gen_join1(-1);
    fixpoint_instructions.insert(fixpoint_instructions.end(),
                                 splited_instr.begin(), splited_instr.end());
    fixpoint_instructions.push_back(persistent(rel_t("path")));
    fixpoint_instructions.push_back(print_size(rel_t("path")));
    ram.add_operator({fixpoint_op(fixpoint_instructions, {rel_t("path")})});

    std::cout << "Start executing" << std::endl;
    KernelTimer timer;
    timer.start_timer();
    ram.execute();
    timer.stop_timer();
    auto elapsed = timer.get_spent_time();
    std::cout << "Elapsed time: " << elapsed << "s" << std::endl;
}

void tc_mir(char *data_path, int splits_mode) {
    using namespace vflog::mir;
    auto prog = program({
        declare("edge", {"a", "b"}, data_path),
        declare("path", {"a", "b"}),
        scc(
            {
                rule(head("path", {var("a"), var("b")}),
                     {body("edge", {var("a"), var("b")})}),
            },
            false),
        scc(
            {
                rule(head("path", {var("a"), var("c")}),
                     {body("path", {var("a"), var("b")}),
                      body("edge", {var("b"), var("c")})}),
            },
            true),
    });

    auto compiler = RAMGenPass(prog);
    compiler.compile();
    auto ram = compiler.ram_instructions;
    std::cout << "Compiled RAM program" << std::endl;
    std::cout << ram->to_string() << std::endl;
    
    vflog::ram::RelationalAlgebraMachine ram_machine;
    ram_machine.add_program(ram);

    std::cout << "Start executing" << std::endl;
    KernelTimer timer;
    timer.start_timer();
    ram_machine.execute();
    timer.stop_timer();
    auto elapsed = timer.get_spent_time();
    std::cout << "Elapsed time: " << elapsed << "s" << std::endl;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <data_path> <memory_system_flag> <split_mode>"
                  << std::endl;
        return 1;
    }

    // enable_rmm_allocator();
    rmm::mr::cuda_memory_resource cuda_mr{};
    // rmm::mr::set_current_device_resource(&cuda_mr);
    // first arg is data path
    char *data_path = argv[1];
    int memory_system_flag = atoi(argv[2]);
    if (memory_system_flag == 0) {
        rmm::mr::set_current_device_resource(&cuda_mr);
    } else if (memory_system_flag == 1) {
        rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr{
            &cuda_mr, 4 * 256 * 1024};
        rmm::mr::set_current_device_resource(&mr);
    } else if (memory_system_flag == 2) {
        rmm::mr::managed_memory_resource mr{};
        rmm::mr::set_current_device_resource(&mr);
    } else {
        rmm::mr::set_current_device_resource(&cuda_mr);
    }
    int splits_mode = atoi(argv[3]);

    // tc_ram(data_path, splits_mode);
    tc_mir(data_path, splits_mode);
    return 0;
}

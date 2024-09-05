
#include "vflog.cuh"

#include <iostream>
#include <memory>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

void sg_ram(char *data_path) {
    using namespace vflog::ram;
    auto ram = RelationalAlgebraMachine();

    auto edge = ram.create_rel("edge", 2, data_path);
    auto sg = ram.create_rel("sg", 2);

    ram.extend_register("r_a");
    ram.extend_register("r_b");
    ram.extend_register("r_x");
    ram.extend_register("r_y");

    ram.add_operator({
        // sg(x, y) :- edge(a, x), edge(a, y), x != y.
        cache_update("c_edge1", "r_x"),
        cache_init("c_edge1", rel_t("edge"), FULL),
        join_op(vflog::column_t("edge", 0, FULL), vflog::column_t("edge", 0, FULL), "c_edge1",
                "r_y"),
        cache_update("c_edge2", "r_y"),
        // filter match x y
        custom_op([](RelationalAlgebraMachine &ram) {
            vflog::device_bitmap_t filter_bitmap(
                ram.get_register("r_x")->size(), false);
            auto edge = ram.rels["edge"];
            thrust::transform(EXE_POLICY, ram.get_register("r_x")->begin(),
                              ram.get_register("r_x")->end(),
                              ram.get_register("r_y")->begin(),
                              filter_bitmap.begin(),
                              [x_ptrs = edge->get_raw_data_ptrs(FULL, 1),
                               y_ptrs = edge->get_raw_data_ptrs(
                                   FULL, 1)] LAMBDA_TAG(auto &x, auto &y) {
                                  return x_ptrs[x] != y_ptrs[y];
                              });
            auto matched_x_end = thrust::remove_if(
                EXE_POLICY, ram.get_register("r_x")->begin(),
                ram.get_register("r_x")->end(), filter_bitmap.begin(),
                thrust::logical_not<bool>());
            ram.get_register("r_x")->resize(matched_x_end -
                                            ram.get_register("r_x")->begin());
            auto matched_y_end = thrust::remove_if(
                EXE_POLICY, ram.get_register("r_y")->begin(),
                ram.get_register("r_y")->end(), filter_bitmap.begin(),
                thrust::logical_not<bool>());
            ram.get_register("r_y")->resize(matched_y_end -
                                            ram.get_register("r_y")->begin());
        }),
        // materialize sg
        prepare_materialization(rel_t("sg"), "c_edge2"),
        project_op(vflog::column_t("edge", 1, FULL), vflog::column_t("sg", 0, NEWT),
                   "c_edge1"),
        project_op(vflog::column_t("edge", 1, FULL), vflog::column_t("sg", 1, NEWT),
                   "c_edge2"),
        end_materialization(rel_t("sg"), "c_edge2"),
        persistent(rel_t("sg")),
        print_size(rel_t("sg")),
        cache_clear(),
        fixpoint_op(
            {
                // sg(x, y) :-  sg(a, b), edge(a, x), edge(b, y).
                cache_update("c_sg", "r_a"),
                cache_init("c_sg", rel_t("sg"), DELTA),
                join_op(vflog::column_t("edge", 0, FULL), vflog::column_t("sg", 0, DELTA),
                        "c_sg", "r_x"),
                cache_update("c_edge1", "r_x"),
                join_op(vflog::column_t("edge", 0, FULL), vflog::column_t("sg", 1, DELTA),
                        "c_sg", "r_y"),
                cache_update("c_edge2", "r_y"),
                prepare_materialization(rel_t("sg"), "c_edge2"),
                project_op(vflog::column_t("edge", 1, FULL), vflog::column_t("sg", 0, NEWT),
                           "c_edge1"),
                project_op(vflog::column_t("edge", 1, FULL), vflog::column_t("sg", 1, NEWT),
                           "c_edge2"),
                end_materialization(rel_t("sg"), "c_edge2"),
                persistent(rel_t("sg")),
                print_size(rel_t("sg")),
                cache_clear(),
            },
            {rel_t("sg")}),
    });

    KernelTimer timer;
    timer.start_timer();
    ram.execute();
    timer.stop_timer();
    auto elapsed = timer.get_spent_time();
    std::cout << "Elapsed time: " << elapsed << "s" << std::endl;
    ram.rels["sg"]->print_stats();
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <data_path> <memory_system_flag>"
                  << std::endl;
        return 1;
    }
    // 116931333
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

    sg_ram(data_path);
    return 0;
}



#include "utils.cuh"
#include "vflog.cuh"

#include <cstddef>
#include <cstdint>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <string>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

using namespace vflog::ram;
auto sg_ram_g = vflog::ram::RelationalAlgebraMachine();

rmm::mr::cuda_memory_resource cuda_mr{};
rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr{&cuda_mr,
                                                                4 * 256 * 1024};

auto sg_fixpoint = fixpoint_op(
    {
        // sg(x, y) :-  sg(a, b), edge(a, x), edge(b, y).
        cache_update("c_sg", "r_a"),
        cache_init("c_sg", rel_t("sg"), DELTA),
        join_op(vflog::column_t("edge", 0, FULL),
                vflog::column_t("sg", 0, DELTA), "c_sg", "r_x"),
        cache_update("c_edge1", "r_x"),
        join_op(vflog::column_t("edge", 0, FULL),
                vflog::column_t("sg", 1, DELTA), "c_sg", "r_y"),
        cache_update("c_edge2", "r_y"),
        prepare_materialization(rel_t("sg"), "c_edge2"),
        project_op(vflog::column_t("edge", 1, FULL),
                   vflog::column_t("sg", 0, NEWT), "c_edge1"),
        project_op(vflog::column_t("edge", 1, FULL),
                   vflog::column_t("sg", 1, NEWT), "c_edge2"),
        end_materialization(rel_t("sg"), "c_edge2"),
        persistent(rel_t("sg")),
        print_size(rel_t("sg")),
        cache_clear(),
    },
    {rel_t("sg")});

extern "C" {
void init_ram();
void ram_write_relation(char *name, uint32_t **host_data, size_t size);
void ram_read_relation(char *name, uint32_t **host_data, size_t *size,
                       size_t version);
}

void init_ram() {

    auto edge = sg_ram_g.create_rel("edge", 2);
    auto sg = sg_ram_g.create_rel("sg", 2);

    sg_ram_g.extend_register("r_a");
    sg_ram_g.extend_register("r_b");
    sg_ram_g.extend_register("r_x");
    sg_ram_g.extend_register("r_y");
    rmm::mr::set_current_device_resource(&mr);
}

void ram_write_relation(char *name, uint32_t **host_data, size_t size) {
    project_host_op(host_data, size, rel_t(std::string(name)))
        ->execute(sg_ram_g);
    persistent(rel_t(std::string(name)));
}

void ram_read_relation(char *name, uint32_t **host_data, size_t *size,
                       size_t version) {
    // transfer the delta of a relation to host
    auto ver = version == 0 ? RelationVersion::FULL : RelationVersion::DELTA;
    auto &rel = sg_ram_g.rels[std::string(name)];
    auto size_delta = rel->get_versioned_size(ver);
    *size = size_delta;
    for (int i = 0; i < rel->arity; i++) {
        auto raw_data = rel->get_raw_data_ptrs(ver, i);
        cudaMemcpy(host_data[i], raw_data, size_delta * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);
    }
}

void ram_iter_once(bool *updated_flag) {
    *updated_flag = sg_fixpoint->step(sg_ram_g);
}

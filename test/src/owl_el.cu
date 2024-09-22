
#include "ram.cuh"
#include "vflog.cuh"
#include <rmm/mr/device/managed_memory_resource.hpp>


void owl_el(char *data_path) {
    using namespace vflog::mir;

    std::string isMainClass_path = std::string(data_path) + "/isMainClass.facts";
    std::string isSubClass_path = std::string(data_path) + "/isSubClass.facts";
    std::string conj_path = std::string(data_path) + "/conj.facts";
    std::string exists_path = std::string(data_path) + "/exists.facts";
    std::string subClassOf_path = std::string(data_path) + "/subClassOf.facts";
    std::string subPropChain_path = std::string(data_path) + "/subPropChain.facts";
    std::string subProp_path = std::string(data_path) + "/subProp.facts";

    auto prog = program({
        // nf
        declare("isMainClass", {"C"}, isMainClass_path.c_str()),
        declare("isSubClass", {"C"}, isSubClass_path.c_str()),
        declare("conj", {"C", "D1", "D2"}, conj_path.c_str()),
        declare("exists", {"E", "R", "C"}, exists_path.c_str()),
        declare("subClassOf", {"C", "D"}, subClassOf_path.c_str()),
        declare("subPropChain", {"S1", "S2", "S"}, subPropChain_path.c_str()),
        declare("subProp", {"S1", "S2"}, subProp_path.c_str()),
        // inf
        declare("init", {"C"}),
        declare("subClassOf", {"C", "D"}),
        declare("ex", {"E", "R", "C"}),
        scc({
            rule(single_head("init", {var("C")}), {body("isMainClass", {var("C")})}),
        }, false),
        custom([](vflog::ram::RelationalAlgebraMachine &ram){
            print_size(vflog::ram::rel_t("init"));
        }),
        scc({
            rule(single_head("init", {var("C")}), {body("ex", {var("E"), var("R"), var("C")})}),
            rule(single_head("subClassOf",{var("C"), number(51430)}), {body("init", {var("C")})}),
        }, true),
        custom([](vflog::ram::RelationalAlgebraMachine &ram){
            print_size(vflog::ram::rel_t("subClassOf"));
        }),
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
    if (argc != 3) {
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

    owl_el(data_path);
    return 0;
}



#include "ram.cuh"
#include "vflog.cuh"
#include <rmm/mr/device/managed_memory_resource.hpp>
#include "scc.cuh" // Ensure the correct header file is included


void owl_el(char *data_path) {
    using namespace vflog::mir;

    // @export nf:isMainClass :- csv{format=(any), compression="gzip", resource="isMainClass.csv.gz"} .
    // @export nf:isSubClass :- csv{format=(any), compression="gzip", resource="isSubClass.csv.gz"} .
    // @export nf:conj :- csv{format=(any, any, any), compression="gzip", resource="conj.csv.gz"} .
    // @export nf:exists :- csv{format=(any, any, any), compression="gzip", resource="exists.csv.gz"} .
    // @export nf:subClassOf :- csv{format=(any, any), compression="gzip", resource="subClassOf.csv.gz"} .
    // @export nf:subPropChain :- csv{format=(any, any, any), compression="gzip", resource="subPropChain.csv.gz"} .
    // @export nf:subProp :- csv{format=(any, any), compression="gzip", resource="subProp.csv.gz"} .


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
                rule(head("init", {var("C")}), {body("isMainClass", {var("C")})}),
                    }, 
                    false),

        scc({
                rule(head("init", {var("C")}), {body("ex", {var("E"), var("R"), var("C")})}),
                    }, 
                    false),
        custom([](vflog::ram::RelationalAlgebraMachine &ram){
            print_size(vflog::ram::rel_t("init"));
        }),
        scc({
                rule(head("subClassOf", {var("C")}), 
                        {body("init", {var("C")})}),
                    }, 
                    true),
        scc({
                rule(head("subClassOf", {var("C"), var("<http://www.w3.org/2002/07/owl#Thing>")}), 
                        {body("isMainClass", {var("c")})}),
                    }, 
                    false),

        // inf:subClassOf(?C,?D1), inf:subClassOf(?C,?D2) :- inf:subClassOf(?C,?Y), nf:conj(?Y,?D1,?D2) .
        scc({
            rule(head("subClassOf", {var("C"), var("D1")}), 
                {body("subClassOf", {var("C"), var("Y")}), body("conj", {var("Y"), var("D1"), var("D2")})}),
            rule(head("subClassOf", {var("C"), var("D2")}),
                {body("subClassOf", {var("C"), var("E")}), body("subClassOf", {var("E"), var("D")})})
            }, true),
        //     inf:subClassOf(?C, ?I) :-
        // inf:subClassOf(?C, ?D1), inf:subClassOf(?C, ?D2),
        // nf:conj(?I, ?D1, ?D2), nf:isSubClass(?I) .
        scc({
            rule(head("subClassOf", {var("C"), var("I")}), 
                {body("subClassOf", {var("C"), var("D1")}), 
                            body("conj", {var("I"), var("D1"), var("D2")})}),
                            body("isSubClass", {var("I")})
            }, true),
        // inf:ex(?E, ?R, ?C) :- inf:subClassOf(?E, ?Y), nf:exists(?Y, ?R, ?C) .
        scc({
            rule(head("ex", {var("E"), var("R"), var("C")}), 
                {body("subClassOf", {var("E"), var("Y")}), body("exists", {var("Y"), var("R"), var("C")})})
            }, true),
        // inf:subClassOf(?E, ?Y) :-
        // inf:ex(?E, ?R, ?C), inf:subClassOf(?C, ?D), nf:subProp(?R, ?S), 
        // nf:exists(?Y, ?S, ?D), nf:isSubClass(?Y) .
        scc({
            rule(head("subClassOf", {var("E"), var("Y")}), 
                {body("ex", {var("E"), var("R"), var("C")}), 
                            body("subClassOf", {var("C"), var("D")}),
                            body("subProp", {var("R"), var("S")}),
                            body("exists", {var("Y"), var("S"), var("D")}),
                            body("isSubClass", {var("Y")})
                })
            }, true),
        
        // inf:ex(?E, ?S, ?D) :-
        // inf:ex(?E, ?R1, ?C), inf:ex(?C, ?R2, ?D),
        // nf:subProp(?R1, ?S1), nf:subProp(?R2, ?S2), 
        // nf:subPropChain(?S1, ?S2, ?S) .
        scc({
            rule(head("ex", {var("E"), var("S"), var("D")}), 
                {body("ex", {var("E"), var("R1"), var ("C")}), 
                            body("ex", {var("C"), var("R2"), var("D")}),
                            body("subProp", {var("R1"), var("S1")}),
                            body("subProp", {var("R2"), var("S2")}),
                            body("subPropChain", {var("S1"), var("S2"), var("S")})
                })
            }, true),
        // inf:subClassOf(?C,?E) :- inf:subClassOf(?C,?D), nf:subClassOf(?D,?E) .
        scc({
            rule(head("subClassOf", {var("C"), var("E")}), 
                {body("subClassOf", {var("C"), var("D")}), 
                            body("subClassOf", {var("D"), var("E")})
                })
            }, true),
        //     inf:subClassOf(?E, "<http://www.w3.org/2002/07/owl#Nothing>") :-
        // inf:ex(?E,?R,?C), inf:subClassOf(?C,"<http://www.w3.org/2002/07/owl#Nothing>") .
        // ???
        // mainSubClassOf(?A,?B) :-
        //     inf:subClassOf(?A,?B), nf:isMainClass(?A), nf:isMainClass(?B) .
        scc({
            rule(head("mainSubClassOf", {var("A"), var("B")}), 
                {body("subClassOf", {var("A"), var("B")}), 
                            body("isMainClass", {var("A")}),
                            body("isMainClass", {var("B")})
                })
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


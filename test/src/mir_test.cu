/**
 * test file for MIR
 */

#include <iostream>
#include <vector>

#include "vflog.cuh"

#define COMPILE_AND_ADD_PROGRAM(prog)                                          \
    auto compiler = RAMGenPass(prog);                                          \
    compiler.compile();                                                        \
    auto ram = compiler.ram_instructions;                                      \
    std::cout << "Compiled RAM program" << std::endl;                          \
    std::cout << ram->to_string() << std::endl;                                \
    auto ram_machine = vflog::ram::RelationalAlgebraMachine();                 \
    ram_machine.add_program(ram);                                              \
    ram_machine.execute();

using namespace vflog::mir;

bool test_fact() {
    auto prog = program({
        declare("foo", {"a", "b"}),
        scc(
            {
                fact("foo", {{1, 2}}),
            },
            false),
    });

    COMPILE_AND_ADD_PROGRAM(prog);
    // ram_machine.rels["foo"]->print_raw_data(FULL);
    std::vector<vflog::internal_data_type> t1 = {1, 2};
    return ram_machine.rels["foo"]->tuple_exists(t1, RelationVersion::FULL);
}

bool test_conj_head1() {
    auto proj = program({
        declare("foo", {"a", "b"}),
        declare("bar", {"b", "c"}),
        declare("foobar1", {"a", "c"}),
        declare("foobar2", {"a", "c"}),
        scc(
            {
                fact("foo", {{1, 2}}),
                fact("bar", {{2, 3}}),
                rule(conj_head({single_head("foobar1", {var("a"), var("c")}),
                                single_head("foobar2", {var("a"), var("c")})}),
                     {body("foo", {var("a"), var("b")}),
                      body("bar", {var("b"), var("c")})}),
            },
            false),
    });

    COMPILE_AND_ADD_PROGRAM(proj);
    // ram_machine.rels["foobar1"]->print_raw_data(FULL);
    // ram_machine.rels["foobar2"]->print_raw_data(FULL);
    std::vector<vflog::internal_data_type> t1 = {1, 3};
    return ram_machine.rels["foobar1"]->tuple_exists(t1,
                                                     RelationVersion::FULL) &&
           ram_machine.rels["foobar2"]->tuple_exists(t1, RelationVersion::FULL);
}

bool test_2_arity_join() {
    auto proj = program({
        declare("foo", {"a", "b"}),
        declare("bar", {"a", "b"}),
        declare("foobar", {"a", "b"}),
        scc(
            {
                fact("foo", {{1, 2}, {1, 5}, {5, 6}}),
                fact("bar", {{1, 2}}),
                rule(single_head("foobar", {var("a"), var("b")}),
                     {body("foo", {var("a"), var("b")}),
                      body("bar", {var("a"), var("b")})}),
            },
            false),
    });

    COMPILE_AND_ADD_PROGRAM(proj);
    // ram_machine.rels["foobar"]->print_raw_data(FULL);
    std::vector<vflog::internal_data_type> t1 = {1, 2};
    auto foobar_size = ram_machine.rels["foobar"]->full_size;
    return ram_machine.rels["foobar"]->tuple_exists(t1,
                                                    RelationVersion::FULL) &&
           foobar_size == 1;
}

bool check(std::string testname, bool res) {
    if (res) {
        std::cout << ">>>>>>>>>>>>>>>>>>>>> Test " << testname
                  << " \x1b[1;35;42m passed \033[0m" << std::endl;
    } else {
        std::cout << ">>>>>>>>>>>>>>>>>>>>> Test " << testname
                  << " \x1b[1;31;42m failed \033[0m" << std::endl;
    }
    return res;
}

int main(int argc, char **argv) {
    ENABLE_RMM_POOL_MEMORY_RESOURCE
    // test fact in MIR
    bool test_res = true;
    test_res = check("test_fact", test_fact()) && test_res;
    test_res = check("test_conj_head1", test_conj_head1()) && test_res;
    test_res = check("test_2_arity_join", test_2_arity_join()) && test_res;

    return test_res;
}

#include "ram.cuh"
#include "utils.cuh"
#include <iostream>

namespace vflog::ram {
    void CartesianOperator::execute(RelationalAlgebraMachine &ram) {
        auto res_reg1 = ram.get_register(result_register1);
        auto res_reg2 = ram.get_register(result_register2);

        size_t rel1_size = ram.get_rel(rel1.name)->get_versioned_size(version1);
        size_t rel2_size = ram.get_rel(rel2.name)->get_versioned_size(version2);

        res_reg1->resize(rel1_size * rel2_size);
        thrust::transform(EXE_POLICY, thrust::make_counting_iterator<size_t>(0),
                          thrust::make_counting_iterator<size_t>(rel1_size * rel2_size),
                          res_reg1->begin(),
                          [rel2_size] LAMBDA_TAG(size_t idx) {
                              return idx / rel2_size;
                          });
        res_reg2->resize(rel1_size * rel2_size);
        thrust::transform(EXE_POLICY, thrust::make_counting_iterator<size_t>(0),
                          thrust::make_counting_iterator<size_t>(rel1_size * rel2_size),
                          res_reg2->begin(),
                          [rel2_size] LAMBDA_TAG(size_t idx) {
                              return idx % rel2_size;
                          });
    }

    std::string CartesianOperator::to_string() {
        std::string res = "CartesianOperator(";
        res += "rel_t(\"" + rel1.name + "\"), " + std::to_string(version1) + ", ";
        res += "rel_t(\"" + rel2.name + "\"), " + std::to_string(version2) + ", ";
        res += "\"" + result_register1 + "\", \"" + result_register2 + "\")";
        return res;
    }
}

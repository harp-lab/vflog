
#include "mir.cuh"
#include "ram_instruction.cuh"
#include <memory>
#include <string>
#include <vector>

namespace vflog::mir {
std::vector<std::shared_ptr<ram::RAMInstruction>>
RAMGenPass::declare_relations() {
    // looking for all lines, if its decl, add to ram_instructions
    std::vector<std::shared_ptr<ram::RAMInstruction>> instrs;
    for (auto line : mir_program->lines()) {
        if (line->type == MIRNodeType::DECL) {
            // cast to MIRDecl
            auto mdecl = std::dynamic_pointer_cast<MIRDecl>(line);
            auto decl_instr =
                ram::declare(mdecl->rel_name, mdecl->arg_names.size());
            if (mdecl->data_path != nullptr) {
                io_instructions.push_back(
                    ram::load_file(ram::rel_t(mdecl->rel_name),
                                   mdecl->data_path, mdecl->arg_names.size()));
            }
            instrs.push_back(decl_instr);
        }
    }
    return instrs;
}
} // namespace vflog::mir

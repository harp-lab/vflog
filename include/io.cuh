
#pragma once

#include "hisa.cuh"
#include <string>

namespace vflog {

void read_kary_relation(const std::string &filename, vflog::multi_hisa &h,
                        int k);

} // namespace vflog

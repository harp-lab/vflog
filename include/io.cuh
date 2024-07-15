
#pragma once

#include "hisa.cuh"
#include <string>
#include <unordered_map>

namespace vflog {

void read_kary_relation(const std::string &filename, vflog::multi_hisa &h,
                        int k);
void read_kary_relation(const std::string &filename, vflog::multi_hisa &h,
                        int k, std::unordered_map<std::string, int> &dict);

void read_string_map(const std::string &filename,
                     std::unordered_map<std::string, int> &dict);

} // namespace vflog

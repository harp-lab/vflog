
#include <fstream>
#include <iostream>
#include <sstream>

#include "io.cuh"

namespace vflog {
std::string trim(const std::string &str) {
    size_t first = str.find_first_not_of(" \t\n\v\f\r");
    if (std::string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(" \t\n\v\f\r");
    return str.substr(first, (last - first + 1));
}

// function read in a k-arity relation file, load into a hisa
void read_kary_relation(const std::string &filename, vflog::multi_hisa &h,
                        int k) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: file not found" << std::endl;
        return;
    }

    HOST_VECTOR<HOST_VECTOR<vflog::internal_data_type>> tuples_vertical(k);

    std::string line;
    uint32_t line_count = 0;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        HOST_VECTOR<vflog::internal_data_type> tokens;
        std::string token;
        while (std::getline(iss, token, '\t')) {
            // trim whitespace on the token and convert into a uint32_t
            auto trimmed = trim(token);
            tokens.push_back(std::stoi(trimmed));
        }

        if (tokens.size() != k) {
            std::cerr << filename << "  Error: invalid arity" << std::endl;
            return;
        }

        for (size_t i = 0; i < k; i++) {
            tuples_vertical[i].push_back(tokens[i]);
        }
        line_count++;
    }

    h.init_load_vectical(tuples_vertical, line_count);
    // h.newt_self_deduplicate();
}

void read_kary_relation(const std::string &filename,
                        std::shared_ptr<multi_hisa> h, int k) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: file not found" << std::endl;
        return;
    }

    HOST_VECTOR<HOST_VECTOR<vflog::internal_data_type>> tuples_vertical(k);

    std::string line;
    uint32_t line_count = 0;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        HOST_VECTOR<vflog::internal_data_type> tokens;
        std::string token;
        while (std::getline(iss, token, '\t')) {
            // trim whitespace on the token and convert into a uint32_t
            auto trimmed = trim(token);
            tokens.push_back(std::stoi(trimmed));
        }

        if (tokens.size() != k) {
            std::cerr << "Error: invalid arity" << std::endl;
            return;
        }

        for (size_t i = 0; i < k; i++) {
            tuples_vertical[i].push_back(tokens[i]);
        }
        line_count++;
    }

    h->init_load_vectical(tuples_vertical, line_count);
    // h.newt_self_deduplicate();
}

// function check if a string is a number
bool is_number(const std::string &s) {
    return !s.empty() && std::find_if(s.begin(), s.end(), [](unsigned char c) {
                             return !std::isdigit(c);
                         }) == s.end();
}

void read_kary_relation(const std::string &filename, vflog::multi_hisa &h,
                        int k, std::unordered_map<std::string, int> &dict) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: file not found" << std::endl;
        return;
    }

    HOST_VECTOR<HOST_VECTOR<vflog::internal_data_type>> tuples_vertical(k);

    std::string line;
    uint32_t line_count = 0;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        HOST_VECTOR<vflog::internal_data_type> tokens;
        std::string token;
        while (std::getline(iss, token, '\t')) {
            // trim whitespace on the token and convert into a uint32_t
            auto trimmed = trim(token);
            if (is_number(trimmed)) {
                tokens.push_back(std::stoi(trimmed));
            } else {
                // found string's id in the dict
                if (dict.find(trimmed) == dict.end()) {
                    // if not error
                    std::cerr << "Error: string " << trimmed
                              << " not found in dict" << std::endl;
                    throw std::runtime_error("Error: string not found in dict");
                }
                tokens.push_back(dict[trimmed]);
            }
        }

        if (tokens.size() != k) {
            std::cerr << "Error: invalid arity" << std::endl;
            return;
        }

        for (size_t i = 0; i < k; i++) {
            tuples_vertical[i].push_back(tokens[i]);
        }
        line_count++;
    }
}

} // namespace vflog

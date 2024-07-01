#include <cstdint>
#include <cstdio>
#include <cuda/std/chrono>
#include <iostream>
#include <vector>
// thrust use TBB
// #define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_TBB

#include "../include/hisa.cuh"

#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <thrust/sequence.h>

void handler(int sig) {
    void *array[10];

    // get void*'s for all entries on the stack
    size_t size = backtrace(array, 10);
    char **strs = backtrace_symbols(array, size);
    for (int i = 0; i < size; i++) {
        printf("%s\n", strs[i]);
    }

    // print out all the frames to stderr
    fprintf(stderr, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(1);
}

const uint32_t TEST_CASE_SIZE = 10 * 10000;
const uint32_t REPEAT = 50;
const uint32_t TEST_ARIRTY = 2;

void generate_random_data(thrust::host_vector<uint32_t> &data_column1) {
    for (int i = 0; i < data_column1.size(); i++) {
        data_column1[i] = rand() % TEST_CASE_SIZE;
    }
}

void test_load_multi(
    thrust::host_vector<thrust::host_vector<uint32_t>> &data_columns,
    hisa::multi_hisa &h) {

    h.init_load_vectical(data_columns, data_columns[0].size());
    h.deduplicate();
    h.persist_newt();
}

void test_map(hisa::multi_hisa &h, uint64_t &hash_time, uint64_t &simple_time) {

    // print current GPU memory usage
    size_t free_byte;
    size_t total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    std::cout << "before simp free_byte: " << free_byte
              << " total_byte: " << total_byte << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    hisa::GpuSimplMap &simple_map = h.full_columns[0].unique_v_map_simp;
    hisa::device_data_t result_key = h.data[0];
    hisa::device_ranges_t result_value(h.total_tuples);
    simple_map.find(result_key, result_value);
    auto end = std::chrono::high_resolution_clock::now();
    simple_time +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    cudaMemGetInfo(&free_byte, &total_byte);
    std::cout << "after simp free_byte: " << free_byte
              << " total_byte: " << total_byte << std::endl;

    start = std::chrono::high_resolution_clock::now();
    // hisa::GpuMap hash_map{
    //     simple_map.keys.size(), 0.8,
    //     cuco::empty_key<hisa::internal_data_type>{UINT32_MAX},
    //     cuco::empty_value<hisa::offset_type>{UINT32_MAX}};
    auto hash_map = CREATE_V_MAP(simple_map.keys.size());

    auto insertpair_begin = thrust::make_transform_iterator(
        thrust::make_zip_iterator(thrust::make_tuple(
            simple_map.keys.begin(), simple_map.values.begin())),
        cuda::proclaim_return_type<hisa::GpuMapPair>([] __device__(auto &t) {
            return HASH_NAMESPACE::make_pair(thrust::get<0>(t),
                                             thrust::get<1>(t));
        }));
    std::cout << "simp map size : " << simple_map.keys.size()
              << " simp value size : " << simple_map.values.size() << std::endl;

    hash_map->insert(insertpair_begin,
                     insertpair_begin + simple_map.keys.size());

    hash_map->find(result_key.begin(), result_key.end(), result_value.begin());
    end = std::chrono::high_resolution_clock::now();
    hash_time +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    cudaMemGetInfo(&free_byte, &total_byte);
    std::cout << "after hashmap free_byte: " << free_byte
              << " total_byte: " << total_byte << std::endl;
    simple_map.keys.resize(0);
    simple_map.values.resize(0);
    simple_map.keys.shrink_to_fit();
    simple_map.values.shrink_to_fit();

    cudaMemGetInfo(&free_byte, &total_byte);
    std::cout << "after shrink free_byte: " << free_byte
              << " total_byte: " << total_byte << std::endl;
}

void raw_vertical_to_horizontal(
    thrust::host_vector<thrust::host_vector<uint32_t>> &data_columns,
    thrust::host_vector<uint32_t> &data_columns_horizontal) {
    auto total_size = data_columns.size() * data_columns[0].size();
    for (int i = 0; i < data_columns[0].size(); i++) {
        for (int j = 0; j < data_columns.size(); j++) {
            data_columns_horizontal.push_back(data_columns[j][i]);
        }
    }
}

void test_gpu_basic() {
    hisa::multi_hisa h(2);
    thrust::host_vector<thrust::host_vector<hisa::internal_data_type>>
        test_raw_host;
    test_raw_host.push_back({{1, 6, 3, 9, 1, 2, 3, 8}});
    test_raw_host.push_back({{3, 2, 3, 9, 3, 7, 3, 1}});
    // 3 8 1 1 3 2
    // 3 7 1 2 3 3
    h.init_load_vectical(test_raw_host, 8);
    auto start = std::chrono::high_resolution_clock::now();
    h.deduplicate();
    std::cout << "deduplicate done" << std::endl;
    thrust::host_vector<hisa::internal_data_type> data_host;
    for (int i = 0; i < 2; i++) {
        data_host = h.data[i];
        for (int j = 0; j < data_host.size(); j++) {
            std::cout << data_host[j] << " ";
        }
        std::cout << std::endl;
    }
    h.persist_newt();
    std::cout << "persist_newt done" << std::endl;
    hisa::device_data_t result_key = test_raw_host[0];
    thrust::host_vector<uint32_t> result_key_host = result_key;
    hisa::device_ranges_t result_value(8);
    if (h.full_columns[0].use_real_map) {
        auto &map_ptr = h.full_columns[0].unique_v_map;
        map_ptr->find(result_key.begin(), result_key.end(),
                      result_value.begin());
    } else {
        h.full_columns[0].unique_v_map_simp.find(result_key, result_value);
    }

    thrust::host_vector<hisa::comp_range_t> result_value_host =
        h.full_columns[0].unique_v_map_simp.values;
    for (int i = 0; i < result_value_host.size(); i++) {
        auto offset = static_cast<uint32_t>(result_value_host[i] >> 32);
        auto length = static_cast<uint32_t>(result_value_host[i]);
        std::cout << "(" << result_key_host[i] << " : " << offset << " "
                  << length << ") ";
    }
    std::cout << std::endl;
    h.print_all();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "total time: " << duration.count() << std::endl;
}

void test_join() {
    hisa::multi_hisa h1(2);
    thrust::host_vector<thrust::host_vector<hisa::internal_data_type>>
        test_raw_host1;
    test_raw_host1.push_back({{1, 6, 3, 9, 1, 2, 3, 8}});
    test_raw_host1.push_back({{3, 2, 3, 9, 3, 7, 3, 1}});
    h1.init_load_vectical(test_raw_host1, 8);
    h1.deduplicate();
    h1.persist_newt();
    // print full of h1
    std::cout << "h1 full: " << std::endl;
    for (int i = 0; i < 2; i++) {
        thrust::host_vector<hisa::internal_data_type> data_host(h1.data[i]);
        for (int j = 0; j < data_host.size(); j++) {
            std::cout << data_host[j] << " ";
        }
        std::cout << std::endl;
    }

    thrust::host_vector<hisa::internal_data_type> column_indices_host;
    column_indices_host = h1.full_columns[0].sorted_indices;
    for (int i = 0; i < column_indices_host.size(); i++) {
        std::cout << column_indices_host[i] << " ";
    }
    std::cout << std::endl;

    hisa::multi_hisa h2(2);
    thrust::host_vector<thrust::host_vector<hisa::internal_data_type>>
        test_raw_host2;
    test_raw_host2.push_back({{1, 6, 3}});
    test_raw_host2.push_back({{2, 2, 3}});
    h2.init_load_vectical(test_raw_host2, 3);
    h2.deduplicate();
    h2.persist_newt();

    // print full of h2
    std::cout << "h2 full: " << std::endl;
    for (int i = 0; i < 2; i++) {
        thrust::host_vector<hisa::internal_data_type> data_host(h2.data[i]);
        for (int j = 0; j < data_host.size(); j++) {
            std::cout << data_host[j] << " ";
        }
        std::cout << std::endl;
    }

    // join h1 h2 on 1th column
    hisa::device_data_t candidate_indices(3);
    thrust::sequence(candidate_indices.begin(), candidate_indices.end());
    hisa::device_pairs_t result_pair;
    // hisa::column_join(h1.full_columns[0], h2.full_columns[0], candidate_indices,
    //                   result_pair);
    hisa::column_join(h1, FULL, 0, h2, FULL, 0, candidate_indices, result_pair);
    thrust::host_vector<hisa::comp_pair_t> result_indices_host = result_pair;
    std::cout << "result_indices: ";
    for (int i = 0; i < result_indices_host.size(); i++) {
        std::cout << "(" << (result_indices_host[i] >> 32) << ","
                  << (result_indices_host[i] & 0xffffffff) << ") ";
    }
    std::cout << std::endl;
    // hisa::column_match(h1.full_columns[1], h2.full_columns[1], result_pair);
    hisa::column_match(h1, FULL, 1, h2, FULL, 1, result_pair);
    // print result
    result_indices_host = result_pair;
    std::cout << "result_indices: ";
    for (int i = 0; i < result_indices_host.size(); i++) {
        std::cout << "(" << (result_indices_host[i] >> 32) << ","
                  << (result_indices_host[i] & 0xffffffff) << ") ";
    }
    std::cout << std::endl;
}

int main() {
    // rmm::mr::cuda_me     mory_resource cuda_mr{};
    // rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr{
    //     &cuda_mr, 4 * 256 * 1024};
    // rmm::mr::set_current_device_resource(&mr);

    signal(SIGSEGV, handler);
    // generate 3 columns of random data
    thrust::host_vector<thrust::host_vector<uint32_t>> data_columns_vertical(
        TEST_ARIRTY);
    thrust::host_vector<uint32_t> data_columns_horizontal;

    for (int i = 0; i < TEST_ARIRTY; i++) {
        data_columns_vertical[i].resize(TEST_CASE_SIZE);
        generate_random_data(data_columns_vertical[i]);
    }
    raw_vertical_to_horizontal(data_columns_vertical, data_columns_horizontal);

    std::cout << "generate_random_data done" << std::endl;
    // hisa::hisa_cpu hashtrie_cpu(TEST_ARIRTY);
    // hashtrie_cpu.load_vectical(data_columns_vertical);
    // hashtrie_cpu.build_index();

    // std::cout << "hisa_cpu >>>>>>>> " << std::endl;
    // testcase_deduplicate();
    // std::cout << "hisa_gpu >>>>>>>> " << std::endl;
    // test_gpu_basic();

    std::cout << "hisa_gpu join >>>>>>>> " << std::endl;
    test_join();

    uint64_t total_hisa_load_time = 0;
    double total_hisa_hash_time = 0;
    uint64_t total_hashtrie_cpu_load_time = 0;
    uint64_t total_hash_time = 0;
    uint64_t total_muti_hisa_load_time = 0;
    uint64_t total_muti_hisa_hash_time = 0;
    uint64_t total_muti_hisa_dedup_time = 0;
    uint64_t total_hash_time_sort_time = 0;
    uint64_t total_muti_hisa_h2d_time = 0;

    uint64_t total_hash_map_find_time = 0;
    uint64_t total_simple_map_find_time = 0;

    for (int i = 0; i < REPEAT; i++) {
        std::cout << "repeat: " << i << std::endl;
        // test load

        // test load multi
        hisa::multi_hisa h(TEST_ARIRTY);
        auto start_multi = std::chrono::high_resolution_clock::now();
        test_load_multi(data_columns_vertical, h);
        auto end_multi = std::chrono::high_resolution_clock::now();
        auto duration_multi =
            std::chrono::duration_cast<std::chrono::microseconds>(end_multi -
                                                                  start_multi);
        // h.clear();
        total_muti_hisa_load_time += duration_multi.count();
        total_muti_hisa_hash_time += h.hash_time;
        total_muti_hisa_dedup_time += h.dedup_time;
        total_hash_time_sort_time += h.sort_time;
        total_muti_hisa_h2d_time += h.load_time;
    }

    std::cout << "total_hashtrie_cpu_load_time: "
              << total_hashtrie_cpu_load_time << std::endl;
    std::cout << "total hash time : " << total_hash_time << std::endl;
    std::cout << "total_hisa_load_time: " << total_hisa_load_time << std::endl;
    std::cout << "total_hisa_hash_time: " << total_hisa_hash_time << std::endl;
    std::cout << "total_muti_hisa_load_time: " << total_muti_hisa_load_time
              << std::endl;
    std::cout << "total_muti_hisa_hash_time: " << total_muti_hisa_hash_time
              << std::endl;
    std::cout << "total_muti_hisa_dedup_time: " << total_muti_hisa_dedup_time
              << std::endl;
    std::cout << "total_hash_time_sort_time: " << total_hash_time_sort_time
              << std::endl;
    std::cout << "total_muti_hisa_h2d_time: " << total_muti_hisa_h2d_time
              << std::endl;

    std::cout << "total_hash_map_find_time: " << total_hash_map_find_time
              << std::endl;
    std::cout << "total_simple_map_find_time: " << total_simple_map_find_time
              << std::endl;

    return 0;
}

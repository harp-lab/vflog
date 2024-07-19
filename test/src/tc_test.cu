
#include "vflog.cuh"

#include <iostream>
#include <thrust/sequence.h>
// #include <rmm/mr/device/managed_memory_resource.hpp>

void tc_barebone(char *data_path) {
    auto global_buffer = std::make_shared<vflog::d_buffer>(40960);
    vflog::multi_hisa edge(2, global_buffer);
    edge.set_default_index_column(0);
    edge.set_index_startegy(0, FULL, vflog::IndexStrategy::EAGER);
    edge.set_index_startegy(1, FULL, vflog::IndexStrategy::LAZY);
    vflog::read_kary_relation(data_path, edge, 2);
    edge.newt_self_deduplicate();
    std::cout << "Edge newt size: " << edge.get_versioned_size(NEWT)
              << std::endl;
    edge.persist_newt();
    std::cout << "Edge full size: " << edge.get_versioned_size(FULL)
              << std::endl;

    vflog::multi_hisa path(2, global_buffer);
    path.set_default_index_column(1);
    path.set_index_startegy(0, FULL, vflog::IndexStrategy::LAZY);
    path.set_index_startegy(0, DELTA, vflog::IndexStrategy::LAZY);
    path.set_index_startegy(1, DELTA, vflog::IndexStrategy::LAZY);
    path.set_index_startegy(1, FULL, vflog::IndexStrategy::EAGER);
    auto init_size = edge.total_tuples;
    // allocate newt on path
    path.allocate_newt(init_size);
    // copy edge FULL to path's newt
    auto &edge_full = edge.get_versioned_columns(FULL);
    auto &path_newt = path.get_versioned_columns(NEWT);
    auto input_indices_ptr = std::make_shared<vflog::device_indices_t>();
    input_indices_ptr->resize(edge_full[0].size());
    thrust::sequence(input_indices_ptr->begin(), input_indices_ptr->end());
    for (size_t i = 0; i < edge.arity; i++) {
        std::cout << "edge full size " << edge_full[i].size() << std::endl;
        vflog::column_copy(edge, FULL, i, path, NEWT, i, input_indices_ptr);
        path.newt_size = edge_full[i].size();
        path.total_tuples += input_indices_ptr->size();
        path_newt[i].raw_size = path.newt_size;
    }
    std::cout << "Copied edge to path" << std::endl;
    path.newt_self_deduplicate();
    path.persist_newt();
    std::cout << "Path DELTA size: " << path.get_versioned_size(FULL)
              << std::endl;

    // evaluate loop
    size_t iteration = 0;
    auto matched_path_ptr = std::make_shared<vflog::device_indices_t>();
    auto matched_edge_ptr = std::make_shared<vflog::device_indices_t>();
    KernelTimer timer;
    timer.start_timer();
    while (true) {
        std::cout << "Iteration " << iteration << std::endl;
        RelationVersion path_version = DELTA;
        vflog::host_buf_ref_t cached;

        // join edge's delta with path's full
        // path(a, c) :- path(a, b), edge(b, c).
        cached["path"] = matched_path_ptr;
        cached["path"]->resize(path.get_versioned_size(path_version));
        thrust::sequence(cached["path"]->begin(), cached["path"]->end());
        vflog::column_join(edge, FULL, 0, path, path_version, 1, cached, "path", matched_edge_ptr);
        cached["edge"] = matched_edge_ptr;
        size_t raw_newt_size = cached["path"]->size();
        // std::cout << "Raw newt size: " << raw_newt_size << std::endl;
        // materialize
        path.allocate_newt(raw_newt_size);
        vflog::column_copy(path, path_version, 0, path, NEWT, 0, cached["path"]);
        vflog::column_copy(edge, FULL, 1, path, NEWT, 1, cached["edge"]);
        path.newt_size = raw_newt_size;
        path.total_tuples += raw_newt_size;
        path.newt_self_deduplicate();
        path.persist_newt();
        cached.clear();

        auto delta_size = path.get_versioned_size(DELTA);
        std::cout << "Full size: " << path.get_versioned_size(FULL)
                  << " Delta size: " << delta_size << std::endl;
        // if (iteration == 0) {
        //     break;
        // }
        if (delta_size == 0) {
            break;
        }
        iteration++;
    }
    timer.stop_timer();
    auto elapsed = timer.get_spent_time();
    // path.print_raw_data(FULL);

    std::cout << "Total paths: " << path.full_size << std::endl;
    std::cout << "Elapsed time: " << elapsed << "s" << std::endl;

    path.print_stats();
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <data_path> <memory_system_flag>"
                  << std::endl;
        return 1;
    }
    // enable_rmm_allocator();
    // rmm::mr::cuda_memory_resource cuda_mr{};
    // rmm::mr::set_current_device_resource(&cuda_mr);
    // first arg is data path
    char *data_path = argv[1];
    int memory_system_flag = atoi(argv[2]);
    // if (memory_system_flag == 0) {
    //     rmm::mr::set_current_device_resource(&cuda_mr);
    // } else if (memory_system_flag == 1) {
    //     rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr{
    //     &cuda_mr, 4 * 256 * 1024};
    //     rmm::mr::set_current_device_resource(&mr);
    // } else if (memory_system_flag == 2) {
    //     rmm::mr::managed_memory_resource mr{};
    //     rmm::mr::set_current_device_resource(&mr);
    // } else {
    //     rmm::mr::set_current_device_resource(&cuda_mr);
    // }

    tc_barebone(data_path);
    return 0;
}

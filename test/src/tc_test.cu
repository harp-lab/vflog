
#include "relation.cuh"

#include "tc_test.h"
#include <iostream>
#include <thrust/sequence.h>

void print_index_map(std::shared_ptr<hisa::GpuMap> &unique_v_map) {
    auto uniq_size = unique_v_map->size();
    hisa::device_data_t all_unqiue_values(uniq_size);
    hisa::device_ranges_t all_ranges(uniq_size);
    unique_v_map->retrieve_all(all_unqiue_values.begin(), all_ranges.begin());
    // print all unique values and ranges
    for (size_t i = 0; i < uniq_size; i++) {
        uint64_t range = all_ranges[i];
        std::cout << "Unique value: " << all_unqiue_values[i] << " Range: ("
                  << (range >> 32) << ", " << (range & 0xFFFFFFFF) << ")"
                  << std::endl;
    }
}

void tc_barebone(char *data_path) {
    hisa::multi_hisa edge(2);
    edge.set_default_index_column(0);
    edge.set_index_startegy(0, FULL, hisa::IndexStrategy::EAGER);
    edge.set_index_startegy(1, FULL, hisa::IndexStrategy::LAZY);
    hisa::read_kary_relation(data_path, edge, 2);
    edge.deduplicate();
    edge.persist_newt();
    std::cout << "Edge full size: " << edge.get_versioned_size(FULL)
              << std::endl;

    hisa::multi_hisa path(2);
    path.set_default_index_column(1);
    path.set_index_startegy(0, FULL, hisa::IndexStrategy::LAZY);
    path.set_index_startegy(0, DELTA, hisa::IndexStrategy::LAZY);
    path.set_index_startegy(1, DELTA, hisa::IndexStrategy::LAZY);
    path.set_index_startegy(1, FULL, hisa::IndexStrategy::EAGER);
    auto init_size = edge.total_tuples;
    // allocate newt on path
    path.allocate_newt(init_size);
    // copy edge FULL to path's newt
    auto &edge_full = edge.get_versioned_columns(FULL);
    auto &path_newt = path.get_versioned_columns(NEWT);
    hisa::device_indices_t input_indices(init_size);
    thrust::sequence(input_indices.begin(), input_indices.end());
    for (size_t i = 0; i < edge.arity; i++) {
        std::cout << "edge full size " << edge_full[i].size() << std::endl;
        hisa::column_copy(edge, FULL, i, path, NEWT, i, input_indices);
        path.newt_size = edge_full[i].size();
        path.total_tuples += input_indices.size();
        path_newt[i].raw_size = path.newt_size;
    }
    std::cout << "Copied edge to path" << std::endl;
    path.deduplicate();
    path.persist_newt();
    std::cout << "Path full size: " << path.get_versioned_size(FULL)
              << std::endl;

    // evaluate loop
    size_t iteration = 0;
    hisa::device_indices_t matched_edges;
    hisa::device_bitmap_t matched_bitmap(init_size, false);
    hisa::device_indices_t matched_paths;
    matched_paths.swap(input_indices);
    KernelTimer timer;
    timer.start_timer();
    while (true) {
        std::cout << "Iteration " << iteration << std::endl;
        RelationVersion path_version = DELTA;
        if (iteration == 0) {
            // in first iter, use full as delta
            path_version = FULL;
        }

        // join edge's delta with path's full
        // path(a, c) :- path(a, b), edge(b, c).
        matched_paths.resize(path.get_versioned_size(path_version));
        thrust::sequence(matched_paths.begin(), matched_paths.end());
        auto &edge_full = edge.get_versioned_columns(FULL);
        auto &path_delta = path.get_versioned_columns(path_version);

        hisa::column_join(edge, FULL, 0, path, path_version, 1, matched_paths,
                          matched_edges, matched_bitmap);
        size_t raw_newt_size = matched_paths.size();

        // std::cout << "Matched edges: " << matched_edges.size() << std::endl;
        // materialize
        path.allocate_newt(raw_newt_size);
        auto &path_newt = path.get_versioned_columns(NEWT);
        // std::cout << path_newt[0].raw_data << std::endl;

        hisa::column_copy(path, path_version, 0, path, NEWT, 0, matched_paths);
        path.newt_columns[0].raw_size = raw_newt_size;
        hisa::column_copy(edge, FULL, 1, path, NEWT, 1, matched_edges);
        path.newt_columns[1].raw_size = raw_newt_size;
        path.newt_size = raw_newt_size;
        path.total_tuples += raw_newt_size;
        // std::cout << "Newt size before dedup: " << path.newt_size << std::endl;
        // deduplicate NEWT itself

        path.deduplicate();
        // path.print_raw_data(NEWT);

        path.persist_newt();
        // path.print_raw_data(DELTA);

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
    // enable_rmm_allocator();
    rmm::mr::cuda_memory_resource cuda_mr{};
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr{
        &cuda_mr, 4 * 256 * 1024};
    rmm::mr::set_current_device_resource(&mr);

    // first arg is data path
    char *data_path = argv[1];
    tc_barebone(data_path);
    return 0;
}

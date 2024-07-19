
#include "vflog.cuh"

#include <iostream>
// #include <rmm/mr/device/managed_memory_resource.hpp>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

void sg_barebone(char *data_path) {
    auto global_buffer = std::make_shared<vflog::d_buffer>(40960);
    // .decl edge(a: int, b: int)
    vflog::multi_hisa edge(2, global_buffer);
    edge.set_default_index_column(0);
    edge.set_index_startegy(0, FULL, vflog::IndexStrategy::EAGER);
    edge.set_index_startegy(1, FULL, vflog::IndexStrategy::LAZY);
    vflog::read_kary_relation(data_path, edge, 2);
    edge.newt_self_deduplicate();
    edge.persist_newt();
    std::cout << "Edge full size: " << edge.get_versioned_size(FULL)
              << std::endl;

    // .decl path(a: int, b: int)
    vflog::multi_hisa sg(2, global_buffer);
    sg.set_default_index_column(0);

    // use ptr for buffer, because we want share buffer between different
    // RA operations
    auto matched_a_ptr = std::make_shared<vflog::device_indices_t>();
    auto matched_b_ptr = std::make_shared<vflog::device_indices_t>();
    auto matched_x_ptr = std::make_shared<vflog::device_indices_t>();
    auto matched_y_ptr = std::make_shared<vflog::device_indices_t>();

    // non recursive part
    vflog::host_buf_ref_t cached;
    // sg(x, y) :- edge(a, x), edge(a, y), x != y.
    matched_x_ptr->resize(edge.get_versioned_size(FULL));
    cached["edge1"] = matched_x_ptr;
    thrust::sequence(EXE_POLICY, cached["edge1"]->begin(),
                     cached["edge1"]->end());
    vflog::column_join(edge, FULL, 0, edge, FULL, 0, cached, "edge1",
                       matched_y_ptr);
    cached["edge2"] = matched_y_ptr;
    // filter match x y
    vflog::device_bitmap_t filter_bitmap(matched_x_ptr->size(), false);
    thrust::transform(EXE_POLICY, matched_x_ptr->begin(), matched_x_ptr->end(),
                      matched_y_ptr->begin(), filter_bitmap.begin(),
                      [x_ptrs = edge.get_raw_data_ptrs(FULL, 1),
                       y_ptrs = edge.get_raw_data_ptrs(
                           FULL, 1)] LAMBDA_TAG(auto &x, auto &y) {
                          return x_ptrs[x] != y_ptrs[y];
                      });
    // filter x y
    auto matched_x_end = thrust::remove_if(
        EXE_POLICY, matched_x_ptr->begin(), matched_x_ptr->end(),
        filter_bitmap.begin(), thrust::logical_not<bool>());
    matched_x_ptr->resize(matched_x_end - matched_x_ptr->begin());
    auto matched_y_end = thrust::remove_if(
        EXE_POLICY, matched_y_ptr->begin(), matched_y_ptr->end(),
        filter_bitmap.begin(), thrust::logical_not<bool>());
    matched_y_ptr->resize(matched_y_end - matched_y_ptr->begin());

    // materialize sg
    sg.allocate_newt(matched_x_ptr->size());
    vflog::column_copy(edge, FULL, 1, sg, NEWT, 0, matched_x_ptr);
    vflog::column_copy(edge, FULL, 1, sg, NEWT, 1, matched_y_ptr);
    sg.newt_size = matched_x_ptr->size();
    sg.total_tuples += matched_x_ptr->size();
    sg.newt_self_deduplicate();
    sg.persist_newt();
    matched_x_ptr->clear();
    matched_x_ptr->shrink_to_fit();
    matched_y_ptr->clear();
    matched_y_ptr->shrink_to_fit();
    std::cout << "SG full size before : " << sg.get_versioned_size(FULL)
              << std::endl;
    // print delta size
    std::cout << "SG delta size before : " << sg.get_versioned_size(DELTA)
              << std::endl;

    // evaluate loop
    size_t iteration = 0;
    KernelTimer timer;
    KernelTimer timer2;
    float alloc_newt_time = 0;
    float copy_time = 0;
    float dedup_time = 0;
    float persist_time = 0;
    float join_time = 0;
    timer.start_timer();

    while (true) {
        std::cout << "Iteration " << iteration << std::endl;
        RelationVersion sg_version = DELTA;

        // sg(x, y) :-  sg(a, b), edge(a, x), edge(b, y).
        timer2.start_timer();
        vflog::host_buf_ref_t cached;
        cached["sg"] = matched_a_ptr;
        cached["sg"]->resize(sg.get_versioned_size(sg_version));
        thrust::sequence(cached["sg"]->begin(), cached["sg"]->end());
        vflog::column_join(edge, FULL, 0, sg, sg_version, 0, cached, "sg",
                           matched_x_ptr);
        cached["edge1"] = matched_x_ptr;
        // metavar sg is useless after join pop it from cache
        vflog::column_join(edge, FULL, 0, sg, sg_version, 1, cached, "sg",
                           matched_y_ptr, true);
        cached["edge2"] = matched_y_ptr;
        timer2.stop_timer();
        join_time += timer2.get_spent_time();
        auto raw_newt_size = cached["edge2"]->size();
        timer2.start_timer();
        sg.allocate_newt(raw_newt_size);
        timer2.stop_timer();
        alloc_newt_time += timer2.get_spent_time();
        timer2.start_timer();
        vflog::column_copy(edge, FULL, 1, sg, NEWT, 0, cached["edge1"]);
        vflog::column_copy(edge, FULL, 1, sg, NEWT, 1, cached["edge2"]);
        timer2.stop_timer();
        copy_time += timer2.get_spent_time();
        sg.newt_size = raw_newt_size;
        sg.total_tuples += raw_newt_size;

        timer2.start_timer();
        sg.newt_self_deduplicate();
        timer2.stop_timer();
        dedup_time += timer2.get_spent_time();
        // sg.print_raw_data(NEWT);

        timer2.start_timer();
        sg.persist_newt();
        timer2.stop_timer();
        persist_time += timer2.get_spent_time();

        // fixpoint check
        auto delta_size = sg.get_versioned_size(DELTA);
        if (delta_size == 0) {
            break;
        }
        iteration++;
    }

    // sg.print_raw_data(DELTA);
    timer.stop_timer();
    auto elapsed = timer.get_spent_time();

    std::cout << "SG full size after : " << sg.get_versioned_size(FULL)
              << std::endl;
    std::cout << "Elapsed time: " << elapsed << "s" << std::endl;
    std::cout << "Join time: " << join_time << "s" << std::endl;
    std::cout << "Alloc newt time: " << alloc_newt_time << "s" << std::endl;
    std::cout << "Copy time: " << copy_time << "s" << std::endl;
    std::cout << "Dedup time: " << dedup_time << "s" << std::endl;
    std::cout << "Persist time: " << persist_time << "s" << std::endl;
    sg.print_stats();
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <data_path> <memory_system_flag>"
                  << std::endl;
        return 1;
    }
    // enable_rmm_allocator();
    // rmm::mr::cuda_memory_resource cuda_mr{};
    // // rmm::mr::set_current_device_resource(&cuda_mr);
    // // first arg is data path
    char *data_path = argv[1];
    // int memory_system_flag = atoi(argv[2]);
    // if (memory_system_flag == 0) {
    //     rmm::mr::set_current_device_resource(&cuda_mr);
    // } else if (memory_system_flag == 1) {
    //     rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr{
    //         &cuda_mr, 4 * 256 * 1024};
    //     rmm::mr::set_current_device_resource(&mr);
    // } else if (memory_system_flag == 2) {
    //     rmm::mr::managed_memory_resource mr{};
    //     rmm::mr::set_current_device_resource(&mr);
    // } else {
    //     rmm::mr::set_current_device_resource(&cuda_mr);
    // }

    sg_barebone(data_path);
    return 0;
}

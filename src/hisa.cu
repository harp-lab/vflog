
#include "hisa.cuh"
#include <iostream>

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

namespace vflog {

multi_hisa::multi_hisa(int arity, d_buffer_ptr buffer, size_t default_idx) {
    this->arity = arity;
    newt_size = 0;
    full_size = 0;
    delta_size = 0;
    full_columns.resize(arity);
    delta_columns.resize(arity);
    newt_columns.resize(arity);
    data.resize(arity);
    if (buffer) {
        this->buffer = buffer;
    } else {
        this->buffer = std::make_shared<d_buffer>(40960);
    }

    for (int i = 0; i < arity; i++) {
        full_columns[i].column_idx = i;
        delta_columns[i].column_idx = i;
        newt_columns[i].column_idx = i;
    }
    set_default_index_column(default_idx);
}

void multi_hisa::init_load_vectical(
    HOST_VECTOR<HOST_VECTOR<internal_data_type>> &tuples, size_t rows) {
    auto load_start = std::chrono::high_resolution_clock::now();
    auto total_tuples = tuples[0].size();
    for (int i = 0; i < arity; i++) {
        // extract the i-th column
        // thrust::device_vector<internal_data_type> column_data(total_tuples);
        data[i].resize(total_tuples);
        cudaMemcpy(data[i].data().get(), tuples[i].data(),
                   tuples[i].size() * sizeof(internal_data_type),
                   cudaMemcpyHostToDevice);
        // save columns raw
    }
    this->total_tuples = total_tuples;
    this->newt_size = total_tuples;
    // set newt
    for (int i = 0; i < arity; i++) {
        newt_columns[i].raw_size = total_tuples;
    }
    auto load_end = std::chrono::high_resolution_clock::now();
    this->load_time += std::chrono::duration_cast<std::chrono::microseconds>(
                           load_end - load_start)
                           .count();
}

void multi_hisa::allocate_newt(size_t size) {
    auto old_size = capacity;
    if (total_tuples + size < capacity) {
        // std::cout << "no need to allocate newt" << std::endl;
        // return;  
        size = 0;
    }
    // compute offset of each version
    auto new_capacity = old_size + size;

    for (int i = 0; i < arity; i++) {
        data[i].resize(old_size + size);
    }
    capacity = old_size + size;

    // newt_size += size;
    // for (int i = 0; i < arity; i++) {
    //     newt_columns[i].raw_data = data[i].data() + old_size;
    // }
}

void multi_hisa::load_column_cpu(VetricalColumnCpu &columns_cpu,
                                 int column_idx) {
    auto total_tuples = columns_cpu.raw_data.size();
    capacity = total_tuples;
    data[column_idx].resize(total_tuples);
    cudaMemcpy(data[column_idx].data().get(), columns_cpu.raw_data.data(),
               columns_cpu.raw_data.size() * sizeof(internal_data_type),
               cudaMemcpyHostToDevice);
    this->total_tuples = total_tuples;
    this->newt_size = columns_cpu.newt_size;
    this->full_size = columns_cpu.full_size;
    this->delta_size = columns_cpu.delta_size;
    if (columns_cpu.full_size == 0) {
        return;
    }
    // set ptr
    full_columns[column_idx].raw_offset = 0;
    delta_columns[column_idx].raw_offset = columns_cpu.delta_head_offset;
    newt_columns[column_idx].raw_offset = columns_cpu.newt_head_offset;
    // copy sorted indices
    if (columns_cpu.full_size != 0) {
        full_columns[column_idx].sorted_indices.resize(columns_cpu.full_size);
        full_columns[column_idx].raw_size = columns_cpu.full_size;
        cudaMemcpy(full_columns[column_idx].sorted_indices.data().get(),
                   columns_cpu.full_sorted_indices.data(),
                   columns_cpu.full_sorted_indices.size() *
                       sizeof(internal_data_type),
                   cudaMemcpyHostToDevice);
    }
    if (columns_cpu.delta_size != 0) {
        delta_columns[column_idx].sorted_indices.resize(columns_cpu.delta_size);
        delta_columns[column_idx].raw_size = columns_cpu.delta_size;
        cudaMemcpy(delta_columns[column_idx].sorted_indices.data().get(),
                   columns_cpu.delta_sorted_indices.data(),
                   columns_cpu.delta_sorted_indices.size() *
                       sizeof(internal_data_type),
                   cudaMemcpyHostToDevice);
    }
    if (columns_cpu.newt_size != 0) {
        newt_columns[column_idx].sorted_indices.resize(columns_cpu.newt_size);
        newt_columns[column_idx].raw_size = columns_cpu.newt_size;
        cudaMemcpy(newt_columns[column_idx].sorted_indices.data().get(),
                   columns_cpu.newt_sorted_indices.data(),
                   columns_cpu.newt_sorted_indices.size() *
                       sizeof(internal_data_type),
                   cudaMemcpyHostToDevice);
    }
}

void multi_hisa::print_all(bool sorted) {
    // print all columns in full
    HOST_VECTOR<internal_data_type> column(total_tuples);
    HOST_VECTOR<internal_data_type> unique_value(total_tuples);
    for (int i = 0; i < arity; i++) {
        thrust::copy(data[i].begin() + full_columns[i].raw_offset,
                     data[i].begin() + full_columns[i].raw_offset +
                         full_columns[i].raw_size,
                     column.begin());
        std::cout << "column data " << i << " " << total_tuples << ":\n";
        for (int j = 0; j < column.size(); j++) {
            std::cout << column[j] << " ";
        }
        std::cout << std::endl;
        std::cout << "unique values " << i << " "
                  << full_columns[i].unique_v.size() << ":\n";
        unique_value.resize(full_columns[i].unique_v.size());
        thrust::copy(full_columns[i].unique_v.begin(),
                     full_columns[i].unique_v.end(), unique_value.begin());
        for (int j = 0; j < full_columns[i].unique_v.size(); j++) {
            std::cout << unique_value[j] << " ";
        }
        std::cout << std::endl;
    }
}

void multi_hisa::print_raw_data(RelationVersion ver) {
    std::printf("print raw data\n");
    HOST_VECTOR<HOST_VECTOR<internal_data_type>> columns_host(arity);
    for (int i = 0; i < arity; i++) {
        columns_host[i].resize(get_versioned_size(ver));
        cudaMemcpy(columns_host[i].data(),
                   data[i].data().get() +
                       get_versioned_columns(ver)[i].raw_offset,
                   get_versioned_size(ver) * sizeof(internal_data_type),
                   cudaMemcpyDeviceToHost);
    }
    // radix sort host
    thrust::host_vector<internal_data_type> column_host(
        get_versioned_size(ver));
    thrust::host_vector<internal_data_type> sorted_indices_host(
        get_versioned_size(ver));
    thrust::sequence(sorted_indices_host.begin(), sorted_indices_host.end());
    // for (int i = arity - 1; i >= 0; i--) {
    //     auto &column = columns_host[i];
    //     thrust::gather(sorted_indices_host.begin(),
    //     sorted_indices_host.end(),
    //                    column.begin(), column_host.begin());
    //     thrust::stable_sort_by_key(column_host.begin(), column_host.end(),
    //                                sorted_indices_host.begin());
    // }
    // permute the columns
    for (int i = 0; i < arity; i++) {
        thrust::gather(sorted_indices_host.begin(), sorted_indices_host.end(),
                       columns_host[i].begin(), column_host.begin());
        thrust::copy(column_host.begin(), column_host.end(),
                     columns_host[i].begin());
    }

    for (size_t i = 0; i < get_versioned_size(ver); i++) {
        for (int j = 0; j < arity; j++) {
            std::cout << columns_host[j][i] << " ";
        }
        std::cout << std::endl;
    }
}

void multi_hisa::fit() {
    total_tuples = newt_size + full_size;
    for (int i = 0; i < arity; i++) {
        data[i].resize(total_tuples);
        data[i].shrink_to_fit();
    }
}


void multi_hisa::print_stats() {
    std::cout << "sort time: " << sort_time / 1000000.0 << std::endl;
    std::cout << "hash time: " << hash_time / 1000000.0 << std::endl;
    std::cout << "dedup time: " << dedup_time / 1000000.0 << std::endl;
    std::cout << "merge time: " << merge_time / 1000000.0 << std::endl;
}

void multi_hisa::clear() {
    for (int i = 0; i < arity; i++) {
        full_columns[i].raw_offset = 0;
        full_columns[i].sorted_indices.resize(0);
        full_columns[i].sorted_indices.shrink_to_fit();
        full_columns[i].clear_unique_v();

        delta_columns[i].raw_offset = 0;
        delta_columns[i].sorted_indices.resize(0);
        delta_columns[i].sorted_indices.shrink_to_fit();
        delta_columns[i].clear_unique_v();

        newt_columns[i].raw_offset = 0;
        newt_columns[i].sorted_indices.resize(0);
        newt_columns[i].sorted_indices.shrink_to_fit();
        newt_columns[i].clear_unique_v();

        data[i].resize(0);
        data[i].shrink_to_fit();
    }
    newt_size = 0;
    full_size = 0;
    delta_size = 0;
    total_tuples = 0;
}

} // namespace vflog

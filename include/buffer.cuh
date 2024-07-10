
#pragma once

#include "../include/utils.cuh"
#include <memory>

namespace vflog {

// a Managed buffer used during
// 1. merge
// 2. compute index
// managed by thrust temporary buffer
// now its just a wrapper of rmm device vector, we need figure out a better version 
struct d_buffer {

    ptr_and_size_t ptr_and_size;

    thrust::device_system_tag tag = thrust::device;

    float ratio = 1.25;

    d_buffer(size_t size) {
        ptr_and_size =
            thrust::get_temporary_buffer<internal_data_type>(tag, size * ratio);
    }

    ~d_buffer() {
        thrust::return_temporary_buffer(tag, ptr_and_size.first,
                                        ptr_and_size.second);
    }

    internal_data_type *data() { return ptr_and_size.first.get(); }

    size_t size() { return ptr_and_size.second; }

    void reserve(size_t size) {
        if (size > ptr_and_size.second) {
            thrust::return_temporary_buffer(tag, ptr_and_size.first,
                                            ptr_and_size.second);
            ptr_and_size =
                thrust::get_temporary_buffer<internal_data_type>(tag, size * ratio);
        }
    }
};

using d_buffer_ptr = std::shared_ptr<d_buffer>;

} // namespace hisa

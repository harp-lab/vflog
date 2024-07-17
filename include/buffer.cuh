
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

   device_data_t internal;

    float ratio = 1.25;

    d_buffer(size_t size) {
        internal = device_data_t(size);
    }

    internal_data_type *data() { return internal.RAW_PTR; }

    size_t size() { return internal.size(); }

    void reserve(size_t size) {
        if (size > internal.size()) {
            internal.resize(size* ratio);
        }
    }
};

using d_buffer_ptr = std::shared_ptr<d_buffer>;

} // namespace hisa

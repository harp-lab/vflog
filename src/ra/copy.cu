
#include "ram.cuh"
#include "utils.cuh"

#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>

namespace vflog {

void column_copy(multi_hisa &src, RelationVersion src_ver, size_t src_idx,
                 multi_hisa &dst, RelationVersion dst_ver, size_t dst_idx,
                 std::shared_ptr<device_indices_t> &indices) {
    if (indices->size() == 0) {
        return;
    }
    // gather
    auto src_raw_begin = src.data[src_idx].begin() +
                         src.get_versioned_columns(src_ver)[src_idx].raw_offset;
    auto dst_raw_begin =
        dst.data[dst_idx].begin() +
        dst.get_versioned_columns(dst_ver)[dst_idx].raw_offset +
        dst.get_versioned_columns(dst_ver)[dst_idx].raw_size;
    thrust::gather(EXE_POLICY, indices->begin(), indices->end(), src_raw_begin,
                   dst_raw_begin);
    // TODO: check this
    dst.get_versioned_columns(dst_ver)[dst_idx].raw_size += indices->size();
}

void column_copy_all(multi_hisa &src, RelationVersion src_version,
                     size_t src_idx, multi_hisa &dst,
                     RelationVersion dst_version, size_t dst_idx, bool append) {
    if (src.get_versioned_columns(src_version)[src_idx].raw_size == 0) {
        return;
    }
    auto &src_column = src.get_versioned_columns(src_version)[src_idx];
    auto &dst_column = dst.get_versioned_columns(dst_version)[dst_idx];
    auto src_raw_begin = src.data[src_idx].begin() + src_column.raw_offset;
    auto dst_raw_begin =
        dst.data[dst_idx].begin() + dst_column.raw_offset + dst_column.raw_size;
    thrust::copy(EXE_POLICY, src_raw_begin, src_raw_begin + src_column.raw_size,
                 dst_raw_begin);
    dst_column.raw_size += src_column.raw_size;
}

void column_copy_indices(multi_hisa &src, RelationVersion src_version,
                         size_t src_idx, multi_hisa &dst,
                         RelationVersion dst_version, size_t dst_idx,
                         std::shared_ptr<device_indices_t> &indices,
                         bool append) {
    if (src.get_versioned_columns(src_version)[src_idx].raw_size == 0) {
        return;
    }
    auto &src_column = src.get_versioned_columns(src_version)[src_idx];
    auto &dst_column = dst.get_versioned_columns(dst_version)[dst_idx];
    auto src_raw_begin = src.data[src_idx].begin() + src_column.raw_offset;
    auto dst_raw_begin =
        dst.data[dst_idx].begin() + dst_column.raw_offset + dst_column.raw_size;

    auto id_begin = indices->begin();
    // add the uid to the top 4 bit of the sequenced value
    thrust::transform(
        EXE_POLICY, id_begin, id_begin + indices->size(), dst_raw_begin,
        [uid = src.uid] LAMBDA_TAG(auto &x) { return x | (uid << 28); });
    dst_column.raw_size += indices->size();
}

void ProjectOperator::execute(RelationalAlgebraMachine &ram) {
    auto src_ptr = ram.rels[src.rel];
    if (src.version == FULL && src.is_frozen()) {
        src_ptr = ram.get_frozen(src.rel, src.frozen_idx);
        if (src_ptr == nullptr) {
            // frozen relation not generated yet
            return;
        }
    }
    if (src_ptr->get_versioned_size(src.version) == 0) {
        return;
    }
    if (ram.overflow_rel_name == dst.rel) {
        std::cout << "<<<<< " << std::endl;
        column_copy(*src_ptr, src.version, src.idx,
                    *ram.overflow_rel, dst.version, dst.idx,
                    ram.cached_indices[meta_var]);
    } else {
        column_copy(*src_ptr, src.version, src.idx,
                    *ram.rels[dst.rel], dst.version, dst.idx,
                    ram.cached_indices[meta_var]);
    }
}

} // namespace vflog
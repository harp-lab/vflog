
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

namespace ram {

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
        // std::cout << "<<<<< " << std::endl;
        column_copy(*src_ptr, src.version, src.idx, *ram.overflow_rel,
                    dst.version, dst.idx, ram.cached_indices[meta_var]);
    } else {
        column_copy(*src_ptr, src.version, src.idx, *ram.rels[dst.rel],
                    dst.version, dst.idx, ram.cached_indices[meta_var]);
    }
}

void ProjectHostOperator::execute(RelationalAlgebraMachine &ram) {
    // get end of the column
    auto rel = ram.rels[dst.name];
    rel->allocate_newt(src_size);
    for (int i = 0; i < rel->arity; i++) {
        auto &dst_column = rel->get_versioned_columns(RelationVersion::NEWT)[i];
        auto dst_raw_begin = rel->data[i].data().get() + dst_column.raw_offset +
                             dst_column.raw_size;
        // copy the data
        cudaMemcpy(dst_raw_begin, host_src,
                   src_size * sizeof(internal_data_type),
                   cudaMemcpyHostToDevice);
        dst_column.raw_size = dst_column.raw_size + src_size;
    }
    rel->newt_size += src_size;
    rel->total_tuples += src_size;
}

std::string ProjectHostOperator::to_string() {
    return "project_host_op(" + dst.to_string() + ", " +
           std::to_string(src_size) + ")";
}

std::string ProjectOperator::to_string() {
    return "project_op(" + src.to_string() + ", " + dst.to_string() + ", \"" +
           meta_var + "\")";
}

void ProjectIdOperator::execute(RelationalAlgebraMachine &ram) {
    auto &src_idx_ptr = ram.cached_indices[meta_var];
    if (src_idx_ptr->size() == 0) {
        return;
    }
    auto &dst_column =
        ram.rels[dst.rel]->get_versioned_columns(dst.version)[dst.idx];
    auto dst_raw_begin = ram.rels[dst.rel]->data[dst.idx].begin() +
                         dst_column.raw_offset + dst_column.raw_size;

    thrust::copy(EXE_POLICY, src_idx_ptr->begin(), src_idx_ptr->end(),
                 dst_raw_begin);
    dst_column.raw_size += src_idx_ptr->size();
}

std::string ProjectIdOperator::to_string() {
    return "project_id_op(" + dst.to_string() + ", \"" + meta_var + "\")";
}

} // namespace ram

} // namespace vflog
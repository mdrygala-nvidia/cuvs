// Experiment harness, part 3: NEW batched versions.
// Not part of the library. Included by pq_exp_ivf_pq_build.cuh (and pq_exp_main.cu).
#pragma once

#include "ivf_pq_build.cuh"

#include <raft/core/resource/cublas_handle.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace cuvs::neighbors::ivf_pq::detail {

/** cudaEvent timer: start(); ...GPU work...; ms = stop(); (stop() synchronizes the event). */
class gpu_timer {
 public:
  gpu_timer()
  {
    RAFT_CUDA_TRY(cudaEventCreate(&start_));
    RAFT_CUDA_TRY(cudaEventCreate(&stop_));
  }
  ~gpu_timer()
  {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  gpu_timer(const gpu_timer&)            = delete;
  gpu_timer& operator=(const gpu_timer&) = delete;

  void start(cudaStream_t stream) { RAFT_CUDA_TRY(cudaEventRecord(start_, stream)); }
  float stop(cudaStream_t stream)
  {
    RAFT_CUDA_TRY(cudaEventRecord(stop_, stream));
    RAFT_CUDA_TRY(cudaEventSynchronize(stop_));
    float ms = 0;
    RAFT_CUDA_TRY(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

 private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

// ---------------------------------------------------------------------------------------------
// Both functions below are derived from train_per_subset() in ivf_pq_build.cuh (approx. lines
// 319-408; re-check line numbers against the current tree).
//
// STAGE 1 (this file now): the `for j in pq_dim` loop is gone. The residuals of ALL subspaces are
// computed at once:
//     residual[r, subspace j] = rotation_block_j * trainset[r]  -  centers_rot[labels[r]]_j
// The two functions differ only in how the residuals are laid out in memory (and so in how they
// are produced). Set PQ_EXP_CHECK=1 to also run the old per-subspace loop, compare, and time it.
//
// The k-means (STAGE 2+) and transpose_pq_centers are not run yet, so the codebooks stay unset.
// See train_per_subset_old for the per-subspace k-means code that will be batched next.
//
// Planned call tree for the later stages (copy a function only when it is changed, and copy
// upward from the change; each copy gets a comment with its origin and what changed):
//  train_per_subset_batched_*
//   `- build_clusters_batched                 (origin: kmeans_balanced.cuh, build_clusters)
//       |- calc_centers_and_sizes_batched    (origin: calc_centers_and_sizes)
//       `- balancing_em_iters_batched        (origin: balancing_em_iters)
//           |- adjust_centers_batched        (origin: adjust_centers)
//           `- predict_batched               (origin: predict)
// ---------------------------------------------------------------------------------------------

/**
 * Batched version of cuvs::cluster::kmeans_balanced::helpers::build_clusters (origin:
 * kmeans_balanced_build_clusters_impl.cuh:44, which calls detail::build_clusters in
 * kmeans_balanced.cuh; re-check line numbers). It clusters `n_batches` independent problems at
 * once, where n_batches is the first extent of every argument. The library function takes one
 * problem; here every argument gets a leading batch dimension:
 *   X             [n_batches, n_rows, dim]
 *   centroids     [n_batches, n_clusters, dim]      (output)
 *   labels        [n_batches, n_rows]               (output)
 *   cluster_sizes [n_batches, n_clusters]           (output)
 *   X_norm        [n_batches, n_rows]               (optional)
 * Problem b is X[b], centroids[b], labels[b], cluster_sizes[b], all contiguous.
 *
 * NOT IMPLEMENTED YET: the body is empty, so the outputs are left untouched.
 */
template <typename T,
          typename MathT,
          typename IdxT,
          typename LabelT,
          typename CounterT,
          typename MappingOpT>
void build_clusters_batched(
  const raft::resources& handle,
  const cuvs::cluster::kmeans::balanced_params& params,
  raft::device_mdspan<const T, raft::extent_3d<IdxT>, raft::row_major> X,
  raft::device_mdspan<MathT, raft::extent_3d<IdxT>, raft::row_major> centroids,
  raft::device_matrix_view<LabelT, IdxT> labels,
  raft::device_matrix_view<CounterT, IdxT> cluster_sizes,
  MappingOpT mapping_op,
  std::optional<raft::device_matrix_view<const MathT, IdxT>> X_norm = std::nullopt)
{
  // TODO: batched version of detail::build_clusters:
  //   1. initialize labels:           labels[b][i] = i % n_clusters   (for every problem b)
  //   2. calc_centers_and_sizes_batched
  //   3. balancing_em_iters_batched   (predict_batched + adjust_centers_batched per iteration)
}

/**
 * 2D layout: sub_trainset is [pq_n_rows, rot_dim], row-major. Subspace j of row r is the pq_len
 * values starting at sub_trainset[r * rot_dim + j * pq_len].
 */
template <typename IdxT>
void train_per_subset_batched_2d(raft::resources const& handle,
                                 owning_impl<IdxT>* impl,
                                 size_t n_rows,
                                 const float* trainset,   // [n_rows, dim]
                                 const uint32_t* labels,  // [n_rows]
                                 uint32_t kmeans_n_iters,
                                 uint32_t max_train_points_per_pq_code)
{
  auto stream        = raft::resource::get_cuda_stream(handle);
  auto device_memory = raft::resource::get_workspace_resource_ref(handle);

  rmm::device_uvector<float> pq_centers_tmp(impl->pq_centers().size(), stream, device_memory);
  // Subsampling the train set for codebook generation based on max_train_points_per_pq_code.
  size_t big_enough = max_train_points_per_pq_code * size_t(impl->pq_book_size());
  auto pq_n_rows    = uint32_t(std::min(big_enough, n_rows));
  // The residuals of all subspaces at once: [pq_n_rows, rot_dim] (was [pq_n_rows, pq_len]).
  rmm::device_uvector<float> sub_trainset(
    pq_n_rows * size_t(impl->rot_dim()), stream, device_memory);
  // One codeword label per (subspace, training row): [pq_dim, pq_n_rows] (was [pq_n_rows]).
  rmm::device_uvector<uint32_t> sub_labels(
    size_t(impl->pq_dim()) * pq_n_rows, stream, device_memory);

  // One size per (subspace, codeword): [pq_dim, pq_book_size] (was [pq_book_size]).
  rmm::device_uvector<uint32_t> pq_cluster_sizes(
    size_t(impl->pq_dim()) * impl->pq_book_size(), stream, device_memory);

  {
    raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> residuals_scope(
      "ivf_pq::build::residuals_2d");
    gpu_timer timer;
    timer.start(stream.get());

    // Fill: row r of sub_trainset = the whole rotated center centers_rot[labels[r], :].
    utils::copy_selected<float, float, size_t, uint32_t>(pq_n_rows,
                                                         impl->rot_dim(),
                                                         impl->centers_rot().data_handle(),
                                                         labels,
                                                         impl->rot_dim(),
                                                         sub_trainset.data(),
                                                         impl->rot_dim(),
                                                         stream);

    // sub_trainset = rotation * trainset^T - (what the fill put there), for all subspaces at once.
    float alpha = 1.0;
    float beta  = -1.0;
    raft::linalg::gemm(handle,
                       true,
                       false,
                       impl->rot_dim(),
                       pq_n_rows,
                       impl->dim(),
                       &alpha,
                       impl->rotation_matrix().data_handle(),
                       impl->dim(),
                       trainset,
                       impl->dim(),
                       &beta,
                       sub_trainset.data(),
                       impl->rot_dim(),
                       stream.get());

    float ms = timer.stop(stream.get());
    std::printf("[timing] residuals 2d: %.3f ms\n", ms);
  }

  // CHECK (PQ_EXP_CHECK=1): old per-subspace loop into ref[subspace][row][pos], then compare.
  const char* check_env = std::getenv("PQ_EXP_CHECK");
  if (check_env != nullptr && std::string(check_env) == "1") {
    const size_t pq_len = impl->pq_len();
    const size_t pq_dim = impl->pq_dim();
    const size_t rot    = impl->rot_dim();
    const size_t total  = size_t(pq_n_rows) * rot;
    rmm::device_uvector<float> ref(total, stream, device_memory);

    gpu_timer timer;
    timer.start(stream.get());
    for (uint32_t j = 0; j < impl->pq_dim(); j++) {
      float* ref_j = ref.data() + size_t(j) * pq_n_rows * pq_len;
      utils::copy_selected<float, float, size_t, uint32_t>(
        pq_n_rows,
        impl->pq_len(),
        impl->centers_rot().data_handle() + impl->pq_len() * j,
        labels,
        impl->rot_dim(),
        ref_j,
        impl->pq_len(),
        stream);
      float alpha = 1.0;
      float beta  = -1.0;
      raft::linalg::gemm(handle,
                         true,
                         false,
                         impl->pq_len(),
                         pq_n_rows,
                         impl->dim(),
                         &alpha,
                         impl->rotation_matrix().data_handle() + impl->dim() * impl->pq_len() * j,
                         impl->dim(),
                         trainset,
                         impl->dim(),
                         &beta,
                         ref_j,
                         impl->pq_len(),
                         stream.get());
    }
    float ref_ms = timer.stop(stream.get());

    std::vector<float> h_new(total), h_ref(total);
    raft::update_host(h_new.data(), sub_trainset.data(), total, stream);
    raft::update_host(h_ref.data(), ref.data(), total, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream.get()));

    double max_abs = 0;
    size_t n_bad   = 0;
    for (size_t subspace = 0; subspace < pq_dim; subspace++) {
      for (size_t row = 0; row < pq_n_rows; row++) {
        for (size_t pos = 0; pos < pq_len; pos++) {
          // 2d layout: [pq_n_rows, rot_dim]
          double x = h_new[row * rot + subspace * pq_len + pos];
          // old per-subspace layout: [pq_dim, pq_n_rows, pq_len]
          double y = h_ref[subspace * pq_n_rows * pq_len + row * pq_len + pos];
          double d = std::abs(x - y);
          max_abs  = std::max(max_abs, d);
          if (d > 1e-4 + 1e-4 * std::abs(y)) n_bad++;
        }
      }
    }
    std::printf("[timing] residuals old loop (reference): %.3f ms\n", ref_ms);
    std::printf("[compare] residuals 2d vs old: n=%zu  max_abs=%.3e  n_bad=%zu  -> %s\n",
                total,
                max_abs,
                n_bad,
                n_bad == 0 ? "MATCH" : "MISMATCH");
  }

  // STAGE 2+: batched k-means and transpose_pq_centers go here.
}

/**
 * 3D layout: sub_trainset is [pq_dim, pq_n_rows, pq_len], row-major. Subspace j is the contiguous
 * block [pq_n_rows, pq_len] starting at sub_trainset[j * pq_n_rows * pq_len].
 */
template <typename IdxT>
void train_per_subset_batched_3d(raft::resources const& handle,
                                 owning_impl<IdxT>* impl,
                                 size_t n_rows,
                                 const float* trainset,   // [n_rows, dim]
                                 const uint32_t* labels,  // [n_rows]
                                 uint32_t kmeans_n_iters,
                                 uint32_t max_train_points_per_pq_code)
{
  auto stream        = raft::resource::get_cuda_stream(handle);
  auto device_memory = raft::resource::get_workspace_resource_ref(handle);

  rmm::device_uvector<float> pq_centers_tmp(impl->pq_centers().size(), stream, device_memory);
  // Subsampling the train set for codebook generation based on max_train_points_per_pq_code.
  size_t big_enough = max_train_points_per_pq_code * size_t(impl->pq_book_size());
  auto pq_n_rows    = uint32_t(std::min(big_enough, n_rows));
  // The residuals of all subspaces at once: [pq_dim, pq_n_rows, pq_len]. Same number of floats as
  // [pq_n_rows, rot_dim] because rot_dim == pq_dim * pq_len.
  rmm::device_uvector<float> sub_trainset(
    pq_n_rows * size_t(impl->rot_dim()), stream, device_memory);
  // One codeword label per (subspace, training row): [pq_dim, pq_n_rows] (was [pq_n_rows]).
  rmm::device_uvector<uint32_t> sub_labels(
    size_t(impl->pq_dim()) * pq_n_rows, stream, device_memory);

  // One size per (subspace, codeword): [pq_dim, pq_book_size] (was [pq_book_size]).
  rmm::device_uvector<uint32_t> pq_cluster_sizes(
    size_t(impl->pq_dim()) * impl->pq_book_size(), stream, device_memory);

  {
    raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> residuals_scope(
      "ivf_pq::build::residuals_3d");
    gpu_timer timer;
    timer.start(stream.get());

    // Fill: sub_trainset[subspace, row, pos] = centers_rot[labels[row], subspace * pq_len + pos].
    // One launch over the whole array; the flat index i is split into (subspace, row, pos).
    const float* centers_rot = impl->centers_rot().data_handle();
    const size_t rot_dim_sz  = impl->rot_dim();
    const size_t pq_len_sz   = impl->pq_len();
    auto sub_trainset_flat   = raft::make_device_vector_view<float, int64_t>(
      sub_trainset.data(), int64_t(sub_trainset.size()));
    raft::linalg::map_offset(
      handle,
      sub_trainset_flat,
      [centers_rot, labels, rot_dim_sz, pq_len_sz, pq_n_rows] __device__(size_t i) {
        size_t pos      = i % pq_len_sz;                  // number inside the subspace (0..pq_len)
        size_t row      = (i / pq_len_sz) % pq_n_rows;    // training row
        size_t subspace = i / (pq_len_sz * pq_n_rows);    // which subspace (0..pq_dim)
        return centers_rot[size_t(labels[row]) * rot_dim_sz + subspace * pq_len_sz + pos];
      });

    // sub_trainset[j] = rotation_block_j * trainset^T - (what the fill put there), as one batched
    // gemm with batch = pq_dim. Block j of the rotation matrix starts dim * pq_len floats after
    // block j-1; every batch reads the same trainset (stride 0); output blocks are
    // pq_n_rows * pq_len floats apart.
    float alpha = 1.0;
    float beta  = -1.0;
    cublasStatus_t status = raft::linalg::detail::cublasgemmStridedBatched<float>(
      raft::resource::get_cublas_handle(handle),
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      int(impl->pq_len()),
      int(pq_n_rows),
      int(impl->dim()),
      &alpha,
      impl->rotation_matrix().data_handle(),
      int(impl->dim()),
      int64_t(impl->dim()) * int64_t(impl->pq_len()),
      trainset,
      int(impl->dim()),
      int64_t(0),
      &beta,
      sub_trainset.data(),
      int(impl->pq_len()),
      int64_t(pq_n_rows) * int64_t(impl->pq_len()),
      int(impl->pq_dim()),
      stream.get());
    RAFT_EXPECTS(status == CUBLAS_STATUS_SUCCESS, "strided batched gemm failed");

    float ms = timer.stop(stream.get());
    std::printf("[timing] residuals 3d: %.3f ms\n", ms);
  }

  // CHECK (PQ_EXP_CHECK=1): old per-subspace loop into ref[subspace][row][pos], then compare.
  // The 3d layout is the same as ref's layout, so positions match one to one.
  const char* check_env = std::getenv("PQ_EXP_CHECK");
  if (check_env != nullptr && std::string(check_env) == "1") {
    const size_t pq_len = impl->pq_len();
    const size_t total  = size_t(pq_n_rows) * impl->rot_dim();
    rmm::device_uvector<float> ref(total, stream, device_memory);

    gpu_timer timer;
    timer.start(stream.get());
    for (uint32_t j = 0; j < impl->pq_dim(); j++) {
      float* ref_j = ref.data() + size_t(j) * pq_n_rows * pq_len;
      utils::copy_selected<float, float, size_t, uint32_t>(
        pq_n_rows,
        impl->pq_len(),
        impl->centers_rot().data_handle() + impl->pq_len() * j,
        labels,
        impl->rot_dim(),
        ref_j,
        impl->pq_len(),
        stream);
      float alpha = 1.0;
      float beta  = -1.0;
      raft::linalg::gemm(handle,
                         true,
                         false,
                         impl->pq_len(),
                         pq_n_rows,
                         impl->dim(),
                         &alpha,
                         impl->rotation_matrix().data_handle() + impl->dim() * impl->pq_len() * j,
                         impl->dim(),
                         trainset,
                         impl->dim(),
                         &beta,
                         ref_j,
                         impl->pq_len(),
                         stream.get());
    }
    float ref_ms = timer.stop(stream.get());

    std::vector<float> h_new(total), h_ref(total);
    raft::update_host(h_new.data(), sub_trainset.data(), total, stream);
    raft::update_host(h_ref.data(), ref.data(), total, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream.get()));

    double max_abs = 0;
    size_t n_bad   = 0;
    for (size_t i = 0; i < total; i++) {
      double x = h_new[i];  // 3d layout
      double y = h_ref[i];  // old per-subspace layout (identical positions)
      double d = std::abs(x - y);
      max_abs  = std::max(max_abs, d);
      if (d > 1e-4 + 1e-4 * std::abs(y)) n_bad++;
    }
    std::printf("[timing] residuals old loop (reference): %.3f ms\n", ref_ms);
    std::printf("[compare] residuals 3d vs old: n=%zu  max_abs=%.3e  n_bad=%zu  -> %s\n",
                total,
                max_abs,
                n_bad,
                n_bad == 0 ? "MATCH" : "MISMATCH");
  }

  // Train the PQ codebooks of ALL subspaces at once. Every buffer holds the subspaces one after
  // the other, so each view has the subspace as its first dimension (the old loop made one 2D/1D
  // view per subspace j; compare with the views in train_per_subset_old).
  const int64_t n_batches = impl->pq_dim();
  auto sub_trainset_view =
    raft::make_mdspan<const float, internal_extents_t, raft::row_major, false, true>(
      sub_trainset.data(), raft::make_extents<internal_extents_t>(n_batches, pq_n_rows, impl->pq_len()));
  auto centers_tmp_view =
    raft::make_mdspan<float, internal_extents_t, raft::row_major, false, true>(
      pq_centers_tmp.data(),
      raft::make_extents<internal_extents_t>(n_batches, impl->pq_book_size(), impl->pq_len()));
  auto sub_labels_view = raft::make_device_matrix_view<uint32_t, internal_extents_t>(
    sub_labels.data(), n_batches, pq_n_rows);
  auto cluster_sizes_view = raft::make_device_matrix_view<uint32_t, internal_extents_t>(
    pq_cluster_sizes.data(), n_batches, impl->pq_book_size());
  cuvs::cluster::kmeans::balanced_params kmeans_params;
  kmeans_params.n_iters = kmeans_n_iters;
  kmeans_params.metric  = cuvs::distance::DistanceType::L2Expanded;
  // Preserve the historical donor selection behavior for IVF-PQ recall stability.
  kmeans_params.donor_selection         = cuvs::cluster::kmeans::balanced_donor_selection::Random;
  kmeans_params.balance_lower_tolerance = 0.25f;
  kmeans_params.balance_upper_tolerance = 4.0f;
  kmeans_params.centroid_offset         = 0.05f;
  RAFT_LOG_DEBUG("tolerance: %f, %f, centroid_offset: %f",
                 kmeans_params.balance_lower_tolerance,
                 kmeans_params.balance_upper_tolerance,
                 kmeans_params.centroid_offset);
  build_clusters_batched(handle,
                         kmeans_params,
                         sub_trainset_view,
                         centers_tmp_view,
                         sub_labels_view,
                         cluster_sizes_view,
                         utils::mapping<float>{});

  // TODO(transpose): once build_clusters_batched fills pq_centers_tmp, add the same final call
  // as at the end of train_per_subset():
  //     transpose_pq_centers(handle, impl, pq_centers_tmp.data());
  // pq_centers_tmp is already [pq_dim, book_size, pq_len], the layout that function expects
  // (it is what the old loop produced, subspace by subspace).
}

}  // namespace cuvs::neighbors::ivf_pq::detail

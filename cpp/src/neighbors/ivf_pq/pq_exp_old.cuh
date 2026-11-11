// Experiment harness, part 2: OLD / reference versions.
// Not part of the library. Included by pq_exp_ivf_pq_build.cuh (and pq_exp_main.cu).
#pragma once

#include "ivf_pq_build.cuh"

#include <optional>

namespace cuvs::neighbors::ivf_pq::detail {

// ---------------------------------------------------------------------------------------------
// build_clusters_old: COPY of the k-means entry point that train_per_subset() calls,
// cuvs::cluster::kmeans_balanced::helpers::build_clusters (kmeans_balanced_build_clusters_impl.cuh,
// approx. lines 44-82), with the body of cuvs::cluster::kmeans::detail::build_clusters
// (kmeans_balanced.cuh, approx. lines 1247-1298) inlined. Re-check line numbers against the tree.
// Same signature as the helpers:: version, so it can replace it in train_per_subset_old.
//
// EVERYTHING IS DISABLED (#if 0) so the old and the batched version can be compared step by step.
// Enable the steps IN ORDER (change `#if 0` to `#if 1`): each step needs the ones before it.
//   STEP 0: unpack the views and check shapes
//   STEP 1: initialize the labels ("randomly": i % n_clusters)
//   STEP 2: calc_centers_and_sizes (centers and sizes matching those labels)
//   STEP 3: balancing_em_iters (the EM loop: predict, adjust_centers, calc_centers_and_sizes)
// Nothing here is compiled while a step is disabled, so check that a step builds when enabling it.
// ---------------------------------------------------------------------------------------------
template <typename T,
          typename MathT,
          typename IdxT,
          typename LabelT,
          typename CounterT,
          typename MappingOpT>
void build_clusters_old(const raft::resources& handle,
                        const cuvs::cluster::kmeans::balanced_params& params,
                        raft::device_matrix_view<const T, IdxT> X,
                        raft::device_matrix_view<MathT, IdxT> centroids,
                        raft::device_vector_view<LabelT, IdxT> labels,
                        raft::device_vector_view<CounterT, IdxT> cluster_sizes,
                        MappingOpT mapping_op,
                        std::optional<raft::device_vector_view<const MathT>> X_norm = std::nullopt)
{
#if 0  // STEP 0: unpack the views and check shapes (from the helpers:: wrapper)
  IdxT n_rows     = X.extent(0);
  IdxT dim        = X.extent(1);
  IdxT n_clusters = centroids.extent(0);
  RAFT_EXPECTS(n_clusters > IdxT{0} && n_clusters <= n_rows,
               "The number of centroids must be strictly positive and cannot exceed the number of "
               "training samples");
  RAFT_EXPECTS(n_rows == labels.extent(0), "Number of rows in dataset and labels are different");
  RAFT_EXPECTS(dim == centroids.extent(1),
               "Number of features in dataset and centroids are different");
  RAFT_EXPECTS(!X_norm.has_value() || X_norm->extent(0) == n_rows,
               "Number of rows in dataset and norms are different");
  RAFT_EXPECTS(n_clusters == cluster_sizes.extent(0),
               "Number of rows in centroids and cluster_sizes are different");
  rmm::device_async_resource_ref device_memory = raft::resource::get_workspace_resource_ref(handle);
#endif

#if 0  // STEP 1: "randomly" initialize labels (detail::build_clusters)
  auto labels_view = raft::make_device_vector_view<LabelT, IdxT>(labels.data_handle(), n_rows);
  raft::linalg::map_offset(
    handle,
    labels_view,
    raft::compose_op(raft::cast_op<LabelT>(), raft::mod_const_op<IdxT>(n_clusters)));
#endif

#if 0  // STEP 2: update centers to match the initialized labels (detail::build_clusters)
  cuvs::cluster::kmeans::detail::calc_centers_and_sizes(handle,
                                                        centroids.data_handle(),
                                                        cluster_sizes.data_handle(),
                                                        n_clusters,
                                                        dim,
                                                        X.data_handle(),
                                                        n_rows,
                                                        labels.data_handle(),
                                                        true,
                                                        params.is_packed_binary,
                                                        mapping_op,
                                                        device_memory);
#endif

#if 0  // STEP 3: run EM (detail::build_clusters)
  cuvs::cluster::kmeans::detail::balancing_em_iters(
    handle,
    params,
    params.n_iters,
    dim,
    X.data_handle(),
    X_norm.has_value() ? X_norm.value().data_handle() : nullptr,
    n_rows,
    n_clusters,
    centroids.data_handle(),
    labels.data_handle(),
    cluster_sizes.data_handle(),
    2,
    static_cast<MathT>(params.balance_lower_tolerance),
    static_cast<MathT>(params.balance_upper_tolerance),
    mapping_op,
    device_memory);
#endif
}

// ---------------------------------------------------------------------------------------------
// COPY of train_per_subset() from ivf_pq_build.cuh (approx. lines 319-408; re-check line numbers
// against the current tree). Body is unchanged except for the name. This is the looped baseline
// that train_per_subset_batched is compared against.
//
// Later: truncate after the stage under test and add lines that save the intermediates (e.g. the
// per-subspace residuals) into caller-provided buffers.
// ---------------------------------------------------------------------------------------------
template <typename IdxT>
void train_per_subset_old(raft::resources const& handle,
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
  rmm::device_uvector<float> sub_trainset(
    pq_n_rows * size_t(impl->pq_len()), stream, device_memory);
  rmm::device_uvector<uint32_t> sub_labels(pq_n_rows, stream, device_memory);

  rmm::device_uvector<uint32_t> pq_cluster_sizes(impl->pq_book_size(), stream, device_memory);

  for (uint32_t j = 0; j < impl->pq_dim(); j++) {
    raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> pq_per_subspace_scope(
      "ivf_pq::build::per_subspace[%u]", j);

    // Get the rotated cluster centers for each training vector.
    // This will be subtracted from the input vectors afterwards.
    utils::copy_selected<float, float, size_t, uint32_t>(
      pq_n_rows,
      impl->pq_len(),
      impl->centers_rot().data_handle() + impl->pq_len() * j,
      labels,
      impl->rot_dim(),
      sub_trainset.data(),
      impl->pq_len(),
      stream);

    // sub_trainset is the slice of: rotate(trainset) - centers_rot
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
                       sub_trainset.data(),
                       impl->pq_len(),
                       stream.get());

    // train PQ codebook for this subspace
    auto sub_trainset_view = raft::make_device_matrix_view<const float, internal_extents_t>(
      sub_trainset.data(), pq_n_rows, impl->pq_len());
    auto centers_tmp_view = raft::make_device_matrix_view<float, internal_extents_t>(
      pq_centers_tmp.data() + impl->pq_book_size() * impl->pq_len() * j,
      impl->pq_book_size(),
      impl->pq_len());
    auto sub_labels_view =
      raft::make_device_vector_view<uint32_t, internal_extents_t>(sub_labels.data(), pq_n_rows);
    auto cluster_sizes_view = raft::make_device_vector_view<uint32_t, internal_extents_t>(
      pq_cluster_sizes.data(), impl->pq_book_size());
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
    build_clusters_old(handle,
                       kmeans_params,
                       sub_trainset_view,
                       centers_tmp_view,
                       sub_labels_view,
                       cluster_sizes_view,
                       utils::mapping<float>{});
  }
#if 0  // STAGE 2+: needs the codebooks from the k-means above
  transpose_pq_centers(handle, impl, pq_centers_tmp.data());
#endif
}

}  // namespace cuvs::neighbors::ivf_pq::detail

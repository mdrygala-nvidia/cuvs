// Experiment harness, part 1: copy of build(), timers, comparison helpers.
// Not part of the library. Included only by pq_exp_main.cu.
#pragma once

#include "ivf_pq_build.cuh"
#include "pq_exp_new.cuh"  // train_per_subset_batched_2d / _3d, gpu_timer
#include "pq_exp_old.cuh"  // train_per_subset_old

#include <raft/util/cudart_utils.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace cuvs::neighbors::ivf_pq::detail {

/** Which PER_SUBSPACE training build_batched runs. Chosen in the test with PQ_EXP_VARIANT. */
enum class pq_exp_variant {
  old_loop,    // train_per_subset_old: looped baseline (copy of the library version)
  batched_2d,  // train_per_subset_batched_2d: residuals as [pq_n_rows, rot_dim]
  batched_3d   // train_per_subset_batched_3d: residuals as [pq_dim, pq_n_rows, pq_len]
};

/** Experiment switches for build_batched. */
struct pq_exp_options {
  pq_exp_variant variant = pq_exp_variant::old_loop;
  // If non-null, build_batched writes the PQ-training time (ms) here and does not print it.
  float* train_ms = nullptr;
};

// ---------------------------------------------------------------------------------------------
// COPY of build() from ivf_pq_build.cuh (the overload that takes `dataset` and returns an index,
// approx. lines 1240-1399; re-check line numbers against the current tree). Body is unchanged
// except for the name (build -> build_batched), the NVTX labels, the options parameter, and the
// training switch (opts.variant picks train_per_subset_old / _batched_2d / _batched_3d;
// PER_CLUSTER still calls the library's train_per_cluster).
// ---------------------------------------------------------------------------------------------
template <typename T, typename IdxT, typename accessor>
auto build_batched(
  raft::resources const& handle,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<IdxT>, raft::row_major, accessor> dataset,
  const pq_exp_options& opts = {}) -> index<IdxT>
{
  IdxT n_rows = dataset.extent(0);
  IdxT dim    = dataset.extent(1);
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "ivf_pq::build_batched(%zu, %u)", size_t(n_rows), dim);
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, uint8_t> ||
                  std::is_same_v<T, int8_t>,
                "Unsupported data type");

  RAFT_EXPECTS(n_rows > 0 && dim > 0, "empty dataset");
  RAFT_EXPECTS(n_rows >= params.n_lists, "number of rows can't be less than n_lists");

  auto impl = std::make_unique<cuvs::neighbors::ivf_pq::owning_impl<IdxT>>(
    handle,
    params.metric,
    params.codebook_kind,
    params.n_lists,
    dim,
    params.pq_bits,
    params.pq_dim == 0 ? index<IdxT>::calculate_pq_dim(dim) : params.pq_dim,
    params.conservative_memory_allocation,
    params.codes_layout);

  auto stream = raft::resource::get_cuda_stream(handle);
  utils::memzero(
    impl->accum_sorted_sizes().data_handle(), impl->accum_sorted_sizes().size(), stream);
  utils::memzero(impl->list_sizes().data_handle(), impl->list_sizes().size(), stream);
  utils::memzero(impl->data_ptrs().data_handle(), impl->data_ptrs().size(), stream);
  utils::memzero(impl->inds_ptrs().data_handle(), impl->inds_ptrs().size(), stream);

  {
    raft::random::RngState random_state{137};
    auto trainset_ratio = std::max<size_t>(
      1,
      size_t(n_rows) / std::max<size_t>(params.kmeans_trainset_fraction * n_rows, impl->n_lists()));
    size_t n_rows_train = n_rows / trainset_ratio;

    rmm::device_async_resource_ref device_memory =
      raft::resource::get_workspace_resource_ref(handle);

    // If the trainset is small enough to comfortably fit into device memory, put it there.
    // Otherwise, use the managed memory.
    constexpr size_t kTolerableRatio = 4;
    rmm::device_async_resource_ref big_memory_resource =
      raft::resource::get_large_workspace_resource_ref(handle);
    if (sizeof(float) * n_rows_train * impl->dim() * kTolerableRatio <
        raft::resource::get_workspace_free_bytes(handle)) {
      big_memory_resource = device_memory;
    }

    // Besides just sampling, we transform the input dataset into floats to make it easier
    // to use gemm operations from cublas.
    auto trainset = raft::make_device_mdarray<float>(
      handle, big_memory_resource, raft::make_extents<int64_t>(0, 0));
    try {
      trainset = raft::make_device_mdarray<float>(
        handle, big_memory_resource, raft::make_extents<int64_t>(n_rows_train, dim));
    } catch (raft::logic_error& e) {
      RAFT_LOG_ERROR(
        "Insufficient memory for kmeans training set allocation. Please decrease "
        "kmeans_trainset_fraction, or set large_workspace_resource appropriately.");
      throw;
    }
    // TODO: a proper sampling
    if constexpr (std::is_same_v<T, float>) {
      raft::matrix::sample_rows<T, int64_t>(handle, random_state, dataset, trainset.view());
    } else {
      raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
        "   ivf_pq::build_batched(%zu, %zu)/sample rows with tmp trainset (%zu rows).",
        size_t(n_rows),
        size_t(dim),
        size_t(n_rows_train));

      // TODO(tfeher): Enable codebook generation with any type T, and then remove trainset tmp.
      auto trainset_tmp = raft::make_device_mdarray<T>(
        handle, big_memory_resource, raft::make_extents<int64_t>(n_rows_train, dim));

      raft::matrix::sample_rows<T, int64_t>(handle, random_state, dataset, trainset_tmp.view());

      raft::linalg::map(handle,
                        raft::make_device_vector_view<float, int64_t>(trainset.data_handle(),
                                                                      (int64_t)trainset.size()),
                        utils::mapping<float>{},
                        raft::make_const_mdspan(raft::make_device_vector_view<const T, int64_t>(
                          trainset_tmp.data_handle(), (int64_t)trainset.size())));
    }

    // NB: here cluster_centers is used as if it is [n_clusters, data_dim] not [n_clusters,
    // dim_ext]!
    rmm::device_uvector<float> cluster_centers_buf(
      impl->n_lists() * impl->dim(), stream, device_memory);
    auto cluster_centers = cluster_centers_buf.data();

    // Train balanced hierarchical kmeans clustering
    auto trainset_const_view = raft::make_const_mdspan(trainset.view());
    auto centers_view        = raft::make_device_matrix_view<float, internal_extents_t>(
      cluster_centers, impl->n_lists(), impl->dim());
    cuvs::cluster::kmeans::balanced_params kmeans_params;
    kmeans_params.n_iters = params.kmeans_n_iters;
    kmeans_params.metric  = coarse_clustering_metric(impl->metric());
    // Use the default SizeSorted donor selection to prioritize balanced coarse partitions.
    if (impl->metric() == distance::DistanceType::CosineExpanded) {
      raft::linalg::row_normalize<raft::linalg::L2Norm>(
        handle, trainset_const_view, trainset.view());
    }
    cuvs::cluster::kmeans::fit(handle, kmeans_params, trainset_const_view, centers_view);

    // Trainset labels are needed for training PQ codebooks
    rmm::device_uvector<uint32_t> labels(n_rows_train, stream, big_memory_resource);
    auto centers_const_view = raft::make_device_matrix_view<const float, internal_extents_t>(
      cluster_centers, impl->n_lists(), impl->dim());
    if (impl->metric() == distance::DistanceType::CosineExpanded) {
      raft::linalg::row_normalize<raft::linalg::L2Norm>(handle, centers_const_view, centers_view);
    }
    auto labels_view =
      raft::make_device_vector_view<uint32_t, internal_extents_t>(labels.data(), n_rows_train);
    cuvs::cluster::kmeans::predict(
      handle, kmeans_params, trainset_const_view, centers_const_view, labels_view);

    // Make rotation matrix
    helpers::make_rotation_matrix(handle, impl->rotation_matrix(), params.force_random_rotation);

    set_centers(handle, impl.get(), cluster_centers);

    // Train PQ codebooks
    // Time only the PQ codebook training (not sampling / coarse k-means / rotation).
    gpu_timer train_timer;
    train_timer.start(stream.get());

    // EXPERIMENT: opts.variant selects which PER_SUBSPACE training runs
    // (PER_CLUSTER always uses the library's train_per_cluster).
    switch (impl->codebook_kind()) {
      case codebook_gen::PER_SUBSPACE:
        switch (opts.variant) {
          case pq_exp_variant::old_loop:
            train_per_subset_old(handle,
                                 impl.get(),
                                 n_rows_train,
                                 trainset.data_handle(),
                                 labels.data(),
                                 params.kmeans_n_iters,
                                 params.max_train_points_per_pq_code);
            break;
          case pq_exp_variant::batched_2d:
            train_per_subset_batched_2d(handle,
                                        impl.get(),
                                        n_rows_train,
                                        trainset.data_handle(),
                                        labels.data(),
                                        params.kmeans_n_iters,
                                        params.max_train_points_per_pq_code);
            break;
          case pq_exp_variant::batched_3d:
            train_per_subset_batched_3d(handle,
                                        impl.get(),
                                        n_rows_train,
                                        trainset.data_handle(),
                                        labels.data(),
                                        params.kmeans_n_iters,
                                        params.max_train_points_per_pq_code);
            break;
        }
        break;
      case codebook_gen::PER_CLUSTER:
        train_per_cluster(handle,
                          impl.get(),
                          n_rows_train,
                          trainset.data_handle(),
                          labels.data(),
                          params.kmeans_n_iters,
                          params.max_train_points_per_pq_code);
        break;
      default: RAFT_FAIL("Unreachable code");
    }

    float train_ms = train_timer.stop(stream.get());
    if (opts.train_ms != nullptr) {
      *opts.train_ms = train_ms;
    } else {
      std::printf("[train] %s: %.1f ms\n",
                  opts.variant == pq_exp_variant::old_loop
                    ? "old"
                    : (opts.variant == pq_exp_variant::batched_2d ? "2d" : "3d"),
                  train_ms);
    }
  }
  index<IdxT> idx(std::move(impl));

  // add the data if necessary
  if (params.add_data_on_build) {
    detail::extend<T, IdxT>(handle, &idx, dataset.data_handle(), nullptr, n_rows);
  }
  return idx;
}

// ---------------------------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------------------------

/**
 * Runs fn() `n_warmup` times (not timed: first calls include allocation / kernel loading), then
 * `n_iters` timed times. Returns the median in milliseconds.
 */
template <typename Fn>
inline float time_ms(raft::resources const& handle,
                     Fn&& fn,
                     int n_warmup = 1,
                     int n_iters  = 5)
{
  // get_cuda_stream() returns a cuda::stream_ref; the raw CUDA calls below need cudaStream_t
  cudaStream_t stream = raft::resource::get_cuda_stream(handle).get();
  for (int i = 0; i < n_warmup; i++) {
    fn();
  }
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  std::vector<float> times;
  gpu_timer timer;
  for (int i = 0; i < n_iters; i++) {
    timer.start(stream);
    fn();
    times.push_back(timer.stop(stream));
  }
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

// ---------------------------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------------------------

struct compare_result {
  double max_abs = 0;  // largest |a - b|
  double max_rel = 0;  // largest |a - b| / max(|b|, tiny)
  size_t n_bad   = 0;  // elements with |a - b| > atol + rtol * |b|
  size_t n       = 0;
};

/** Compares two device buffers of n elements. a = new result, b = reference. */
template <typename T>
inline auto compare_device(raft::resources const& handle,
                           const T* a,
                           const T* b,
                           size_t n,
                           double atol = 1e-5,
                           double rtol = 1e-4) -> compare_result
{
  auto stream = raft::resource::get_cuda_stream(handle);
  std::vector<T> ha(n), hb(n);
  raft::update_host(ha.data(), a, n, stream);
  raft::update_host(hb.data(), b, n, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream.get()));

  compare_result r;
  r.n = n;
  for (size_t i = 0; i < n; i++) {
    double x = static_cast<double>(ha[i]);
    double y = static_cast<double>(hb[i]);
    double d = std::abs(x - y);
    r.max_abs = std::max(r.max_abs, d);
    r.max_rel = std::max(r.max_rel, d / std::max(std::abs(y), 1e-30));
    if (d > atol + rtol * std::abs(y)) r.n_bad++;
  }
  return r;
}

inline void print_compare(const char* stage, const compare_result& r)
{
  std::printf("[compare] %-28s n=%zu  max_abs=%.3e  max_rel=%.3e  n_bad=%zu  -> %s\n",
              stage,
              r.n,
              r.max_abs,
              r.max_rel,
              r.n_bad,
              r.n_bad == 0 ? "MATCH" : "MISMATCH");
}

inline void print_timing(const char* stage, float old_ms, float new_ms)
{
  std::printf("[timing]  %-28s old=%8.3f ms  new=%8.3f ms  speedup=%.2fx\n",
              stage,
              old_ms,
              new_ms,
              new_ms > 0 ? old_ms / new_ms : 0.0f);
}

}  // namespace cuvs::neighbors::ivf_pq::detail

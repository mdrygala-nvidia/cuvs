// Experiment harness for batching train_per_subset / train_per_cluster.
// Not part of the library: built only as the PQ_EXPERIMENT test target
// (see the PQ_EXPERIMENT block at the end of cpp/tests/CMakeLists.txt).
//
// Run:
//   ./cpp/build/gtests/PQ_EXPERIMENT                         # old looped baseline
//   PQ_EXP_VARIANT=2d ./cpp/build/gtests/PQ_EXPERIMENT       # residuals as [pq_n_rows, rot_dim]
//   PQ_EXP_VARIANT=3d ./cpp/build/gtests/PQ_EXPERIMENT       # residuals as [pq_dim, pq_n_rows, pq_len]
//
// Environment variables:
//   PQ_EXP_VARIANT=old|2d|3d   which PER_SUBSPACE training to run (default: old)
//   PQ_EXP_REPEATS=N           number of timed runs (default 5)
//   PQ_EXP_WARMUP=N            number of untimed warm-up runs before them (default 1)
//   PQ_EXP_CHECK=1             2d/3d only: also run the old per-subspace residual loop, time it and
//                              compare the results (use with PQ_EXP_REPEATS=1)
//
// Only output lines starting with [train], [build], [summary], [timing], [compare] are ours:
//   ./cpp/build/gtests/PQ_EXPERIMENT | grep -E '^\[(train|build|summary|timing|compare)\]'

#include <gtest/gtest.h>

#include "pq_exp_ivf_pq_build.cuh"  // build_batched, timers, compare helpers
#include "pq_exp_new.cuh"           // new batched versions
#include "pq_exp_old.cuh"           // old / reference versions

#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng.cuh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace pqd = cuvs::neighbors::ivf_pq::detail;

namespace {

int env_int(const char* name, int default_value)
{
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return default_value;
  return std::atoi(v);
}

struct stats {
  double mean = 0, stddev = 0, median = 0, min = 0, max = 0;
};

stats compute_stats(std::vector<float> v)
{
  stats s;
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  double sum = 0;
  for (float x : v) {
    sum += x;
  }
  s.mean = sum / v.size();
  double sq = 0;
  for (float x : v) {
    sq += (x - s.mean) * (x - s.mean);
  }
  s.stddev = v.size() > 1 ? std::sqrt(sq / (v.size() - 1)) : 0.0;
  s.median = v[v.size() / 2];
  s.min    = v.front();
  s.max    = v.back();
  return s;
}

void print_stats(const char* what, const char* variant, const stats& s, int n, int warmup)
{
  std::printf(
    "[summary] %-6s %-7s mean=%7.1f ms  std=%5.1f  median=%7.1f  min=%7.1f  max=%7.1f  "
    "(n=%d, warmup=%d)\n",
    what,
    variant,
    s.mean,
    s.stddev,
    s.median,
    s.min,
    s.max,
    n,
    warmup);
}

// Builds an index on a toy dataset several times and reports timing statistics.
void run_build(pqd::pq_exp_variant variant, const char* variant_name, int n_warmup, int n_repeats)
{
  raft::resources handle;

  // toy dataset: random floats on the device
  int64_t n_rows = 100000, dim = 128;
  auto dataset = raft::make_device_matrix<float, int64_t>(handle, n_rows, dim);
  raft::random::RngState rng(1234);
  raft::random::normal(handle, rng, dataset.data_handle(), n_rows * dim, 0.0f, 1.0f);

  // params
  cuvs::neighbors::ivf_pq::index_params params;
  params.n_lists           = 256;
  params.pq_dim            = 32;
  params.pq_bits           = 8;
  params.kmeans_n_iters    = 20;
  params.codebook_kind     = cuvs::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE;
  params.metric            = cuvs::distance::DistanceType::L2Expanded;
  params.add_data_on_build = false;  // we only care about training the codebooks

  cudaStream_t stream = raft::resource::get_cuda_stream(handle).get();

  std::vector<float> train_times, build_times;
  for (int i = 0; i < n_warmup + n_repeats; i++) {
    float train_ms = 0;
    pqd::pq_exp_options opts;
    opts.variant  = variant;
    opts.train_ms = &train_ms;  // build_batched stores the PQ-training time here

    pqd::gpu_timer timer;
    timer.start(stream);
    auto idx = pqd::build_batched<float, int64_t>(
      handle, params, raft::make_const_mdspan(dataset.view()), opts);
    float build_ms = timer.stop(stream);

    bool is_warmup = i < n_warmup;
    std::printf("[train] %s run %d%s: %.1f ms   [build] %.1f ms\n",
                variant_name,
                i,
                is_warmup ? " (warmup)" : "",
                train_ms,
                build_ms);
    if (!is_warmup) {
      train_times.push_back(train_ms);
      build_times.push_back(build_ms);
    }
  }

  print_stats("train", variant_name, compute_stats(train_times), n_repeats, n_warmup);
  print_stats("build", variant_name, compute_stats(build_times), n_repeats, n_warmup);
}

}  // namespace

// Variant is selected with PQ_EXP_VARIANT=old|2d|3d (default: old).
TEST(PqExperiment, Build)
{
  const char* env  = std::getenv("PQ_EXP_VARIANT");
  std::string name = (env != nullptr && *env != '\0') ? env : "old";
  pqd::pq_exp_variant variant;
  if (name == "old") {
    variant = pqd::pq_exp_variant::old_loop;
  } else if (name == "2d") {
    variant = pqd::pq_exp_variant::batched_2d;
  } else if (name == "3d") {
    variant = pqd::pq_exp_variant::batched_3d;
  } else {
    FAIL() << "PQ_EXP_VARIANT must be old, 2d or 3d (got '" << name << "')";
  }
  int n_repeats = std::max(1, env_int("PQ_EXP_REPEATS", 5));
  int n_warmup  = std::max(0, env_int("PQ_EXP_WARMUP", 1));
  run_build(variant, name.c_str(), n_warmup, n_repeats);
}

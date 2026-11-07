// dmk_host.hpp -- host/device-agnostic helpers shared by the B2-B4 benchmarks.
//
// Pure C++17 when compiled by g++ (so benchmarks/src/test_host.cpp can test it on a machine
// without a GPU); the same functions become __host__ __device__ under nvcc.
// Everything here is deterministic given a seed.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef __CUDACC__
#define DMK_HD __host__ __device__ __forceinline__
#else
#define DMK_HD inline
#endif

namespace dmk {

DMK_HD uint64_t splitmix64_mix(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Same definition as gen_value() in copy_stride.cu: value depends only on (i, seed).
DMK_HD uint32_t gen_value(uint64_t i, uint32_t seed) {
  uint64_t z = (i + 1) * 0x9E3779B97F4A7C15ull ^ (static_cast<uint64_t>(seed) << 32 | seed);
  z = splitmix64_mix(z);
  return static_cast<uint32_t>(z ^ (z >> 32));
}

// A finite, normal fp32 in [1, 2) with 23 pseudo-random mantissa bits. Used where data is
// interpreted as float (cuBLAS geam, the fusion pipeline): no NaN/Inf/denormal surprises.
DMK_HD uint32_t gen_unit_float_bits(uint64_t i, uint32_t seed) {
  return (gen_value(i, seed) & 0x007FFFFFu) | 0x3F800000u;
}

// ---------------------------------------------------------------------------
// Host-only RNG and index generators
// ---------------------------------------------------------------------------
struct Rng {
  uint64_t state;
  explicit Rng(uint64_t seed) : state(seed) {}
  uint64_t next() {
    state += 0x9E3779B97F4A7C15ull;
    return splitmix64_mix(state);
  }
  // Uniform double in [0, 1) with 53 random bits.
  double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
  // Uniform integer in [0, n), Lemire multiply-shift (bias < n / 2^64, negligible here).
  uint32_t below(uint32_t n) {
    return static_cast<uint32_t>((static_cast<unsigned __int128>(next()) * n) >> 64);
  }
};

// Fisher-Yates permutation of [0, n).
inline std::vector<uint32_t> random_permutation(uint32_t n, uint64_t seed) {
  std::vector<uint32_t> p(n);
  for (uint32_t i = 0; i < n; ++i) p[i] = i;
  Rng r(seed);
  for (uint32_t i = n; i > 1; --i) std::swap(p[i - 1], p[r.below(i)]);
  return p;
}

// Zipf(alpha) over ranks [0, n): P(rank = k) proportional to 1 / (k + 1)^alpha.
// Sampled by inverse CDF (exact up to double rounding). alpha = 0 gives uniform.
class ZipfSampler {
 public:
  ZipfSampler(uint32_t n, double alpha) : cdf_(n) {
    if (n == 0) throw std::invalid_argument("ZipfSampler: n == 0");
    if (alpha < 0) throw std::invalid_argument("ZipfSampler: alpha < 0");
    double acc = 0;
    for (uint32_t k = 0; k < n; ++k) {
      acc += std::pow(static_cast<double>(k) + 1.0, -alpha);
      cdf_[k] = acc;
    }
    for (double& c : cdf_) c /= acc;
    cdf_.back() = 1.0;
  }
  uint32_t sample(Rng& r) const {
    const double u = r.uniform();
    return static_cast<uint32_t>(std::upper_bound(cdf_.begin(), cdf_.end(), u) - cdf_.begin());
  }
  double pmf(uint32_t k) const { return k == 0 ? cdf_[0] : cdf_[k] - cdf_[k - 1]; }
  uint32_t size() const { return static_cast<uint32_t>(cdf_.size()); }

 private:
  std::vector<double> cdf_;
};

enum class IndexDist { Seq, Uniform, Zipf };

struct IndexSpec {
  IndexDist dist = IndexDist::Seq;
  double alpha = 0.0;  // only for Zipf
};

// Parse "seq", "uniform", or "zipf<alpha>" (e.g. "zipf1.0").
inline IndexSpec parse_index_spec(const char* s) {
  IndexSpec spec;
  if (std::strcmp(s, "seq") == 0) {
    spec.dist = IndexDist::Seq;
  } else if (std::strcmp(s, "uniform") == 0) {
    spec.dist = IndexDist::Uniform;
  } else if (std::strncmp(s, "zipf", 4) == 0 && s[4] != '\0') {
    spec.dist = IndexDist::Zipf;
    char* end = nullptr;
    spec.alpha = std::strtod(s + 4, &end);
    if (end == s + 4 || *end != '\0' || spec.alpha < 0)
      throw std::invalid_argument("bad zipf alpha");
  } else {
    throw std::invalid_argument("unknown index distribution");
  }
  return spec;
}

// Generate m indices into a table of n elements.
// For Zipf, ranks are mapped to table slots through a random permutation so that hot
// elements are scattered across the table (otherwise the hottest elements would already be
// packed into the first few sectors, which is the *optimization* H3 wants to test).
inline std::vector<uint32_t> make_indices(const IndexSpec& spec, uint32_t n, size_t m,
                                          uint64_t seed) {
  std::vector<uint32_t> idx(m);
  switch (spec.dist) {
    case IndexDist::Seq:
      for (size_t i = 0; i < m; ++i) idx[i] = static_cast<uint32_t>(i % n);
      break;
    case IndexDist::Uniform: {
      Rng r(seed);
      for (size_t i = 0; i < m; ++i) idx[i] = r.below(n);
      break;
    }
    case IndexDist::Zipf: {
      ZipfSampler z(n, spec.alpha);
      const std::vector<uint32_t> perm = random_permutation(n, seed ^ 0xA5A5A5A5ull);
      Rng r(seed);
      for (size_t i = 0; i < m; ++i) idx[i] = perm[z.sample(r)];
      break;
    }
  }
  return idx;
}

}  // namespace dmk

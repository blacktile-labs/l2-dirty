// test_host.cpp -- CPU-only tests for dmk_host.hpp (no GPU, no CUDA needed).
// Build & run: bash benchmarks/scripts/test_host_cpp.sh
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "dmk_host.hpp"
#include "stencil_host.hpp"

static int g_fail = 0;
#define CHECK(cond, ...)                                       \
  do {                                                         \
    if (!(cond)) {                                             \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__);         \
      std::printf(__VA_ARGS__);                                \
      std::printf("\n");                                       \
      ++g_fail;                                                \
    }                                                          \
  } while (0)

static void test_determinism() {
  dmk::Rng a(42), b(42), c(43);
  bool same = true, differs = false;
  for (int i = 0; i < 1000; ++i) {
    uint64_t x = a.next(), y = b.next(), z = c.next();
    same &= (x == y);
    differs |= (x != z);
  }
  CHECK(same, "same seed must give the same stream");
  CHECK(differs, "different seeds must give different streams");
  CHECK(dmk::gen_value(5, 7) == dmk::gen_value(5, 7), "gen_value deterministic");
  CHECK(dmk::gen_value(5, 7) != dmk::gen_value(6, 7), "gen_value depends on index");
}

static void test_unit_float_bits() {
  for (uint64_t i = 0; i < 100000; ++i) {
    uint32_t b = dmk::gen_unit_float_bits(i, 3);
    float f;
    std::memcpy(&f, &b, 4);
    if (!(f >= 1.0f && f < 2.0f)) {
      CHECK(false, "gen_unit_float_bits out of [1,2): i=%llu f=%g", (unsigned long long)i, f);
      return;
    }
  }
}

static void test_below_range_and_balance() {
  dmk::Rng r(1);
  const uint32_t n = 10;
  std::vector<int> h(n, 0);
  const int m = 1000000;
  for (int i = 0; i < m; ++i) {
    uint32_t v = r.below(n);
    if (v >= n) { CHECK(false, "below() out of range"); return; }
    ++h[v];
  }
  for (uint32_t k = 0; k < n; ++k)
    CHECK(std::abs(h[k] - m / 10) < 1500, "below() bucket %u count %d far from %d", k, h[k], m / 10);
}

static void test_permutation_is_bijection() {
  const uint32_t n = 100003;
  auto p = dmk::random_permutation(n, 9);
  std::vector<char> seen(n, 0);
  for (uint32_t v : p) {
    if (v >= n || seen[v]) { CHECK(false, "not a bijection at %u", v); return; }
    seen[v] = 1;
  }
  CHECK(dmk::random_permutation(n, 9) == p, "permutation deterministic");
}

static void test_zipf_frequencies() {
  // n = 1000, alpha = 1.0, 2e6 samples: empirical frequency of the first ranks must match
  // the pmf within 5 standard errors.
  const uint32_t n = 1000;
  const double alpha = 1.0;
  dmk::ZipfSampler z(n, alpha);
  double h = 0;
  for (uint32_t k = 1; k <= n; ++k) h += 1.0 / k;
  CHECK(std::fabs(z.pmf(0) - 1.0 / h) < 1e-12, "pmf(0) %.12f vs %.12f", z.pmf(0), 1.0 / h);
  dmk::Rng r(123);
  const int m = 2000000;
  std::vector<int> cnt(n, 0);
  for (int i = 0; i < m; ++i) ++cnt[z.sample(r)];
  for (uint32_t k = 0; k < 20; ++k) {
    const double p = z.pmf(k), se = std::sqrt(p * (1 - p) / m), f = double(cnt[k]) / m;
    CHECK(std::fabs(f - p) < 5 * se, "zipf rank %u freq %.5f vs pmf %.5f", k, f, p);
  }
  // alpha = 0 must be uniform
  dmk::ZipfSampler u(n, 0.0);
  CHECK(std::fabs(u.pmf(0) - 1.0 / n) < 1e-12 && std::fabs(u.pmf(n - 1) - 1.0 / n) < 1e-12,
        "alpha=0 must be uniform");
}

static void test_make_indices() {
  const uint32_t n = 4096;
  auto s = dmk::make_indices(dmk::parse_index_spec("seq"), n, 10000, 1);
  CHECK(s[0] == 0 && s[4095] == 4095 && s[4096] == 0, "seq wraps modulo n");
  auto zi = dmk::make_indices(dmk::parse_index_spec("zipf1.2"), n, 200000, 5);
  bool in_range = true;
  for (uint32_t v : zi) in_range &= (v < n);
  CHECK(in_range, "zipf indices in range");
  // Hot items must be scattered: the most frequent index should not simply be 0.
  std::vector<int> c(n, 0);
  for (uint32_t v : zi) ++c[v];
  int arg = 0;
  for (uint32_t k = 1; k < n; ++k) if (c[k] > c[arg]) arg = k;
  CHECK(arg != 0, "hottest zipf element should be permuted away from slot 0 (got %d)", arg);
  CHECK(dmk::make_indices(dmk::parse_index_spec("uniform"), n, 1000, 3) ==
            dmk::make_indices(dmk::parse_index_spec("uniform"), n, 1000, 3),
        "uniform indices deterministic");
  bool threw = false;
  try { dmk::parse_index_spec("zipf"); } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw, "parse_index_spec rejects 'zipf' without alpha");
  threw = false;
  try { dmk::parse_index_spec("random"); } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw, "parse_index_spec rejects unknown names");
}


static void test_stencil_composition_exact() {
  // Two 5-point sweeps == one 13-point sweep with self-convolved coefficients, bit-exact on a
  // torus, including tiny grids where the radius-2 diamond wraps onto itself.
  const dmk::Coef13 k13 = dmk::compose_twice(dmk::kStencil5);
  uint64_t sum = 0;
  for (uint32_t v : k13.v) sum += v;
  // sum of 13-pt coefficients == (sum of 5-pt coefficients)^2 = 39^2
  CHECK(sum == 39ull * 39ull, "coefficient sum %llu != 1521", (unsigned long long)sum);
  const size_t shapes[][2] = {{8, 32}, {5, 7}, {3, 3}, {1, 4}, {64, 1024}, {17, 33}};
  for (auto& sh : shapes) {
    const size_t H = sh[0], W = sh[1];
    std::vector<uint32_t> x(H * W), a, b, c;
    for (size_t i = 0; i < x.size(); ++i) x[i] = dmk::gen_value(i, 99);
    dmk::sweep5_host(x, a, H, W);
    dmk::sweep5_host(a, b, H, W);
    dmk::sweep13_host(x, c, H, W, k13);
    CHECK(b == c, "2x sweep5 != sweep13 on %zux%zu", H, W);
  }
  // Index sanity: a delta at (r0, c0) must spread to exactly the 5-point pattern.
  const size_t H = 8, W = 32, r0 = 3, c0 = 10;
  std::vector<uint32_t> d(H * W, 0), o;
  d[r0 * W + c0] = 1;
  dmk::sweep5_host(d, o, H, W);
  // out[r][c] picks in[r-1][c] with weight N, so the delta at r0 appears at r0+1 with weight N.
  CHECK(o[r0 * W + c0] == dmk::kStencil5.c && o[(r0 + 1) * W + c0] == dmk::kStencil5.n &&
            o[(r0 - 1) * W + c0] == dmk::kStencil5.s && o[r0 * W + c0 + 1] == dmk::kStencil5.w &&
            o[r0 * W + c0 - 1] == dmk::kStencil5.e,
        "delta response does not match the 5-point pattern");
}

int main() {
  test_determinism();
  test_unit_float_bits();
  test_below_range_and_balance();
  test_permutation_is_bijection();
  test_zipf_frequencies();
  test_make_indices();
  test_stencil_composition_exact();
  if (g_fail == 0) std::printf("test_host: all checks passed\n");
  return g_fail == 0 ? 0 : 1;
}

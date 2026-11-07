// fusion.cu -- Gate B baseline #4: materialized vs. fused two-stage pointwise pipeline
//
//   t = f(x);  y = g(t)        x, t, y: n fp32 values
//   f(v): K iterations of v = fma(v, 0.999, 0.001)
//   g(v): K iterations of v = fma(v, 1.0001, -0.0001)
//
// The question is a data-movement one: materializing t costs "write t + read t" bytes that the
// algorithm does not need -- but *at which tier*? If t stays in L2 between the two kernels the
// extra bytes never reach DRAM. Array sizes are therefore given as multiples of the L2 size
// measured at runtime, and K sweeps arithmetic intensity toward the compute roof.
//
// Variants (per n, K):
//   fused            one kernel x -> y
//   materialized     stage1 (x -> t) then stage2 (t -> y), timed together
//   stage2_cold      stage2 alone after an L2 flush            (t comes from DRAM)
//   stage2_warm      stage2 alone right after an untimed stage1 (t possibly still in L2)
// Each is timed "isolated" (flush before each rep) and "steady" (back-to-back calls; deferred
// write-backs of dirty lines land inside the timed window).
// Results are compared bit-for-bit with a host reference (fmaf is correctly rounded on both).

#include "common.cuh"

#include <cmath>

using namespace dmk;

DMK_HD float stage_f(float v, int k) {
  for (int i = 0; i < k; ++i) v = fmaf(v, 0.999f, 0.001f);
  return v;
}
DMK_HD float stage_g(float v, int k) {
  for (int i = 0; i < k; ++i) v = fmaf(v, 1.0001f, -0.0001f);
  return v;
}

__global__ void stage1_kernel(const float* __restrict__ x, float* __restrict__ t, size_t n, int k) {
  DMK_GRID_STRIDE_LOOP(i, n) t[i] = stage_f(x[i], k);
}
__global__ void stage2_kernel(const float* __restrict__ t, float* __restrict__ y, size_t n, int k) {
  DMK_GRID_STRIDE_LOOP(i, n) y[i] = stage_g(t[i], k);
}
__global__ void fused_kernel(const float* __restrict__ x, float* __restrict__ y, size_t n, int k) {
  DMK_GRID_STRIDE_LOOP(i, n) y[i] = stage_g(stage_f(x[i], k), k);
}

static uint32_t bits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return b;
}
static float from_bits(uint32_t b) {
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}

int main(int argc, char** argv) {
  Args a(argc, argv, {"--smoke", "--no-flush-l2"},
         {"--sizes-l2x", "--ks", "--reps", "--warmup", "--steady-iters", "--seed", "--block", "--device",
          "--csv", "--meta"});
  if (a.help()) {
    std::printf("Usage: fusion [--smoke] [--sizes-l2x F,...] [--ks K,...] [--reps N] [--warmup N]\n"
                "              [--steady-iters N] [--seed N] [--block N] [--no-flush-l2] [--device N]\n"
                "              [--csv PATH] [--meta PATH]\n"
                "  --sizes-l2x: bytes per array as multiples of the device L2 size\n"
                "--smoke = sizes-l2x 0.25,0.5,1,4,16, ks 1,64, reps 10, warmup 3, steady-iters 10\n");
    return 0;
  }
  const bool smoke = a.flag("--smoke");
  const std::vector<double> factors =
      a.list_f64("--sizes-l2x", smoke ? "0.25,0.5,1,4,16" : "0.125,0.25,0.5,0.75,1,1.5,2,4,8,16,64");
  std::vector<int> ks;
  for (double k : a.list_f64("--ks", smoke ? "1,64" : "1,8,32,128,512")) {
    if (k < 0 || k > 100000 || k != std::floor(k)) Args::die("bad K");
    ks.push_back(static_cast<int>(k));
  }
  const int reps = static_cast<int>(a.i64("--reps", smoke ? 10 : 30));
  const int warmup = static_cast<int>(a.i64("--warmup", smoke ? 3 : 10));
  const int steady_iters = static_cast<int>(a.i64("--steady-iters", 10));
  const uint32_t seed = static_cast<uint32_t>(a.i64("--seed", 12345));
  const int block = static_cast<int>(a.i64("--block", 256));
  const bool flush = !a.flag("--no-flush-l2");
  if (reps <= 0 || warmup < 0 || steady_iters <= 0 || block <= 0 || block > 1024 || block % 32)
    Args::die("invalid arguments");

  Device d;
  if (!d.open(static_cast<int>(a.i64("--device", 0)))) return 3;
  if (d.l2_bytes <= 0) Args::die("device reports no L2 size");
  std::vector<size_t> sizes;
  for (double f : factors) {
    if (f <= 0) Args::die("size factor must be > 0");
    size_t n = static_cast<size_t>(f * d.l2_bytes / sizeof(uint32_t));
    n = std::max<size_t>(256, n / 256 * 256);
    sizes.push_back(n);
  }
  const size_t max_n = *std::max_element(sizes.begin(), sizes.end());
  d.require((3 * max_n + 4 * kGuardElems) * sizeof(uint32_t) + (64ull << 20));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  Timer timer;
  timer.init(stream, d.l2_bytes);
  uint32_t* x_u = nullptr;
  CUDA_CHECK(cudaMalloc(&x_u, max_n * sizeof(uint32_t)));
  fill_gen(x_u, max_n, seed, /*unit_float=*/true, stream);
  Guarded t, y;
  t.alloc(max_n);
  y.alloc(max_n);
  const float* x = reinterpret_cast<const float*>(x_u);
  float* tf = reinterpret_cast<float*>(t.p);
  float* yf = reinterpret_cast<float*>(y.p);

  FILE* csv = std::fopen(a.str("--csv", "fusion.reps.csv").c_str(), "w");
  if (!csv) { std::perror("open csv"); return 6; }
  std::fputs(CSV_HEADER, csv);
  std::vector<std::pair<std::string, size_t>> correctness;
  std::vector<uint32_t> host;
  int failures = 0;

  for (size_t si = 0; si < sizes.size(); ++si) {
    const size_t n = sizes[si];
    const unsigned grid = grid_1d(n, block);
    for (int K : ks) {
      const std::string params = "n=" + std::to_string(n) + ";l2x=" + std::to_string(factors[si]) +
                                 ";K=" + std::to_string(K) + ";l2_bytes=" + std::to_string(d.l2_bytes);
      auto s1 = [&] { stage1_kernel<<<grid, block, 0, stream>>>(x, tf, n, K); };
      auto s2 = [&] { stage2_kernel<<<grid, block, 0, stream>>>(tf, yf, n, K); };
      auto fu = [&] { fused_kernel<<<grid, block, 0, stream>>>(x, yf, n, K); };
      auto mat = [&] { s1(); s2(); };

      auto ref_t = [&](size_t i) { return bits(stage_f(from_bits(gen_unit_float_bits(i, seed)), K)); };
      auto ref_y = [&](size_t i) { return bits(stage_g(stage_f(from_bits(gen_unit_float_bits(i, seed)), K), K)); };

      // ---- correctness: fused (y) and materialized (t and y) ----
      bool ok = true;
      {
        y.arm(stream);
        fill_u32(y.p, n, 0u, stream);  // 0.0f is never a valid output (all values stay near 1)
        fu();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::string id = "fused@" + params;
        const size_t bad = y.verify(id.c_str(), n, ref_y, host);
        correctness.push_back({id, bad});
        ok &= (bad == 0);
      }
      {
        t.arm(stream);
        y.arm(stream);
        fill_u32(t.p, n, 0u, stream);
        fill_u32(y.p, n, 0u, stream);
        mat();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::string id = "materialized@" + params;
        const size_t bad = t.verify((id + ":t").c_str(), n, ref_t, host) +
                           y.verify((id + ":y").c_str(), n, ref_y, host);
        correctness.push_back({id, bad});
        ok &= (bad == 0);
      }
      if (!ok) {
        std::fprintf(stderr, "FAIL %s; not timed\n", params.c_str());
        ++failures;
        continue;
      }

      // ---- timing ----
      struct V {
        const char* name;
        std::function<void()> call, pre;
        size_t rd, wr;  // algorithmic bytes of the *task* the variant performs
      };
      const size_t B = n * sizeof(float);
      const std::vector<V> vs = {
          {"fused", fu, nullptr, B, B},
          {"materialized", mat, nullptr, B, B},  // same task as fused: x -> y
          {"stage2_cold", s2, nullptr, B, B},
          {"stage2_warm", s2, s1, B, B},
      };
      for (const V& v : vs) {
        Row row;
        row.variant = v.name;
        row.params = params;
        row.n_elems = n;
        row.read_bytes = v.rd;
        row.write_bytes = v.wr;
        row.timing_mode = "isolated";
        row.flush = flush;
        // stage2_warm needs no flush semantics of its own: flush, then untimed stage1, then time stage2.
        write_rows(csv, "fusion", row, timer.isolated(v.call, warmup, reps, flush, v.pre));
        if (v.pre == nullptr && std::strcmp(v.name, "stage2_cold") != 0) {
          row.timing_mode = "steady";
          row.iters = steady_iters;
          row.flush = false;
          write_rows(csv, "fusion", row, timer.steady(v.call, warmup, reps, steady_iters));
        }
      }
      std::printf("%-48s ok\n", params.c_str());
    }
  }
  std::fclose(csv);

  FILE* mf = std::fopen(a.str("--meta", "fusion.meta.json").c_str(), "w");
  if (!mf) { std::perror("open meta"); return 6; }
  meta_begin(mf, "fusion", d, smoke);
  std::string sz = "[", fs = "[", kj = "[";
  for (size_t i = 0; i < sizes.size(); ++i) {
    sz += (i ? ", " : "") + std::to_string(sizes[i]);
    fs += (i ? ", " : "") + std::to_string(factors[i]);
  }
  for (size_t i = 0; i < ks.size(); ++i) kj += (i ? ", " : "") + std::to_string(ks[i]);
  meta_kv(mf, "sizes_elems", sz + "]");
  meta_kv(mf, "sizes_l2x", fs + "]");
  meta_kv(mf, "ks", kj + "]");
  meta_kv(mf, "block", std::to_string(block));
  meta_kv(mf, "flush_l2", flush ? "true" : "false");
  meta_kv(mf, "flush_buffer_bytes", std::to_string(timer.flush_elems * sizeof(uint32_t)));
  meta_kv(mf, "reps", std::to_string(reps));
  meta_kv(mf, "warmup", std::to_string(warmup));
  meta_kv(mf, "steady_iters", std::to_string(steady_iters));
  meta_kv(mf, "seed", std::to_string(seed));
  meta_end(mf, correctness, failures);
  std::fclose(mf);

  t.release();
  y.release();
  CUDA_CHECK(cudaFree(x_u));
  timer.destroy();
  CUDA_CHECK(cudaStreamDestroy(stream));
  std::printf("correctness_failures=%d\n", failures);
  return failures == 0 ? 0 : 1;
}

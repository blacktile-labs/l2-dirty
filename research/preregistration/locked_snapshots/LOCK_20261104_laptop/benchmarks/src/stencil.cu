// stencil.cu -- Gate B baseline #5: multi-sweep 2D stencil, materialized vs. fused sweeps
// (second workload family for RQ5 / H2)
//
// Grid: H x W uint32 on a torus (periodic boundaries), W fixed at 1024 (4-KiB rows).
// One sweep = 5-point stencil (stencil_host.hpp). S = 4 sweeps are computed two ways:
//   mat4      4 kernels of the 5-point sweep, ping-pong  X -> P -> Q -> P -> Q
//   fused2    2 kernels of the 13-point composite (= two sweeps), X -> P -> Q
//   fused2c   same as fused2 but conditional wrap instead of integer modulo (B5b ablation)
// Both produce bit-identical Q (integer ring + periodic boundaries), checked against a host
// reference. Like B4, the H2 question is where the intermediates live (L2 or DRAM):
//   sweep_cold  one 5-point sweep P -> Q after an L2 flush
//   sweep_warm  the same sweep right after an untimed X -> P sweep (P possibly still in L2)
// Array size per buffer is given as a multiple of the measured L2 size (--sizes-l2x).
// Every variant is timed "isolated" (flush before each rep); mat4 and fused2 also "steady".

#include "common.cuh"
#include "stencil_host.hpp"

using namespace dmk;

constexpr int W = 1024;
constexpr int BX = 32, BY = 8;

__device__ __forceinline__ int wrapi(int x, int m) {
  x %= m;
  return x < 0 ? x + m : x;
}

__global__ void sweep5_kernel(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, int H, Coef5 k) {
  const int c = blockIdx.x * BX + threadIdx.x;  // W % BX == 0, so c < W
  const int r = blockIdx.y * BY + threadIdx.y;
  if (r >= H) return;
  const int rn = r == 0 ? H - 1 : r - 1, rs = r == H - 1 ? 0 : r + 1;
  const int cw = c == 0 ? W - 1 : c - 1, ce = c == W - 1 ? 0 : c + 1;
  const size_t row = static_cast<size_t>(r) * W;
  out[row + c] = k.c * in[row + c] + k.n * in[static_cast<size_t>(rn) * W + c] +
                 k.s * in[static_cast<size_t>(rs) * W + c] + k.w * in[row + cw] + k.e * in[row + ce];
}

__global__ void sweep13_kernel(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, int H, Coef13 k) {
  const int c = blockIdx.x * BX + threadIdx.x;
  const int r = blockIdx.y * BY + threadIdx.y;
  if (r >= H) return;
  size_t R[5];
  int C[5];
#pragma unroll
  for (int d = 0; d < 5; ++d) {
    R[d] = static_cast<size_t>(wrapi(r + d - 2, H)) * W;
    C[d] = wrapi(c + d - 2, W);
  }
  // Order must match kOff13 in stencil_host.hpp.
  const uint32_t acc =
      k.v[0] * in[R[0] + C[2]] +                                                     // (-2, 0)
      k.v[1] * in[R[1] + C[1]] + k.v[2] * in[R[1] + C[2]] + k.v[3] * in[R[1] + C[3]] +  // (-1,-1..1)
      k.v[4] * in[R[2] + C[0]] + k.v[5] * in[R[2] + C[1]] + k.v[6] * in[R[2] + C[2]] +  // (0,-2..0)
      k.v[7] * in[R[2] + C[3]] + k.v[8] * in[R[2] + C[4]] +                             // (0, 1..2)
      k.v[9] * in[R[3] + C[1]] + k.v[10] * in[R[3] + C[2]] + k.v[11] * in[R[3] + C[3]] +  // (1,-1..1)
      k.v[12] * in[R[4] + C[2]];                                                       // (2, 0)
  out[static_cast<size_t>(r) * W + c] = acc;
}

// B5b ablation (one variable): identical to sweep13_kernel except that the radius-2 periodic
// wrap uses conditional add/subtract instead of integer modulo (valid because |d| <= 2 and
// H >= 8, W = 1024). Tests whether the modulo instructions made the fused kernel compute-bound.
__device__ __forceinline__ int wrap2(int x, int m) { return x < 0 ? x + m : (x >= m ? x - m : x); }

__global__ void sweep13c_kernel(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, int H, Coef13 k) {
  const int c = blockIdx.x * BX + threadIdx.x;
  const int r = blockIdx.y * BY + threadIdx.y;
  if (r >= H) return;
  size_t R[5];
  int C[5];
#pragma unroll
  for (int d = 0; d < 5; ++d) {
    R[d] = static_cast<size_t>(wrap2(r + d - 2, H)) * W;
    C[d] = wrap2(c + d - 2, W);
  }
  const uint32_t acc =
      k.v[0] * in[R[0] + C[2]] +
      k.v[1] * in[R[1] + C[1]] + k.v[2] * in[R[1] + C[2]] + k.v[3] * in[R[1] + C[3]] +
      k.v[4] * in[R[2] + C[0]] + k.v[5] * in[R[2] + C[1]] + k.v[6] * in[R[2] + C[2]] +
      k.v[7] * in[R[2] + C[3]] + k.v[8] * in[R[2] + C[4]] +
      k.v[9] * in[R[3] + C[1]] + k.v[10] * in[R[3] + C[2]] + k.v[11] * in[R[3] + C[3]] +
      k.v[12] * in[R[4] + C[2]];
  out[static_cast<size_t>(r) * W + c] = acc;
}

int main(int argc, char** argv) {
  Args a(argc, argv, {"--smoke", "--no-flush-l2"},
         {"--sizes-l2x", "--reps", "--warmup", "--steady-iters", "--seed", "--device", "--csv", "--meta"});
  if (a.help()) {
    std::printf("Usage: stencil [--smoke] [--sizes-l2x F,...] [--reps N] [--warmup N] [--steady-iters N]\n"
                "               [--seed N] [--no-flush-l2] [--device N] [--csv PATH] [--meta PATH]\n"
                "  --sizes-l2x: bytes per buffer (H x 1024 uint32) as multiples of the device L2 size\n"
                "--smoke = sizes-l2x 0.25,0.625,1,4, reps 10, warmup 3, steady-iters 10\n");
    return 0;
  }
  const bool smoke = a.flag("--smoke");
  const std::vector<double> factors = a.list_f64(
      "--sizes-l2x", smoke ? "0.25,0.625,1,4" : "0.0625,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1,1.25,1.5,2,4");
  const int reps = static_cast<int>(a.i64("--reps", smoke ? 10 : 30));
  const int warmup = static_cast<int>(a.i64("--warmup", smoke ? 3 : 5));
  const int steady_iters = static_cast<int>(a.i64("--steady-iters", 10));
  const uint32_t seed = static_cast<uint32_t>(a.i64("--seed", 12345));
  const bool flush = !a.flag("--no-flush-l2");
  if (reps <= 0 || warmup < 0 || steady_iters <= 0) Args::die("invalid arguments");

  Device d;
  if (!d.open(static_cast<int>(a.i64("--device", 0)))) return 3;
  if (d.l2_bytes <= 0) Args::die("device reports no L2 size");
  std::vector<int> heights;
  for (double f : factors) {
    if (f <= 0) Args::die("size factor must be > 0");
    long long h = static_cast<long long>(f * d.l2_bytes / sizeof(uint32_t) / W);
    h = std::max(8ll, h / BY * BY);  // whole thread-block rows, at least 8 rows
    if (h > (65535ll * BY)) Args::die("grid too tall");
    heights.push_back(static_cast<int>(h));
  }
  const size_t max_n = static_cast<size_t>(*std::max_element(heights.begin(), heights.end())) * W;
  d.require((3 * max_n + 4 * kGuardElems) * sizeof(uint32_t) + (64ull << 20));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  Timer timer;
  timer.init(stream, d.l2_bytes);
  uint32_t* X = nullptr;
  CUDA_CHECK(cudaMalloc(&X, max_n * sizeof(uint32_t)));
  fill_gen(X, max_n, seed, /*unit_float=*/false, stream);
  Guarded P, Q;
  P.alloc(max_n);
  Q.alloc(max_n);
  const Coef5 k5 = kStencil5;
  const Coef13 k13 = compose_twice(k5);

  FILE* csv = std::fopen(a.str("--csv", "stencil.reps.csv").c_str(), "w");
  if (!csv) { std::perror("open csv"); return 6; }
  std::fputs(CSV_HEADER, csv);
  std::vector<std::pair<std::string, size_t>> correctness;
  std::vector<uint32_t> host;
  int failures = 0;

  for (size_t si = 0; si < heights.size(); ++si) {
    const int H = heights[si];
    const size_t n = static_cast<size_t>(H) * W;
    const dim3 block(BX, BY), grid(W / BX, (H + BY - 1) / BY);
    const std::string params = "W=" + std::to_string(W) + ";H=" + std::to_string(H) + ";n=" + std::to_string(n) +
                               ";l2x=" + std::to_string(factors[si]) + ";S=4;l2_bytes=" + std::to_string(d.l2_bytes);
    auto s5 = [&](const uint32_t* in, uint32_t* out) { sweep5_kernel<<<grid, block, 0, stream>>>(in, out, H, k5); };
    auto s13 = [&](const uint32_t* in, uint32_t* out) { sweep13_kernel<<<grid, block, 0, stream>>>(in, out, H, k13); };
    auto s13c = [&](const uint32_t* in, uint32_t* out) { sweep13c_kernel<<<grid, block, 0, stream>>>(in, out, H, k13); };
    auto mat4 = [&] { s5(X, P.p); s5(P.p, Q.p); s5(Q.p, P.p); s5(P.p, Q.p); };
    auto fused2 = [&] { s13(X, P.p); s13(P.p, Q.p); };
    auto fused2c = [&] { s13c(X, P.p); s13c(P.p, Q.p); };

    // ---- host reference (X is regenerated identically on the host) ----
    std::vector<uint32_t> x(n), r1, r2, r3, r4;
    for (size_t i = 0; i < n; ++i) x[i] = gen_value(i, seed);
    sweep5_host(x, r1, H, W);
    sweep5_host(r1, r2, H, W);
    sweep5_host(r2, r3, H, W);
    sweep5_host(r3, r4, H, W);

    // ---- correctness gates (untimed) ----
    bool ok = true;
    auto gate = [&](const std::string& id, const std::function<void()>& run, Guarded& buf,
                    const std::vector<uint32_t>& ref) {
      P.arm(stream);
      Q.arm(stream);
      fill_u32(P.p, n, 0u, stream);
      fill_u32(Q.p, n, 0u, stream);
      run();
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream));
      size_t bad = buf.verify(id.c_str(), n, [&](size_t i) { return ref[i]; }, host);
      if (!bad) {  // second poison: an unwritten slot cannot match under both
        fill_u32(buf.p, n, 0xFFFFFFFFu, stream);
        run();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        bad = buf.verify(id.c_str(), n, [&](size_t i) { return ref[i]; }, host);
      }
      correctness.push_back({id, bad});
      ok &= (bad == 0);
    };
    gate("sweep5@" + params, [&] { s5(X, P.p); }, P, r1);
    gate("mat4@" + params, mat4, Q, r4);
    gate("fused2@" + params, fused2, Q, r4);
    gate("fused2c@" + params, fused2c, Q, r4);
    if (!ok) {
      std::fprintf(stderr, "FAIL %s; not timed\n", params.c_str());
      ++failures;
      continue;
    }

    // ---- timing ----
    const size_t B = n * sizeof(uint32_t);
    struct V {
      const char* name;
      std::function<void()> call, pre;
      bool steady;
    };
    const std::vector<V> vs = {
        {"mat4", mat4, nullptr, true},
        {"fused2", fused2, nullptr, true},
        {"fused2c", fused2c, nullptr, true},
        {"sweep_cold", [&] { s5(P.p, Q.p); }, nullptr, false},
        {"sweep_warm", [&] { s5(P.p, Q.p); }, [&] { s5(X, P.p); }, false},
    };
    for (const V& v : vs) {
      Row row;
      row.variant = v.name;
      row.params = params;
      row.n_elems = n;
      row.read_bytes = B;  // task: read the input grid once ...
      row.write_bytes = B;  // ... and write the result once
      row.timing_mode = "isolated";
      row.flush = flush;
      write_rows(csv, "stencil", row, timer.isolated(v.call, warmup, reps, flush, v.pre));
      if (v.steady) {
        row.timing_mode = "steady";
        row.iters = steady_iters;
        row.flush = false;
        write_rows(csv, "stencil", row, timer.steady(v.call, warmup, reps, steady_iters));
      }
    }
    std::printf("%-70s ok\n", params.c_str());
  }
  std::fclose(csv);

  FILE* mf = std::fopen(a.str("--meta", "stencil.meta.json").c_str(), "w");
  if (!mf) { std::perror("open meta"); return 6; }
  meta_begin(mf, "stencil", d, smoke);
  std::string hs = "[", fs = "[", c13 = "[";
  for (size_t i = 0; i < heights.size(); ++i) {
    hs += (i ? ", " : "") + std::to_string(heights[i]);
    fs += (i ? ", " : "") + std::to_string(factors[i]);
  }
  for (int j = 0; j < 13; ++j) c13 += (j ? ", " : "") + std::to_string(k13.v[j]);
  meta_kv(mf, "width", std::to_string(W));
  meta_kv(mf, "heights", hs + "]");
  meta_kv(mf, "sizes_l2x", fs + "]");
  meta_kv(mf, "sweeps", "4");
  meta_kv(mf, "coef5", "[" + std::to_string(k5.c) + ", " + std::to_string(k5.n) + ", " + std::to_string(k5.s) +
                           ", " + std::to_string(k5.w) + ", " + std::to_string(k5.e) + "]");
  meta_kv(mf, "coef13", c13 + "]");
  meta_kv(mf, "flush_l2", flush ? "true" : "false");
  meta_kv(mf, "flush_buffer_bytes", std::to_string(timer.flush_elems * sizeof(uint32_t)));
  meta_kv(mf, "reps", std::to_string(reps));
  meta_kv(mf, "warmup", std::to_string(warmup));
  meta_kv(mf, "steady_iters", std::to_string(steady_iters));
  meta_kv(mf, "seed", std::to_string(seed));
  meta_end(mf, correctness, failures);
  std::fclose(mf);

  P.release();
  Q.release();
  CUDA_CHECK(cudaFree(X));
  timer.destroy();
  CUDA_CHECK(cudaStreamDestroy(stream));
  std::printf("correctness_failures=%d\n", failures);
  return failures == 0 ? 0 : 1;
}

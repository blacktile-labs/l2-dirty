// transpose.cu -- Gate B baseline #2: matrix transpose (Blacktile data-move-kernel-lab)
//
//   out[c][r] = in[r][c]      in: R x C row-major uint32, out: C x R row-major
//
// Variants (same 32x32 tile, 32x8 threads, 4 elements per thread, bounds-checked):
//   copy_2d          out = in with the same indexing: the bandwidth ceiling (Harris 2013)
//   naive            coalesced reads, writes strided by R elements
//   tiled            shared-memory tile, no padding (32-way bank conflicts on the column read)
//   tiled_pad        shared-memory tile [32][33]: conflict-free
//   cublas_geam      vendor baseline (only when built with -DWITH_CUBLAS)
//
// Data is fp32-safe bit patterns in [1, 2) so cuBLAS (which computes 1*A + 0*B in float)
// reproduces the bits exactly. Every variant is checked element-for-element before timing.
// Traffic model and predictions: research/HYPOTHESES.md (Gate C, B2).

#include "common.cuh"

#ifdef WITH_CUBLAS
#include <cublas_v2.h>
#endif

using namespace dmk;

constexpr int TILE = 32;
constexpr int BROWS = 8;

__global__ void copy_2d(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, int R, int C) {
  const int x = blockIdx.x * TILE + threadIdx.x;
  const int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += BROWS)
    if (x < C && y + j < R) out[static_cast<size_t>(y + j) * C + x] = in[static_cast<size_t>(y + j) * C + x];
}

__global__ void transpose_naive(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, int R, int C) {
  const int x = blockIdx.x * TILE + threadIdx.x;  // column of in
  const int y = blockIdx.y * TILE + threadIdx.y;  // row of in
  for (int j = 0; j < TILE; j += BROWS)
    if (x < C && y + j < R) out[static_cast<size_t>(x) * R + (y + j)] = in[static_cast<size_t>(y + j) * C + x];
}

template <int PAD>
__global__ void transpose_tiled(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, int R, int C) {
  __shared__ uint32_t tile[TILE][TILE + PAD];
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += BROWS)
    if (x < C && y + j < R) tile[threadIdx.y + j][threadIdx.x] = in[static_cast<size_t>(y + j) * C + x];
  __syncthreads();
  x = blockIdx.y * TILE + threadIdx.x;  // column of out (= row of in)
  y = blockIdx.x * TILE + threadIdx.y;  // row of out (= column of in)
  for (int j = 0; j < TILE; j += BROWS)
    if (x < R && y + j < C) out[static_cast<size_t>(y + j) * R + x] = tile[threadIdx.x][threadIdx.y + j];
}

struct Shape {
  int R, C;
};

static std::vector<Shape> parse_shapes(const std::vector<std::string>& toks) {
  std::vector<Shape> out;
  for (const std::string& t : toks) {
    const size_t xpos = t.find('x');
    if (xpos == std::string::npos) Args::die("shape must be RxC: " + t);
    const long long R = std::atoll(t.substr(0, xpos).c_str()), C = std::atoll(t.substr(xpos + 1).c_str());
    if (R <= 0 || C <= 0 || R > (1 << 26) || C > (1 << 26)) Args::die("bad shape: " + t);
    if ((R + TILE - 1) / TILE > 65535) Args::die("too many rows for gridDim.y: " + t);
    out.push_back({static_cast<int>(R), static_cast<int>(C)});
  }
  return out;
}

int main(int argc, char** argv) {
  Args a(argc, argv, {"--smoke", "--no-flush-l2"},
         {"--shapes", "--reps", "--warmup", "--steady-iters", "--seed", "--device", "--csv", "--meta"});
  if (a.help()) {
    std::printf("Usage: transpose [--smoke] [--shapes RxC,...] [--reps N] [--warmup N] [--steady-iters N]\n"
                "                 [--seed N] [--no-flush-l2] [--device N] [--csv PATH] [--meta PATH]\n"
                "--smoke = shapes 4096x4096,1000x777, reps 10, warmup 3, steady-iters 10\n");
    return 0;
  }
  const bool smoke = a.flag("--smoke");
  const std::vector<Shape> shapes =
      parse_shapes(a.list("--shapes", smoke ? "4096x4096,1000x777"
                                            : "1024x1024,2048x2048,4096x4096,8192x8192,16384x1024,1024x16384,4097x4095"));
  const int reps = static_cast<int>(a.i64("--reps", smoke ? 10 : 30));
  const int warmup = static_cast<int>(a.i64("--warmup", smoke ? 3 : 10));
  const int steady_iters = static_cast<int>(a.i64("--steady-iters", 10));
  const uint32_t seed = static_cast<uint32_t>(a.i64("--seed", 12345));
  const bool flush = !a.flag("--no-flush-l2");
  if (reps <= 0 || warmup < 0 || steady_iters <= 0) Args::die("invalid reps/warmup/steady-iters");

  Device d;
  if (!d.open(static_cast<int>(a.i64("--device", 0)))) return 3;
  size_t max_n = 0;
  for (const Shape& s : shapes) max_n = std::max(max_n, static_cast<size_t>(s.R) * s.C);
  d.require((2 * max_n + 2 * kGuardElems) * sizeof(uint32_t) + (64ull << 20));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  Timer timer;
  timer.init(stream, d.l2_bytes);
  uint32_t* in = nullptr;
  CUDA_CHECK(cudaMalloc(&in, max_n * sizeof(uint32_t)));
  Guarded out;
  out.alloc(max_n);

#ifdef WITH_CUBLAS
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) Args::die("cublasCreate failed");
  cublasSetStream(handle, stream);
  const bool have_cublas = true;
#else
  const bool have_cublas = false;
#endif

  FILE* csv = std::fopen(a.str("--csv", "transpose.reps.csv").c_str(), "w");
  if (!csv) { std::perror("open csv"); return 6; }
  std::fputs(CSV_HEADER, csv);
  std::vector<std::pair<std::string, size_t>> correctness;
  std::vector<uint32_t> host;
  int failures = 0;

  for (const Shape& sh : shapes) {
    const int R = sh.R, C = sh.C;
    const size_t n = static_cast<size_t>(R) * C;
    fill_gen(in, n, seed, /*unit_float=*/true, stream);
    const dim3 block(TILE, BROWS);
    const dim3 grid((C + TILE - 1) / TILE, (R + TILE - 1) / TILE);
    const std::string params = "R=" + std::to_string(R) + ";C=" + std::to_string(C);

    struct V {
      const char* name;
      bool transposes;
      std::function<void()> call;
    };
    std::vector<V> vs = {
        {"copy_2d", false, [&] { copy_2d<<<grid, block, 0, stream>>>(in, out.p, R, C); }},
        {"naive", true, [&] { transpose_naive<<<grid, block, 0, stream>>>(in, out.p, R, C); }},
        {"tiled", true, [&] { transpose_tiled<0><<<grid, block, 0, stream>>>(in, out.p, R, C); }},
        {"tiled_pad", true, [&] { transpose_tiled<1><<<grid, block, 0, stream>>>(in, out.p, R, C); }},
    };
#ifdef WITH_CUBLAS
    vs.push_back({"cublas_geam", true, [&] {
                    const float one = 1.0f, zero = 0.0f;
                    // Row-major in (R x C) is column-major (C x R) with lda = C. We want the
                    // column-major (R x C) result with ldc = R, i.e. out = in^T.
                    cublasStatus_t st = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, R, C, &one,
                                                    reinterpret_cast<const float*>(in), C, &zero,
                                                    reinterpret_cast<const float*>(out.p), R,
                                                    reinterpret_cast<float*>(out.p), R);
                    if (st != CUBLAS_STATUS_SUCCESS) Args::die("cublasSgeam failed");
                  }});
#endif

    for (const V& v : vs) {
      const std::string id = std::string(v.name) + "@" + params;
      // ---- correctness gate ----
      out.arm(stream);
      // Poison the used range with values that differ from the expected ones.
      fill_u32(out.p, n, 0u, stream);  // 0 is never a [1,2) float bit pattern
      v.call();
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream));
      auto expect = [&](size_t k) -> uint32_t {
        if (!v.transposes) return gen_unit_float_bits(k, seed);
        const size_t c = k / R, r = k % R;  // out is C x R
        return gen_unit_float_bits(r * static_cast<size_t>(C) + c, seed);
      };
      const size_t bad = out.verify(id.c_str(), n, expect, host);
      correctness.push_back({id, bad});
      if (bad) {
        std::fprintf(stderr, "FAIL %s: %zu mismatching words; not timed\n", id.c_str(), bad);
        ++failures;
        continue;
      }
      // ---- timing ----
      Row row;
      row.variant = v.name;
      row.params = params;
      row.n_elems = n;
      row.read_bytes = n * sizeof(uint32_t);
      row.write_bytes = n * sizeof(uint32_t);
      row.timing_mode = "isolated";
      row.flush = flush;
      write_rows(csv, "transpose", row, timer.isolated(v.call, warmup, reps, flush));
      row.timing_mode = "steady";
      row.iters = steady_iters;
      row.flush = false;
      write_rows(csv, "transpose", row, timer.steady(v.call, warmup, reps, steady_iters));
      std::printf("%-12s %-22s ok\n", v.name, params.c_str());
    }
  }
  std::fclose(csv);

  FILE* mf = std::fopen(a.str("--meta", "transpose.meta.json").c_str(), "w");
  if (!mf) { std::perror("open meta"); return 6; }
  meta_begin(mf, "transpose", d, smoke);
  std::string sh_json = "[";
  for (size_t i = 0; i < shapes.size(); ++i)
    sh_json += (i ? ", " : "") + jstr(std::to_string(shapes[i].R) + "x" + std::to_string(shapes[i].C));
  meta_kv(mf, "shapes", sh_json + "]");
  meta_kv(mf, "tile", std::to_string(TILE));
  meta_kv(mf, "block_rows", std::to_string(BROWS));
  meta_kv(mf, "have_cublas", have_cublas ? "true" : "false");
  meta_kv(mf, "flush_l2", flush ? "true" : "false");
  meta_kv(mf, "flush_buffer_bytes", std::to_string(timer.flush_elems * sizeof(uint32_t)));
  meta_kv(mf, "reps", std::to_string(reps));
  meta_kv(mf, "warmup", std::to_string(warmup));
  meta_kv(mf, "steady_iters", std::to_string(steady_iters));
  meta_kv(mf, "seed", std::to_string(seed));
  meta_end(mf, correctness, failures);
  std::fclose(mf);

#ifdef WITH_CUBLAS
  cublasDestroy(handle);
#endif
  out.release();
  CUDA_CHECK(cudaFree(in));
  timer.destroy();
  CUDA_CHECK(cudaStreamDestroy(stream));
  std::printf("correctness_failures=%d\n", failures);
  return failures == 0 ? 0 : 1;
}

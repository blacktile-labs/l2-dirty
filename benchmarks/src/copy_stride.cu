// copy_stride.cu -- Gate B baseline #1 (Blacktile data-move-kernel-lab)
//
// Contiguous copy vs. strided-read copy on GPU global memory:
//
//     dst[i] = src[i * stride]      for i in [0, n_src / stride)
//
// The *source footprint* is fixed (bounded VRAM use); the number of useful
// elements n_out = n_src / stride shrinks as stride grows. This makes the
// DRAM-traffic model in research/HYPOTHESES.md (H1) directly falsifiable:
// for 4-byte elements and a DRAM fetch granularity of g bytes, the model
// predicts read traffic = min(4*n_src, g*n_out), i.e. time stays flat for
// strides up to g/4 and then falls as 1/stride.
//
// This program MEASURES only. It prints no model predictions; those live in
// benchmarks/scripts/bench_common.py and are labelled as estimates.
//
// Outputs:
//   --csv  <path>  one row per timed repetition (raw data)
//   --meta <path>  JSON with device attributes, versions, args, correctness
//
// Build: see benchmarks/scripts/build.sh

#include <cuda_runtime.h>

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t e_ = (expr);                                                    \
    if (e_ != cudaSuccess) {                                                    \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), \
                   __FILE__, __LINE__, cudaGetErrorString(e_));                 \
      std::exit(2);                                                             \
    }                                                                           \
  } while (0)

static constexpr uint32_t kGuardWord = 0xDEADBEEFu;
static constexpr size_t kGuardElems = 4096;  // 16 KiB guard on each side of dst

// ---------------------------------------------------------------------------
// Deterministic input: value depends only on (index, seed). SplitMix64 finalizer.
// ---------------------------------------------------------------------------
__host__ __device__ __forceinline__ uint32_t gen_value(uint64_t i, uint32_t seed) {
  uint64_t z = (i + 1) * 0x9E3779B97F4A7C15ull ^ (static_cast<uint64_t>(seed) << 32 | seed);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z ^= z >> 31;
  return static_cast<uint32_t>(z ^ (z >> 32));
}

#define GRID_STRIDE_LOOP(i, n)                                                    \
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < (n); \
       i += static_cast<size_t>(gridDim.x) * blockDim.x)

// ---------------------------------------------------------------------------
// Setup kernels (never timed)
// ---------------------------------------------------------------------------
__global__ void init_src(uint32_t* __restrict__ p, size_t n, uint32_t seed) {
  GRID_STRIDE_LOOP(i, n) p[i] = gen_value(i, seed);
}

__global__ void fill_u32(uint32_t* __restrict__ p, size_t n, uint32_t v) {
  GRID_STRIDE_LOOP(i, n) p[i] = v;
}

// Poison every output slot with a value guaranteed to differ from the
// expected one, so an element the kernel forgot to write cannot pass.
__global__ void poison_dst(uint32_t* __restrict__ dst, size_t n_out, size_t stride,
                           uint32_t seed) {
  GRID_STRIDE_LOOP(i, n_out) dst[i] = ~gen_value(i * stride, seed);
}

// Best-effort L2 eviction: stream-read a buffer several times larger than L2.
// Reads leave clean lines, so the *timed* kernel does not pay for write-backs
// of the flush buffer. The data-dependent store prevents dead-code removal.
__global__ void flush_l2_read(const uint32_t* __restrict__ buf, size_t n,
                              uint32_t* __restrict__ sink) {
  uint32_t acc = 0;
  GRID_STRIDE_LOOP(i, n) acc ^= buf[i];
  if (acc == 0x9E3779B9u) *sink = acc;
}

__global__ void empty_kernel() {}

// ---------------------------------------------------------------------------
// Measured kernels
// ---------------------------------------------------------------------------
__global__ void copy_contig_u32(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst,
                                size_t n) {
  GRID_STRIDE_LOOP(i, n) dst[i] = src[i];
}

// 16-byte vector loads/stores. Requires n % 4 == 0 and 16-byte alignment
// (cudaMalloc returns >= 256-byte aligned pointers; dst offset is checked).
__global__ void copy_contig_u32x4(const uint4* __restrict__ src, uint4* __restrict__ dst,
                                  size_t n4) {
  GRID_STRIDE_LOOP(i, n4) dst[i] = src[i];
}

__global__ void copy_strided_read_u32(const uint32_t* __restrict__ src,
                                      uint32_t* __restrict__ dst, size_t n_out,
                                      size_t stride) {
  GRID_STRIDE_LOOP(i, n_out) dst[i] = src[i * stride];
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
struct Args {
  size_t footprint_mib = 256;
  std::vector<size_t> strides = {1, 2, 4, 8, 16, 32, 64};
  int reps = 50;
  int warmup = 10;
  int block = 256;
  uint32_t seed = 12345u;
  bool flush_l2 = true;
  int l2_fetch_granularity = -1;  // -1 = leave driver default untouched
  int device = 0;
  bool smoke = false;
  std::string csv_path = "copy_stride.reps.csv";
  std::string meta_path = "copy_stride.meta.json";
};

static void usage(const char* argv0) {
  std::fprintf(stderr,
               "Usage: %s [--smoke] [--footprint-mib N] [--strides 1,2,4,...]\n"
               "          [--reps N] [--warmup N] [--block N] [--seed N] [--no-flush-l2]\n"
               "          [--l2-fetch-granularity BYTES] [--device N]\n"
               "          [--csv PATH] [--meta PATH]\n"
               "--smoke = footprint 64 MiB, strides 1,2,8,16,32, reps 10, warmup 3\n"
               "          (explicit flags given after --smoke override these)\n",
               argv0);
}

static std::vector<size_t> parse_list(const char* s) {
  std::vector<size_t> out;
  std::string tok;
  for (const char* p = s;; ++p) {
    if (*p == ',' || *p == '\0') {
      if (!tok.empty()) out.push_back(static_cast<size_t>(std::strtoull(tok.c_str(), nullptr, 10)));
      tok.clear();
      if (*p == '\0') break;
    } else {
      tok.push_back(*p);
    }
  }
  return out;
}

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", name);
        std::exit(1);
      }
      return argv[++i];
    };
    if (k == "--smoke") {
      a.smoke = true;
      a.footprint_mib = 64;
      a.strides = {1, 2, 8, 16, 32};
      a.reps = 10;
      a.warmup = 3;
    } else if (k == "--footprint-mib") a.footprint_mib = std::strtoull(need("--footprint-mib"), nullptr, 10);
    else if (k == "--strides") a.strides = parse_list(need("--strides"));
    else if (k == "--reps") a.reps = std::atoi(need("--reps"));
    else if (k == "--warmup") a.warmup = std::atoi(need("--warmup"));
    else if (k == "--block") a.block = std::atoi(need("--block"));
    else if (k == "--seed") a.seed = static_cast<uint32_t>(std::strtoul(need("--seed"), nullptr, 10));
    else if (k == "--no-flush-l2") a.flush_l2 = false;
    else if (k == "--l2-fetch-granularity") a.l2_fetch_granularity = std::atoi(need("--l2-fetch-granularity"));
    else if (k == "--device") a.device = std::atoi(need("--device"));
    else if (k == "--csv") a.csv_path = need("--csv");
    else if (k == "--meta") a.meta_path = need("--meta");
    else if (k == "-h" || k == "--help") { usage(argv[0]); std::exit(0); }
    else { std::fprintf(stderr, "unknown argument: %s\n", k.c_str()); usage(argv[0]); std::exit(1); }
  }
  if (a.footprint_mib == 0 || a.reps <= 0 || a.warmup < 0 || a.block <= 0 || a.block > 1024 ||
      a.block % 32 != 0 || a.strides.empty()) {
    std::fprintf(stderr, "invalid arguments\n");
    std::exit(1);
  }
  for (size_t s : a.strides)
    if (s == 0) { std::fprintf(stderr, "stride must be >= 1\n"); std::exit(1); }
  return a;
}

// ---------------------------------------------------------------------------
// Variants
// ---------------------------------------------------------------------------
enum class Kind { Empty, MemcpyD2D, ContigU32, ContigU32x4, StridedReadU32 };

struct Variant {
  std::string name;
  Kind kind;
  size_t stride;
};

struct Ctx {
  uint32_t* src = nullptr;       // n_src elements
  uint32_t* dst_base = nullptr;  // n_src + 2*guard elements
  uint32_t* dst = nullptr;       // dst_base + guard
  uint32_t* flush_buf = nullptr;
  uint32_t* sink = nullptr;
  size_t n_src = 0;
  size_t flush_elems = 0;
  int block = 256;
  cudaStream_t stream = nullptr;
};

static unsigned grid_for(size_t n, int block) {
  size_t g = (n + block - 1) / block;
  if (g == 0) g = 1;
  if (g > 0x7FFFFFFFull) g = 0x7FFFFFFFull;  // grid-stride loop covers the rest
  return static_cast<unsigned>(g);
}

static size_t n_out_of(const Variant& v, size_t n_src) {
  switch (v.kind) {
    case Kind::Empty: return 0;
    case Kind::StridedReadU32: return n_src / v.stride;
    default: return n_src;
  }
}

static unsigned grid_of(const Variant& v, const Ctx& c) {
  switch (v.kind) {
    case Kind::Empty: return 1;
    case Kind::MemcpyD2D: return 0;
    case Kind::ContigU32x4: return grid_for(c.n_src / 4, c.block);
    default: return grid_for(n_out_of(v, c.n_src), c.block);
  }
}

static void launch(const Variant& v, const Ctx& c) {
  const size_t n_out = n_out_of(v, c.n_src);
  switch (v.kind) {
    case Kind::Empty:
      empty_kernel<<<1, 32, 0, c.stream>>>();
      break;
    case Kind::MemcpyD2D:
      CUDA_CHECK(cudaMemcpyAsync(c.dst, c.src, c.n_src * sizeof(uint32_t),
                                 cudaMemcpyDeviceToDevice, c.stream));
      break;
    case Kind::ContigU32:
      copy_contig_u32<<<grid_of(v, c), c.block, 0, c.stream>>>(c.src, c.dst, c.n_src);
      break;
    case Kind::ContigU32x4:
      copy_contig_u32x4<<<grid_of(v, c), c.block, 0, c.stream>>>(
          reinterpret_cast<const uint4*>(c.src), reinterpret_cast<uint4*>(c.dst), c.n_src / 4);
      break;
    case Kind::StridedReadU32:
      copy_strided_read_u32<<<grid_of(v, c), c.block, 0, c.stream>>>(c.src, c.dst, n_out,
                                                                      v.stride);
      break;
  }
}

// Returns number of mismatching words (0 = pass). Checks the full dst buffer:
// [0, guard) and [guard + n_out, end) must still hold kGuardWord.
static size_t verify(const Variant& v, const Ctx& c, uint32_t seed, std::vector<uint32_t>& host) {
  const size_t total = c.n_src + 2 * kGuardElems;
  const size_t n_out = n_out_of(v, c.n_src);
  host.resize(total);
  CUDA_CHECK(cudaMemcpy(host.data(), c.dst_base, total * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  size_t bad = 0;
  int printed = 0;
  for (size_t j = 0; j < total; ++j) {
    uint32_t expect;
    if (j >= kGuardElems && j < kGuardElems + n_out) {
      const size_t i = j - kGuardElems;
      expect = gen_value(static_cast<uint64_t>(i) * v.stride, seed);
    } else {
      expect = kGuardWord;
    }
    if (host[j] != expect) {
      if (printed < 5) {
        std::fprintf(stderr, "  [%s] mismatch at dst_base[%zu] (%s): got 0x%08x expected 0x%08x\n",
                     v.name.c_str(), j,
                     (j < kGuardElems || j >= kGuardElems + n_out) ? "guard" : "data", host[j],
                     expect);
        ++printed;
      }
      ++bad;
    }
  }
  return bad;
}

static std::string json_escape(const std::string& s) {
  std::string o;
  for (char ch : s) {
    if (ch == '"' || ch == '\\') { o.push_back('\\'); o.push_back(ch); }
    else if (static_cast<unsigned char>(ch) < 0x20) o.push_back(' ');
    else o.push_back(ch);
  }
  return o;
}

static int attr(cudaDeviceAttr a, int dev) {
  int v = -1;
  if (cudaDeviceGetAttribute(&v, a, dev) != cudaSuccess) { cudaGetLastError(); return -1; }
  return v;
}

int main(int argc, char** argv) {
  const Args a = parse_args(argc, argv);

  int ndev = 0;
  cudaError_t e = cudaGetDeviceCount(&ndev);
  if (e != cudaSuccess || ndev == 0) {
    std::fprintf(stderr, "No usable CUDA device: %s\n",
                 e != cudaSuccess ? cudaGetErrorString(e) : "device count is 0");
    return 3;
  }
  CUDA_CHECK(cudaSetDevice(a.device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, a.device));
  int drv = 0, rt = 0;
  CUDA_CHECK(cudaDriverGetVersion(&drv));
  CUDA_CHECK(cudaRuntimeGetVersion(&rt));

  // The limit is only a hint; the driver may clamp or ignore it. We record the
  // value read back, and treat a failed set as a warning, not an error.
  if (a.l2_fetch_granularity >= 0) {
    cudaError_t le = cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity,
                                        static_cast<size_t>(a.l2_fetch_granularity));
    if (le != cudaSuccess) {
      std::fprintf(stderr, "warning: cudaLimitMaxL2FetchGranularity=%d not applied: %s\n",
                   a.l2_fetch_granularity, cudaGetErrorString(le));
      cudaGetLastError();
    }
  }
  size_t l2_fetch = 0;
  if (cudaDeviceGetLimit(&l2_fetch, cudaLimitMaxL2FetchGranularity) != cudaSuccess) {
    cudaGetLastError();
    l2_fetch = 0;  // reported as -1 below
  }

  const int l2_bytes = attr(cudaDevAttrL2CacheSize, a.device);
  Ctx c;
  c.block = a.block;
  c.n_src = a.footprint_mib * 1024ull * 1024ull / sizeof(uint32_t);
  c.n_src -= c.n_src % 4;  // keep the uint4 variant exact
  c.flush_elems = std::max<size_t>(4ull * static_cast<size_t>(std::max(l2_bytes, 0)),
                                   16ull << 20) / sizeof(uint32_t);

  const size_t need = (c.n_src + c.n_src + 2 * kGuardElems + c.flush_elems + 1) * sizeof(uint32_t);
  size_t free_b = 0, total_b = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
  if (need > free_b / 2) {
    std::fprintf(stderr,
                 "Refusing: needs %.1f MiB but only %.1f MiB free (limit is 50%% of free). "
                 "Lower --footprint-mib.\n",
                 need / 1048576.0, free_b / 1048576.0);
    return 4;
  }

  CUDA_CHECK(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaMalloc(&c.src, c.n_src * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&c.dst_base, (c.n_src + 2 * kGuardElems) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&c.flush_buf, c.flush_elems * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&c.sink, sizeof(uint32_t)));
  c.dst = c.dst_base + kGuardElems;
  if (reinterpret_cast<uintptr_t>(c.dst) % 16 != 0) {
    std::fprintf(stderr, "dst not 16-byte aligned\n");
    return 5;
  }

  const unsigned init_grid = std::min<unsigned>(grid_for(c.n_src, 256), 65535u);
  init_src<<<init_grid, 256, 0, c.stream>>>(c.src, c.n_src, a.seed);
  fill_u32<<<std::min<unsigned>(grid_for(c.flush_elems, 256), 65535u), 256, 0, c.stream>>>(
      c.flush_buf, c.flush_elems, 0u);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(c.stream));

  std::vector<Variant> variants;
  variants.push_back({"empty_kernel", Kind::Empty, 1});
  variants.push_back({"memcpy_d2d", Kind::MemcpyD2D, 1});
  variants.push_back({"copy_contig_u32", Kind::ContigU32, 1});
  variants.push_back({"copy_contig_u32x4", Kind::ContigU32x4, 1});
  for (size_t s : a.strides)
    variants.push_back({"copy_strided_read_u32_s" + std::to_string(s), Kind::StridedReadU32, s});

  FILE* csv = std::fopen(a.csv_path.c_str(), "w");
  if (!csv) { std::perror("open csv"); return 6; }
  std::fprintf(csv,
               "variant,kernel,elem_bytes,n_src,stride,n_out,logical_read_bytes,"
               "logical_write_bytes,block,grid,flush_l2,rep,time_ms\n");

  cudaEvent_t ev0, ev1;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));
  std::vector<uint32_t> host;
  std::vector<std::pair<std::string, size_t>> correctness;  // name -> mismatches
  int failures = 0;

  for (const Variant& v : variants) {
    const size_t n_out = n_out_of(v, c.n_src);
    const char* kernel_name = v.kind == Kind::Empty          ? "empty_kernel"
                              : v.kind == Kind::MemcpyD2D    ? "cudaMemcpyAsync(D2D)"
                              : v.kind == Kind::ContigU32    ? "copy_contig_u32"
                              : v.kind == Kind::ContigU32x4  ? "copy_contig_u32x4"
                                                             : "copy_strided_read_u32";

    // ---- correctness gate (untimed) ----
    if (v.kind != Kind::Empty) {
      const size_t total = c.n_src + 2 * kGuardElems;
      fill_u32<<<std::min<unsigned>(grid_for(total, 256), 65535u), 256, 0, c.stream>>>(
          c.dst_base, total, kGuardWord);
      poison_dst<<<std::min<unsigned>(grid_for(n_out, 256), 65535u), 256, 0, c.stream>>>(
          c.dst, n_out, v.stride, a.seed);
      CUDA_CHECK(cudaGetLastError());
      launch(v, c);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(c.stream));
      const size_t bad = verify(v, c, a.seed, host);
      correctness.push_back({v.name, bad});
      if (bad != 0) {
        std::fprintf(stderr, "FAIL %s: %zu mismatching words; not timed\n", v.name.c_str(), bad);
        ++failures;
        continue;
      }
    } else {
      correctness.push_back({v.name, 0});
    }

    // ---- warmup ----
    for (int w = 0; w < a.warmup; ++w) launch(v, c);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(c.stream));

    // ---- timed repetitions: each rep isolated by a sync ----
    const unsigned fgrid = std::min<unsigned>(grid_for(c.flush_elems, 256), 65535u);
    for (int r = 0; r < a.reps; ++r) {
      if (a.flush_l2) flush_l2_read<<<fgrid, 256, 0, c.stream>>>(c.flush_buf, c.flush_elems, c.sink);
      CUDA_CHECK(cudaEventRecord(ev0, c.stream));
      launch(v, c);
      CUDA_CHECK(cudaEventRecord(ev1, c.stream));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float ms = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
      std::fprintf(csv, "%s,%s,%zu,%zu,%zu,%zu,%zu,%zu,%d,%u,%d,%d,%.6f\n", v.name.c_str(),
                   kernel_name, sizeof(uint32_t), c.n_src, v.stride, n_out,
                   n_out * sizeof(uint32_t), n_out * sizeof(uint32_t),
                   v.kind == Kind::MemcpyD2D ? 0 : (v.kind == Kind::Empty ? 32 : c.block),
                   grid_of(v, c), a.flush_l2 ? 1 : 0, r, ms);
    }
    CUDA_CHECK(cudaGetLastError());
    std::printf("%-32s n_out=%-10zu done (%d reps)\n", v.name.c_str(), n_out, a.reps);
  }
  std::fclose(csv);

  // ---- metadata ----
  FILE* mf = std::fopen(a.meta_path.c_str(), "w");
  if (!mf) { std::perror("open meta"); return 6; }
  std::fprintf(mf, "{\n");
  std::fprintf(mf, "  \"benchmark\": \"copy_stride\",\n");
  std::fprintf(mf, "  \"smoke\": %s,\n", a.smoke ? "true" : "false");
  std::fprintf(mf, "  \"device_index\": %d,\n", a.device);
  std::fprintf(mf, "  \"device_name\": \"%s\",\n", json_escape(prop.name).c_str());
  std::fprintf(mf, "  \"compute_capability\": \"%d.%d\",\n",
               attr(cudaDevAttrComputeCapabilityMajor, a.device),
               attr(cudaDevAttrComputeCapabilityMinor, a.device));
  std::fprintf(mf, "  \"cuda_driver_version\": %d,\n", drv);
  std::fprintf(mf, "  \"cuda_runtime_version\": %d,\n", rt);
  std::fprintf(mf, "  \"nvcc_version\": \"%d.%d\",\n", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__);
  std::fprintf(mf, "  \"mem_total_bytes\": %zu,\n", total_b);
  std::fprintf(mf, "  \"mem_free_bytes_at_start\": %zu,\n", free_b);
  std::fprintf(mf, "  \"sm_count\": %d,\n", attr(cudaDevAttrMultiProcessorCount, a.device));
  std::fprintf(mf, "  \"sm_clock_khz_attr\": %d,\n", attr(cudaDevAttrClockRate, a.device));
  std::fprintf(mf, "  \"mem_clock_khz_attr\": %d,\n", attr(cudaDevAttrMemoryClockRate, a.device));
  std::fprintf(mf, "  \"mem_bus_width_bits\": %d,\n", attr(cudaDevAttrGlobalMemoryBusWidth, a.device));
  std::fprintf(mf, "  \"l2_cache_bytes\": %d,\n", l2_bytes);
  std::fprintf(mf, "  \"l2_fetch_granularity_bytes\": %lld,\n",
               l2_fetch == 0 ? -1LL : static_cast<long long>(l2_fetch));
  std::fprintf(mf, "  \"l2_fetch_granularity_requested\": %d,\n", a.l2_fetch_granularity);
  std::fprintf(mf, "  \"max_threads_per_sm\": %d,\n", attr(cudaDevAttrMaxThreadsPerMultiProcessor, a.device));
  std::fprintf(mf, "  \"max_regs_per_sm\": %d,\n", attr(cudaDevAttrMaxRegistersPerMultiprocessor, a.device));
  std::fprintf(mf, "  \"max_smem_per_sm_bytes\": %d,\n", attr(cudaDevAttrMaxSharedMemoryPerMultiprocessor, a.device));
  std::fprintf(mf, "  \"integrated\": %d,\n", attr(cudaDevAttrIntegrated, a.device));
  std::fprintf(mf, "  \"ecc_enabled\": %d,\n", attr(cudaDevAttrEccEnabled, a.device));
  std::fprintf(mf, "  \"footprint_mib\": %zu,\n", a.footprint_mib);
  std::fprintf(mf, "  \"n_src\": %zu,\n", c.n_src);
  std::fprintf(mf, "  \"flush_l2\": %s,\n", a.flush_l2 ? "true" : "false");
  std::fprintf(mf, "  \"flush_buffer_bytes\": %zu,\n", c.flush_elems * sizeof(uint32_t));
  std::fprintf(mf, "  \"reps\": %d,\n  \"warmup\": %d,\n  \"block\": %d,\n  \"seed\": %u,\n",
               a.reps, a.warmup, a.block, a.seed);
  std::fprintf(mf, "  \"correctness\": {");
  for (size_t i = 0; i < correctness.size(); ++i)
    std::fprintf(mf, "%s\"%s\": %zu", i ? ", " : "", correctness[i].first.c_str(),
                 correctness[i].second);
  std::fprintf(mf, "},\n");
  std::fprintf(mf, "  \"correctness_failures\": %d\n}\n", failures);
  std::fclose(mf);

  CUDA_CHECK(cudaEventDestroy(ev0));
  CUDA_CHECK(cudaEventDestroy(ev1));
  CUDA_CHECK(cudaFree(c.src));
  CUDA_CHECK(cudaFree(c.dst_base));
  CUDA_CHECK(cudaFree(c.flush_buf));
  CUDA_CHECK(cudaFree(c.sink));
  CUDA_CHECK(cudaStreamDestroy(c.stream));
  std::printf("correctness_failures=%d\n", failures);
  return failures == 0 ? 0 : 1;
}

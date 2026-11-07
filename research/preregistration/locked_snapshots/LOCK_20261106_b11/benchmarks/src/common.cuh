// common.cuh -- shared CUDA infrastructure for the B2-B4 benchmarks.
//
// (copy_stride.cu predates this header and is kept byte-identical because its smoke results
//  were measured from that exact source; see research/DECISION_LOG.md D-009.)
//
// Conventions shared by every benchmark binary:
//  * correctness is checked (all outputs + guard words) before anything is timed;
//    a failing variant is not timed and the process exits with rc = 1
//  * CSV: one row per timed repetition, generic schema (CSV_HEADER below)
//  * JSON meta: device attributes, versions, args, per-variant correctness
//  * the binary measures only; traffic models live in benchmarks/scripts/bench_common.py
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "dmk_host.hpp"

#define CUDA_CHECK(expr)                                                         \
  do {                                                                           \
    cudaError_t e_ = (expr);                                                     \
    if (e_ != cudaSuccess) {                                                     \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), \
                   __FILE__, __LINE__, cudaGetErrorString(e_));                  \
      std::exit(2);                                                              \
    }                                                                            \
  } while (0)

#define DMK_GRID_STRIDE_LOOP(i, n)                                                         \
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < (n); \
       i += static_cast<size_t>(gridDim.x) * blockDim.x)

namespace dmk {

constexpr uint32_t kGuardWord = 0xDEADBEEFu;
constexpr size_t kGuardElems = 4096;  // 16 KiB of guard words on each side of every output

inline unsigned grid_1d(size_t n, int block, unsigned cap = 0x7FFFFFFFu) {
  size_t g = (n + block - 1) / block;
  if (g == 0) g = 1;
  if (g > cap) g = cap;  // grid-stride loops cover the rest
  return static_cast<unsigned>(g);
}

// ---------------------------------------------------------------------------
// Utility kernels (never timed)
// ---------------------------------------------------------------------------
__global__ void fill_u32_kernel(uint32_t* __restrict__ p, size_t n, uint32_t v) {
  DMK_GRID_STRIDE_LOOP(i, n) p[i] = v;
}

__global__ void fill_gen_kernel(uint32_t* __restrict__ p, size_t n, uint32_t seed, int unit_float) {
  DMK_GRID_STRIDE_LOOP(i, n) p[i] = unit_float ? gen_unit_float_bits(i, seed) : gen_value(i, seed);
}

// Best-effort L2 eviction by stream-reading a buffer >= 4x L2. Reads leave clean lines,
// so the timed kernel does not pay for the flush buffer's write-backs.
__global__ void flush_l2_read_kernel(const uint32_t* __restrict__ buf, size_t n,
                                     uint32_t* __restrict__ sink) {
  uint32_t acc = 0;
  DMK_GRID_STRIDE_LOOP(i, n) acc ^= buf[i];
  if (acc == 0x9E3779B9u) *sink = acc;
}

inline void fill_u32(uint32_t* p, size_t n, uint32_t v, cudaStream_t s) {
  fill_u32_kernel<<<grid_1d(n, 256, 65535), 256, 0, s>>>(p, n, v);
  CUDA_CHECK(cudaGetLastError());
}

inline void fill_gen(uint32_t* p, size_t n, uint32_t seed, bool unit_float, cudaStream_t s) {
  fill_gen_kernel<<<grid_1d(n, 256, 65535), 256, 0, s>>>(p, n, seed, unit_float ? 1 : 0);
  CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Command line: "--key value" pairs plus bare flags. Unknown keys are rejected.
// ---------------------------------------------------------------------------
class Args {
 public:
  Args(int argc, char** argv, std::vector<std::string> flags, std::vector<std::string> keys)
      : flags_(std::move(flags)), keys_(std::move(keys)) {
    for (int i = 1; i < argc; ++i) {
      std::string k = argv[i];
      if (k == "-h" || k == "--help") { help_ = true; continue; }
      if (std::find(flags_.begin(), flags_.end(), k) != flags_.end()) { set_.insert({k, "1"}); continue; }
      if (std::find(keys_.begin(), keys_.end(), k) != keys_.end()) {
        if (i + 1 >= argc) die("missing value for " + k);
        set_[k] = argv[++i];
        continue;
      }
      die("unknown argument: " + k);
    }
  }
  bool help() const { return help_; }
  bool flag(const std::string& k) const { return set_.count(k) != 0; }
  bool has(const std::string& k) const { return set_.count(k) != 0; }
  std::string str(const std::string& k, const std::string& def) const {
    auto it = set_.find(k);
    return it == set_.end() ? def : it->second;
  }
  long long i64(const std::string& k, long long def) const {
    auto it = set_.find(k);
    if (it == set_.end()) return def;
    char* end = nullptr;
    long long v = std::strtoll(it->second.c_str(), &end, 10);
    if (*end != '\0') die("not an integer: " + k + " " + it->second);
    return v;
  }
  std::vector<std::string> list(const std::string& k, const std::string& def) const {
    std::vector<std::string> out;
    std::string s = str(k, def), tok;
    for (char c : s + ",") {
      if (c == ',') { if (!tok.empty()) out.push_back(tok); tok.clear(); }
      else tok.push_back(c);
    }
    return out;
  }
  std::vector<double> list_f64(const std::string& k, const std::string& def) const {
    std::vector<double> out;
    for (const std::string& t : list(k, def)) {
      char* end = nullptr;
      double v = std::strtod(t.c_str(), &end);
      if (*end != '\0') die("not a number in " + k + ": " + t);
      out.push_back(v);
    }
    return out;
  }
  [[noreturn]] static void die(const std::string& msg) {
    std::fprintf(stderr, "%s\n", msg.c_str());
    std::exit(1);
  }

 private:
  std::vector<std::string> flags_, keys_;
  std::map<std::string, std::string> set_;
  bool help_ = false;
};

// ---------------------------------------------------------------------------
// Output buffer surrounded by guard words.
// ---------------------------------------------------------------------------
struct Guarded {
  uint32_t* base = nullptr;
  uint32_t* p = nullptr;  // base + kGuardElems
  size_t capacity = 0;    // usable elements

  void alloc(size_t n) {
    capacity = n;
    CUDA_CHECK(cudaMalloc(&base, (n + 2 * kGuardElems) * sizeof(uint32_t)));
    p = base + kGuardElems;
  }
  void release() { if (base) CUDA_CHECK(cudaFree(base)); base = p = nullptr; }
  size_t total() const { return capacity + 2 * kGuardElems; }
  // Fill everything (including unused capacity) with guard words.
  void arm(cudaStream_t s) { fill_u32(base, total(), kGuardWord, s); }
  // Copy the whole buffer back and compare: [0, n_used) against expect(i), everything else
  // against kGuardWord. Returns the number of mismatching words; prints the first few.
  size_t verify(const char* name, size_t n_used, const std::function<uint32_t(size_t)>& expect,
                std::vector<uint32_t>& host) const {
    host.resize(total());
    CUDA_CHECK(cudaMemcpy(host.data(), base, total() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    size_t bad = 0;
    int printed = 0;
    for (size_t j = 0; j < total(); ++j) {
      const bool data = j >= kGuardElems && j < kGuardElems + n_used;
      const uint32_t e = data ? expect(j - kGuardElems) : kGuardWord;
      if (host[j] != e) {
        if (printed++ < 5)
          std::fprintf(stderr, "  [%s] mismatch at %s[%zu]: got 0x%08x expected 0x%08x\n", name,
                       data ? "data" : "guard", data ? j - kGuardElems : j, host[j], e);
        ++bad;
      }
    }
    return bad;
  }
};

// ---------------------------------------------------------------------------
// Timing. Two modes, both recorded per repetition:
//   isolated: optional L2 flush, then one CUDA-event pair around one call (sync per rep)
//   steady:   one event pair around `iters` back-to-back calls, no flush; reports ms/call
// ---------------------------------------------------------------------------
struct Timer {
  cudaStream_t s = nullptr;
  cudaEvent_t e0 = nullptr, e1 = nullptr;
  uint32_t* flush_buf = nullptr;
  uint32_t* sink = nullptr;
  size_t flush_elems = 0;

  void init(cudaStream_t stream, int l2_bytes) {
    s = stream;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    flush_elems = std::max<size_t>(4ull * static_cast<size_t>(std::max(l2_bytes, 0)), 16ull << 20) /
                  sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(&flush_buf, flush_elems * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&sink, sizeof(uint32_t)));
    fill_u32(flush_buf, flush_elems, 0u, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
  }
  void destroy() {
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaFree(flush_buf));
    CUDA_CHECK(cudaFree(sink));
  }
  void flush() {
    flush_l2_read_kernel<<<grid_1d(flush_elems, 256, 65535), 256, 0, s>>>(flush_buf, flush_elems, sink);
  }
  // `pre` (optional) runs after the flush and before the start event, i.e. untimed; used to
  // time a kernel in the cache state left behind by its producer.
  std::vector<float> isolated(const std::function<void()>& call, int warmup, int reps, bool do_flush,
                              const std::function<void()>& pre = nullptr) {
    for (int w = 0; w < warmup; ++w) { if (pre) pre(); call(); }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<float> ms(reps);
    for (int r = 0; r < reps; ++r) {
      if (do_flush) flush();
      if (pre) pre();
      CUDA_CHECK(cudaEventRecord(e0, s));
      call();
      CUDA_CHECK(cudaEventRecord(e1, s));
      CUDA_CHECK(cudaEventSynchronize(e1));
      CUDA_CHECK(cudaEventElapsedTime(&ms[r], e0, e1));
    }
    CUDA_CHECK(cudaGetLastError());
    return ms;
  }
  std::vector<float> steady(const std::function<void()>& call, int warmup, int reps, int iters) {
    for (int w = 0; w < warmup; ++w) call();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<float> ms(reps);
    for (int r = 0; r < reps; ++r) {
      CUDA_CHECK(cudaEventRecord(e0, s));
      for (int k = 0; k < iters; ++k) call();
      CUDA_CHECK(cudaEventRecord(e1, s));
      CUDA_CHECK(cudaEventSynchronize(e1));
      CUDA_CHECK(cudaEventElapsedTime(&ms[r], e0, e1));
      ms[r] /= static_cast<float>(iters);
    }
    CUDA_CHECK(cudaGetLastError());
    return ms;
  }
};

// ---------------------------------------------------------------------------
// CSV (generic schema shared by B2-B4)
// ---------------------------------------------------------------------------
constexpr const char* CSV_HEADER =
    "benchmark,variant,params,n_elems,logical_read_bytes,logical_write_bytes,"
    "timing_mode,iters_per_rep,flush_l2,rep,time_ms\n";

struct Row {
  std::string variant, params, timing_mode;
  size_t n_elems = 0, read_bytes = 0, write_bytes = 0;
  int iters = 1;
  bool flush = false;
};

inline void write_rows(FILE* f, const char* bench, const Row& r, const std::vector<float>& ms) {
  for (size_t i = 0; i < ms.size(); ++i)
    std::fprintf(f, "%s,%s,%s,%zu,%zu,%zu,%s,%d,%d,%zu,%.6f\n", bench, r.variant.c_str(),
                 r.params.c_str(), r.n_elems, r.read_bytes, r.write_bytes, r.timing_mode.c_str(),
                 r.iters, r.flush ? 1 : 0, i, ms[i]);
}

// ---------------------------------------------------------------------------
// Device info + JSON metadata
// ---------------------------------------------------------------------------
inline int attr(cudaDeviceAttr a, int dev) {
  int v = -1;
  if (cudaDeviceGetAttribute(&v, a, dev) != cudaSuccess) { cudaGetLastError(); return -1; }
  return v;
}

struct Device {
  int index = 0;
  cudaDeviceProp prop{};
  int driver = 0, runtime = 0;
  size_t free_bytes = 0, total_bytes = 0;
  int l2_bytes = 0;

  // Returns false (after printing why) when no usable device exists.
  bool open(int dev) {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess || n == 0) {
      std::fprintf(stderr, "No usable CUDA device: %s\n",
                   e != cudaSuccess ? cudaGetErrorString(e) : "device count is 0");
      return false;
    }
    index = dev;
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    l2_bytes = attr(cudaDevAttrL2CacheSize, dev);
    return true;
  }
  // Refuse to run if the request exceeds half of the free memory.
  void require(size_t bytes) const {
    if (bytes > free_bytes / 2) {
      std::fprintf(stderr, "Refusing: needs %.1f MiB but only %.1f MiB free (limit 50%%).\n",
                   bytes / 1048576.0, free_bytes / 1048576.0);
      std::exit(4);
    }
  }
};

inline std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') { o.push_back('\\'); o.push_back(c); }
    else if (static_cast<unsigned char>(c) < 0x20) o.push_back(' ');
    else o.push_back(c);
  }
  return o;
}

// Writes the opening brace and device fields; the caller appends benchmark fields with
// meta_kv(), then calls meta_end().
inline void meta_begin(FILE* f, const char* bench, const Device& d, bool smoke) {
  size_t l2_fetch = 0;
  if (cudaDeviceGetLimit(&l2_fetch, cudaLimitMaxL2FetchGranularity) != cudaSuccess) {
    cudaGetLastError();
    l2_fetch = 0;
  }
  const int dev = d.index;
  std::fprintf(f, "{\n  \"benchmark\": \"%s\",\n  \"smoke\": %s,\n", bench, smoke ? "true" : "false");
  std::fprintf(f, "  \"device_index\": %d,\n  \"device_name\": \"%s\",\n", dev,
               json_escape(d.prop.name).c_str());
  std::fprintf(f, "  \"compute_capability\": \"%d.%d\",\n", attr(cudaDevAttrComputeCapabilityMajor, dev),
               attr(cudaDevAttrComputeCapabilityMinor, dev));
  std::fprintf(f, "  \"cuda_driver_version\": %d,\n  \"cuda_runtime_version\": %d,\n", d.driver, d.runtime);
  std::fprintf(f, "  \"nvcc_version\": \"%d.%d\",\n", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__);
  std::fprintf(f, "  \"mem_total_bytes\": %zu,\n  \"mem_free_bytes_at_start\": %zu,\n", d.total_bytes, d.free_bytes);
  std::fprintf(f, "  \"sm_count\": %d,\n", attr(cudaDevAttrMultiProcessorCount, dev));
  std::fprintf(f, "  \"sm_clock_khz_attr\": %d,\n", attr(cudaDevAttrClockRate, dev));
  std::fprintf(f, "  \"mem_clock_khz_attr\": %d,\n", attr(cudaDevAttrMemoryClockRate, dev));
  std::fprintf(f, "  \"mem_bus_width_bits\": %d,\n", attr(cudaDevAttrGlobalMemoryBusWidth, dev));
  std::fprintf(f, "  \"l2_cache_bytes\": %d,\n", d.l2_bytes);
  std::fprintf(f, "  \"l2_fetch_granularity_bytes\": %lld,\n", l2_fetch == 0 ? -1LL : (long long)l2_fetch);
  std::fprintf(f, "  \"max_threads_per_sm\": %d,\n", attr(cudaDevAttrMaxThreadsPerMultiProcessor, dev));
  std::fprintf(f, "  \"max_regs_per_sm\": %d,\n", attr(cudaDevAttrMaxRegistersPerMultiprocessor, dev));
  std::fprintf(f, "  \"max_smem_per_sm_bytes\": %d,\n", attr(cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
}

inline void meta_kv(FILE* f, const char* key, const std::string& json_value) {
  std::fprintf(f, "  \"%s\": %s,\n", key, json_value.c_str());
}

inline void meta_end(FILE* f, const std::vector<std::pair<std::string, size_t>>& correctness, int failures) {
  std::fprintf(f, "  \"correctness\": {");
  for (size_t i = 0; i < correctness.size(); ++i)
    std::fprintf(f, "%s\"%s\": %zu", i ? ", " : "", json_escape(correctness[i].first).c_str(),
                 correctness[i].second);
  std::fprintf(f, "},\n  \"correctness_failures\": %d\n}\n", failures);
}

inline std::string jstr(const std::string& s) { return "\"" + json_escape(s) + "\""; }

}  // namespace dmk

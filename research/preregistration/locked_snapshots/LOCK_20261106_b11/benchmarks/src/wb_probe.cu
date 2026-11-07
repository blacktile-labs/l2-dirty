// wb_probe.cu -- B6: does the L2 write dirty lines back to DRAM before it evicts them?
//
// Motivation: the 2026-11-05 counter run (DECISION_LOG D-032) measured DRAM writes in B4 windows
// whose whole working set fits in L2, where the write-back LRU model of cache_sim.py (assumption
// S3: write-back on eviction only) predicts zero. In B4 the clean footprint (x) and the dirty
// footprint (t, y) always change together, so the data cannot say which one drives the writes.
// This probe varies them independently.
//
// Every launch of rw_kernel streams two disjoint buffers in one pass (element i of both at once):
//   C: r * L2 bytes, read only. ld.global.cg = cached in L2 only, so L1 cannot serve re-reads.
//   D: w * L2 bytes, written. A new value every launch (tag), so no store is ever silent.
// Launch sequence per cell (r, w):
//   flush (16 MiB read) -> warm rw -> steady rw x S -> readback (ld.global.cg of all of D)
// The binary measures no time; DRAM bytes per launch come from ncu (benchmarks/scripts/wb_probe.sh).
// --profiler-ranges brackets steady launches 2..S of every cell with cudaProfilerStart/Stop, so
// `ncu --replay-mode app-range` can measure them back to back, without ncu's per-kernel gaps.
//
// Correctness (before anything is measured, for every cell): D is compared word for word against
// the host reference with guard words on both sides; the XOR of everything rw_kernel read from C
// and everything readback_kernel read from D is compared with the host XOR. After the measured
// sequence D is checked again against the last tag.

#include "common.cuh"

#include <cuda_profiler_api.h>

#include <cmath>

using namespace dmk;

// Value stored at D[i] by the launch with tag `tag`. Depends only on (i, tag).
DMK_HD uint32_t probe_value(size_t i, uint32_t tag) { return gen_value(i, 0xB6000000u ^ tag); }

// XOR over the block; the result is valid in thread 0 only. blockDim.x must be a multiple of 32.
__device__ __forceinline__ uint32_t block_xor(uint32_t v) {
  __shared__ uint32_t warp_acc[32];
  for (int o = 16; o > 0; o >>= 1) v ^= __shfl_xor_sync(0xFFFFFFFFu, v, o);
  if ((threadIdx.x & 31) == 0) warp_acc[threadIdx.x >> 5] = v;
  __syncthreads();
  uint32_t r = 0;
  if (threadIdx.x == 0)
    for (unsigned w = 0; w < (blockDim.x >> 5); ++w) r ^= warp_acc[w];
  return r;
}

__global__ void rw_kernel(const uint32_t* __restrict__ c, size_t nc, uint32_t* __restrict__ d,
                          size_t nd, uint32_t tag, uint32_t* __restrict__ sink) {
  uint32_t acc = 0;
  const size_t n = nc > nd ? nc : nd;
  DMK_GRID_STRIDE_LOOP(i, n) {
    if (i < nc) acc ^= __ldcg(c + i);
    if (i < nd) d[i] = probe_value(i, tag);
  }
  acc = block_xor(acc);
  if (threadIdx.x == 0 && acc != 0) atomicXor(sink, acc);
}

__global__ void readback_kernel(const uint32_t* __restrict__ d, size_t nd, uint32_t* __restrict__ sink) {
  uint32_t acc = 0;
  DMK_GRID_STRIDE_LOOP(i, nd) acc ^= __ldcg(d + i);
  acc = block_xor(acc);
  if (threadIdx.x == 0 && acc != 0) atomicXor(sink, acc);
}

// --one-sector variants (B6b, added 2026-11-05 after B6; rw_kernel/readback_kernel above are
// unchanged). D spans nd words = nd/32 lines of 128 B, but only the first 32-B sector of each line
// is written: line occupancy as in rw_kernel, dirty bytes / 4. Element j of D is written iff
// (j % 32) < 8. Thread i handles written element (i / 8) * 32 + i % 8, so each 8-thread group
// writes one whole sector (no fill read is needed).
DMK_HD size_t sector_elem(size_t i) { return (i >> 3) * 32 + (i & 7); }

__global__ void rw1s_kernel(const uint32_t* __restrict__ c, size_t nc, uint32_t* __restrict__ d,
                            size_t nd, uint32_t tag, uint32_t* __restrict__ sink) {
  uint32_t acc = 0;
  const size_t nw = nd / 4;
  const size_t n = nc > nw ? nc : nw;
  DMK_GRID_STRIDE_LOOP(i, n) {
    if (i < nc) acc ^= __ldcg(c + i);
    if (i < nw) {
      const size_t j = sector_elem(i);
      d[j] = probe_value(j, tag);
    }
  }
  acc = block_xor(acc);
  if (threadIdx.x == 0 && acc != 0) atomicXor(sink, acc);
}

__global__ void readback1s_kernel(const uint32_t* __restrict__ d, size_t nd, uint32_t* __restrict__ sink) {
  uint32_t acc = 0;
  DMK_GRID_STRIDE_LOOP(i, nd / 4) acc ^= __ldcg(d + sector_elem(i));
  acc = block_xor(acc);
  if (threadIdx.x == 0 && acc != 0) atomicXor(sink, acc);
}

struct Cell {
  double r = 0, w = 0;
  size_t nc = 0, nd = 0;
};

static size_t elems_for(double f, int l2_bytes) {
  if (f == 0) return 0;
  size_t n = static_cast<size_t>(f * l2_bytes / sizeof(uint32_t));
  return std::max<size_t>(256, n / 256 * 256);
}

// Default grid: r, w in steps of 1/8 of L2 with 0 < r + w <= 7/8 (a write-back LRU L2 holds the
// whole working set), plus two anchors far above L2 where every model predicts ~1 B per launch.
static std::string default_cells() {
  std::string s;
  for (int ri = 0; ri <= 7; ++ri)
    for (int wi = 0; ri + wi <= 7; ++wi) {
      if (ri == 0 && wi == 0) continue;
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%g:%g,", ri / 8.0, wi / 8.0);
      s += buf;
    }
  return s + "2:0,0:2";
}

int main(int argc, char** argv) {
  Args a(argc, argv, {"--smoke", "--profiler-ranges", "--one-sector"},
         {"--cells", "--steady", "--seed", "--block", "--device", "--meta"});
  if (a.help()) {
    std::printf("Usage: wb_probe [--smoke] [--cells r:w,...] [--steady S] [--profiler-ranges] [--one-sector]\n"
                "                [--seed N] [--block N] [--device N] [--meta PATH]\n"
                "  r, w: clean (read) and dirty (written) footprint as multiples of the device L2 size\n"
                "  default cells: r, w in {0, 1/8, ..., 7/8} with 0 < r + w <= 7/8, plus 2:0 and 0:2\n"
                "  --one-sector: write only the first 32-B sector of every 128-B line of D (rw1s_kernel)\n"
                "--smoke = cells 0.0625:0.0625,1.5:0,0:1.5, steady 3\n");
    return 0;
  }
  const bool smoke = a.flag("--smoke");
  const bool ranges = a.flag("--profiler-ranges");
  const bool one_sector = a.flag("--one-sector");
  const int steady = static_cast<int>(a.i64("--steady", smoke ? 3 : 5));
  const uint32_t seed = static_cast<uint32_t>(a.i64("--seed", 12345));
  const int block = static_cast<int>(a.i64("--block", 256));
  if (steady < 2 || block <= 0 || block > 1024 || block % 32) Args::die("invalid arguments");

  Device dev;
  if (!dev.open(static_cast<int>(a.i64("--device", 0)))) return 3;
  if (dev.l2_bytes <= 0) Args::die("device reports no L2 size");

  std::vector<Cell> cells;
  for (const std::string& tok : a.list("--cells", smoke ? "0.0625:0.0625,1.5:0,0:1.5" : default_cells())) {
    const size_t colon = tok.find(':');
    if (colon == std::string::npos) Args::die("cell must be r:w, got " + tok);
    char* e1 = nullptr;
    char* e2 = nullptr;
    const std::string rs = tok.substr(0, colon), ws = tok.substr(colon + 1);
    Cell c;
    c.r = std::strtod(rs.c_str(), &e1);
    c.w = std::strtod(ws.c_str(), &e2);
    if (*e1 || *e2 || c.r < 0 || c.w < 0 || (c.r == 0 && c.w == 0)) Args::die("bad cell " + tok);
    c.nc = elems_for(c.r, dev.l2_bytes);
    c.nd = elems_for(c.w, dev.l2_bytes);
    cells.push_back(c);
  }
  size_t max_c = 0, max_d = 0;
  for (const Cell& c : cells) {
    max_c = std::max(max_c, c.nc);
    max_d = std::max(max_d, c.nd);
  }
  max_c = std::max<size_t>(max_c, 256);
  max_d = std::max<size_t>(max_d, 256);
  dev.require((max_c + max_d + 2 * kGuardElems) * sizeof(uint32_t) + (64ull << 20));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  Timer timer;  // used only for its L2 flush buffer and kernel
  timer.init(stream, dev.l2_bytes);
  uint32_t* cbuf = nullptr;
  uint32_t* sink = nullptr;
  CUDA_CHECK(cudaMalloc(&cbuf, max_c * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&sink, sizeof(uint32_t)));
  fill_gen(cbuf, max_c, seed, /*unit_float=*/false, stream);
  Guarded d;
  d.alloc(max_d);
  std::vector<uint32_t> host;

  auto grid_for = [&](size_t n) { return grid_1d(std::max<size_t>(n, 1), block); };
  auto rw = [&](const Cell& c, uint32_t tag) {
    if (one_sector)
      rw1s_kernel<<<grid_for(std::max(c.nc, c.nd / 4)), block, 0, stream>>>(cbuf, c.nc, d.p, c.nd, tag, sink);
    else
      rw_kernel<<<grid_for(std::max(c.nc, c.nd)), block, 0, stream>>>(cbuf, c.nc, d.p, c.nd, tag, sink);
  };
  auto readback = [&](const Cell& c) {
    if (one_sector)
      readback1s_kernel<<<grid_for(c.nd / 4), block, 0, stream>>>(d.p, c.nd, sink);
    else
      readback_kernel<<<grid_for(c.nd), block, 0, stream>>>(d.p, c.nd, sink);
  };
  // Expected content of D[j] (j < nd) after a launch with `tag`; unwritten words keep the guard.
  auto expect_d = [&](size_t j, uint32_t tag) {
    return (!one_sector || (j % 32) < 8) ? probe_value(j, tag) : kGuardWord;
  };
  auto read_sink = [&] {
    uint32_t v = 0;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(&v, sink, sizeof(v), cudaMemcpyDeviceToHost));
    return v;
  };
  auto zero_sink = [&] { CUDA_CHECK(cudaMemsetAsync(sink, 0, sizeof(uint32_t), stream)); };

  // ---- correctness for every cell, before anything is measured ----
  std::vector<std::pair<std::string, size_t>> correctness;
  int failures = 0;
  for (const Cell& c : cells) {
    char id[96];
    std::snprintf(id, sizeof(id), "r=%g;w=%g;nc=%zu;nd=%zu", c.r, c.w, c.nc, c.nd);
    const uint32_t tag = 0xC0FFEEu;
    d.arm(stream);
    zero_sink();
    rw(c, tag);
    CUDA_CHECK(cudaGetLastError());
    const uint32_t got_c = read_sink();
    uint32_t want_c = 0, want_d = 0;
    for (size_t i = 0; i < c.nc; ++i) want_c ^= gen_value(i, seed);
    for (size_t i = 0; i < c.nd; ++i)
      if (!one_sector || (i % 32) < 8) want_d ^= probe_value(i, tag);
    size_t bad = d.verify(id, c.nd, [&](size_t i) { return expect_d(i, tag); }, host);
    zero_sink();
    readback(c);
    CUDA_CHECK(cudaGetLastError());
    const uint32_t got_d = read_sink();
    if (got_c != want_c) { std::fprintf(stderr, "  [%s] read XOR of C: 0x%08x != 0x%08x\n", id, got_c, want_c); ++bad; }
    if (got_d != want_d) { std::fprintf(stderr, "  [%s] readback XOR of D: 0x%08x != 0x%08x\n", id, got_d, want_d); ++bad; }
    correctness.push_back({id, bad});
    if (bad) ++failures;
  }
  if (failures) {
    std::fprintf(stderr, "correctness failures in %d cell(s); nothing measured\n", failures);
    return 1;
  }

  // ---- measured sequences (bytes come from the profiler; no timing here) ----
  std::vector<std::pair<std::string, size_t>> post;
  for (const Cell& c : cells) {
    uint32_t tag = 1;
    d.arm(stream);  // guard words for the post-check; the flush below evicts these dirty lines
    timer.flush();
    rw(c, tag++);  // warm: C comes from DRAM, D is allocated in L2
    for (int s = 0; s < steady; ++s) {
      if (ranges && s == 1) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaProfilerStart());
      }
      rw(c, tag++);
    }
    if (ranges) {
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaProfilerStop());
    }
    readback(c);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const uint32_t last = tag - 1;
    char id[96];
    std::snprintf(id, sizeof(id), "post:r=%g;w=%g", c.r, c.w);
    const size_t bad = d.verify(id, c.nd, [&](size_t i) { return expect_d(i, last); }, host);
    post.push_back({id, bad});
    if (bad) ++failures;
    std::printf("r=%-7g w=%-7g nc=%-8zu nd=%-8zu %s\n", c.r, c.w, c.nc, c.nd, bad ? "POST-CHECK FAIL" : "ok");
  }

  FILE* mf = std::fopen(a.str("--meta", "wb_probe.meta.json").c_str(), "w");
  if (!mf) { std::perror("open meta"); return 6; }
  meta_begin(mf, "wb_probe", dev, smoke);
  std::string cj = "[";
  for (size_t i = 0; i < cells.size(); ++i) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%s{\"r\": %g, \"w\": %g, \"nc\": %zu, \"nd\": %zu}", i ? ", " : "",
                  cells[i].r, cells[i].w, cells[i].nc, cells[i].nd);
    cj += buf;
  }
  meta_kv(mf, "cells", cj + "]");
  meta_kv(mf, "steady", std::to_string(steady));
  meta_kv(mf, "profiler_ranges", ranges ? "true" : "false");
  meta_kv(mf, "one_sector", one_sector ? "true" : "false");
  meta_kv(mf, "launch_plan_per_cell",
          "[\"flush_l2_read_kernel\", \"rw_kernel(warm)\", \"rw_kernel(steady) x steady\", \"readback_kernel\"]");
  meta_kv(mf, "block", std::to_string(block));
  meta_kv(mf, "flush_buffer_bytes", std::to_string(timer.flush_elems * sizeof(uint32_t)));
  meta_kv(mf, "seed", std::to_string(seed));
  for (auto& p : post) correctness.push_back(p);
  meta_end(mf, correctness, failures);
  std::fclose(mf);

  d.release();
  CUDA_CHECK(cudaFree(cbuf));
  CUDA_CHECK(cudaFree(sink));
  timer.destroy();
  CUDA_CHECK(cudaStreamDestroy(stream));
  std::printf("correctness_failures=%d\n", failures);
  return failures == 0 ? 0 : 1;
}

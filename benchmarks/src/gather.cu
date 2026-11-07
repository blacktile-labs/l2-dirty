// gather.cu -- Gate B baseline #3: gather and scatter-add with regular, random, skewed indices
//
//   gather:       out[i] = table[idx[i]]                 i in [0, M)
//   scatter_add:  atomicAdd(&acc[idx[i]], val[i])        (uint32, wraps mod 2^32)
//
// Index distributions (host-generated, deterministic; see dmk_host.hpp):
//   seq       idx[i] = i mod T
//   uniform   uniform over [0, T)
//   zipf<a>   Zipf(a) ranks mapped through a random permutation, so hot elements are
//             scattered over the table (not pre-packed)
//
// scatter_add uses atomics so that skewed (duplicate-heavy) indices have deterministic
// semantics; it exposes a different competing bottleneck (same-address serialization).
// For each (T, M, dist) the binary also records *exact* statistics of the generated index
// stream (distinct 32-B sectors, share of lookups in the hottest L2-sized set of sectors,
// hottest-element count) -- properties of the input, not GPU measurements.

#include "common.cuh"

using namespace dmk;

__global__ void gather_kernel(const uint32_t* __restrict__ table, const uint32_t* __restrict__ idx,
                              uint32_t* __restrict__ out, size_t m) {
  DMK_GRID_STRIDE_LOOP(i, m) out[i] = table[idx[i]];
}

// Fetch-granularity probe: each lookup also reads the element `flip` slots away. flip = 8
// (32 B) lands in the other 32-B sector of the same 64-B chunk; flip = 16 (64 B) lands in the
// other 64-B half of the same 128-B line. If the hardware already fetched that chunk for the
// first load, the second read is (nearly) free; if it fetches 32 B at a time, it is not.
// Requires T % 32 == 0 so that j ^ flip stays inside the table.
__global__ void gather_pair_kernel(const uint32_t* __restrict__ table, const uint32_t* __restrict__ idx,
                                   uint32_t* __restrict__ out, size_t m, uint32_t flip) {
  DMK_GRID_STRIDE_LOOP(i, m) {
    const uint32_t j = idx[i];
    out[i] = table[j] + table[j ^ flip];
  }
}

__global__ void scatter_add_kernel(const uint32_t* __restrict__ val, const uint32_t* __restrict__ idx,
                                   uint32_t* acc, size_t m) {
  DMK_GRID_STRIDE_LOOP(i, m) atomicAdd(&acc[idx[i]], val[i]);
}

struct IndexStats {
  size_t distinct_sectors = 0;
  double hot_sector_share = 0;  // share of lookups landing in the top-K sectors, K = L2/32
  uint32_t max_element_count = 0;
  size_t k_sectors = 0;
};

static IndexStats index_stats(const std::vector<uint32_t>& idx, uint32_t T, int l2_bytes) {
  IndexStats s;
  const size_t n_sectors = (static_cast<size_t>(T) + 7) / 8;  // 8 uint32 per 32-B sector
  std::vector<uint32_t> sec(n_sectors, 0), elem(T, 0);
  for (uint32_t v : idx) { ++sec[v / 8]; ++elem[v]; }
  for (uint32_t c : sec) s.distinct_sectors += (c != 0);
  s.max_element_count = *std::max_element(elem.begin(), elem.end());
  s.k_sectors = std::min<size_t>(n_sectors, static_cast<size_t>(std::max(l2_bytes, 0)) / 32);
  if (s.k_sectors > 0) {
    std::nth_element(sec.begin(), sec.begin() + (s.k_sectors - 1), sec.end(), std::greater<uint32_t>());
    double top = 0;
    for (size_t k = 0; k < s.k_sectors; ++k) top += sec[k];
    s.hot_sector_share = idx.empty() ? 0.0 : top / static_cast<double>(idx.size());
  }
  return s;
}

int main(int argc, char** argv) {
  Args a(argc, argv, {"--smoke", "--no-flush-l2"},
         {"--table-mib", "--lookups", "--dists", "--reps", "--warmup", "--steady-iters", "--seed",
          "--block", "--device", "--csv", "--meta", "--variants", "--l2-fetch-granularity"});
  if (a.help()) {
    std::printf("Usage: gather [--smoke] [--table-mib A,B,...] [--lookups M] [--dists seq,uniform,zipf1.0,...]\n"
                "              [--reps N] [--warmup N] [--steady-iters N] [--seed N] [--block N]\n"
                "              [--no-flush-l2] [--device N] [--csv PATH] [--meta PATH]\n"
                "              [--variants gather,gather_pair64,gather_pair128,scatter_add]\n"
                "              [--l2-fetch-granularity BYTES]   (hint: cudaLimitMaxL2FetchGranularity)\n"
                "--smoke = table 64 MiB, lookups 4194304, dists seq,uniform,zipf1.0, all variants, reps 10, warmup 3\n");
    return 0;
  }
  const bool smoke = a.flag("--smoke");
  const std::vector<double> table_mibs = a.list_f64("--table-mib", smoke ? "64" : "1,4,16,64,256");
  const size_t M = static_cast<size_t>(a.i64("--lookups", smoke ? (4ll << 20) : (16ll << 20)));
  const std::vector<std::string> dists =
      a.list("--dists", smoke ? "seq,uniform,zipf1.0" : "seq,uniform,zipf0.6,zipf0.8,zipf1.0,zipf1.2,zipf1.4");
  const int reps = static_cast<int>(a.i64("--reps", smoke ? 10 : 30));
  const int warmup = static_cast<int>(a.i64("--warmup", smoke ? 3 : 10));
  const int steady_iters = static_cast<int>(a.i64("--steady-iters", 10));
  const uint32_t seed = static_cast<uint32_t>(a.i64("--seed", 12345));
  const int block = static_cast<int>(a.i64("--block", 256));
  const bool flush = !a.flag("--no-flush-l2");
  const std::vector<std::string> variants =
      a.list("--variants", "gather,gather_pair64,gather_pair128,scatter_add");
  const long long fetch_req = a.i64("--l2-fetch-granularity", -1);
  auto want = [&](const char* v) { return std::find(variants.begin(), variants.end(), v) != variants.end(); };
  for (const std::string& v : variants)
    if (v != "gather" && v != "gather_pair64" && v != "gather_pair128" && v != "scatter_add")
      Args::die("unknown variant: " + v);
  if (M == 0 || M > 0xFFFFFFFFull || reps <= 0 || warmup < 0 || steady_iters <= 0 || block <= 0 ||
      block > 1024 || block % 32)
    Args::die("invalid arguments");
  std::vector<IndexSpec> specs;
  for (const std::string& s : dists) {
    try { specs.push_back(parse_index_spec(s.c_str())); }
    catch (const std::exception&) { Args::die("bad distribution: " + s); }
  }

  Device d;
  if (!d.open(static_cast<int>(a.i64("--device", 0)))) return 3;
  // The limit is only a hint; the driver may clamp or ignore it. meta.json records the value
  // read back (meta_begin), and a failed set is a warning, not an error.
  if (fetch_req >= 0) {
    cudaError_t le = cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, static_cast<size_t>(fetch_req));
    if (le != cudaSuccess) {
      std::fprintf(stderr, "warning: cudaLimitMaxL2FetchGranularity=%lld not applied: %s\n", fetch_req,
                   cudaGetErrorString(le));
      cudaGetLastError();
    }
  }
  size_t max_T = 0;
  for (double mib : table_mibs) {
    if (mib <= 0) Args::die("table size must be > 0");
    max_T = std::max(max_T, static_cast<size_t>(mib * 1048576.0 / sizeof(uint32_t)));
  }
  if (max_T > 0xFFFFFFFFull) Args::die("table too large for uint32 indices");
  d.require((max_T * 2 + M * 3 + 4 * kGuardElems) * sizeof(uint32_t) + (64ull << 20));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  Timer timer;
  timer.init(stream, d.l2_bytes);
  uint32_t *table = nullptr, *idx = nullptr, *val = nullptr;
  CUDA_CHECK(cudaMalloc(&table, max_T * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&idx, M * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&val, M * sizeof(uint32_t)));
  Guarded out, acc;
  out.alloc(M);
  acc.alloc(max_T);
  fill_gen(val, M, seed + 1, false, stream);

  FILE* csv = std::fopen(a.str("--csv", "gather.reps.csv").c_str(), "w");
  if (!csv) { std::perror("open csv"); return 6; }
  std::fputs(CSV_HEADER, csv);
  std::vector<std::pair<std::string, size_t>> correctness;
  std::string stats_json = "{";
  std::vector<uint32_t> host;
  int failures = 0;
  const unsigned grid = grid_1d(M, block);

  for (double mib : table_mibs) {
    const uint32_t T = static_cast<uint32_t>(mib * 1048576.0 / sizeof(uint32_t));
    if (T == 0) Args::die("table too small");
    if ((want("gather_pair64") || want("gather_pair128")) && T % 32 != 0)
      Args::die("pair variants need a table size that is a multiple of 32 elements");
    fill_gen(table, T, seed, false, stream);
    for (size_t di = 0; di < specs.size(); ++di) {
      const std::vector<uint32_t> h_idx = make_indices(specs[di], T, M, seed + 7 * di + 3);
      CUDA_CHECK(cudaMemcpy(idx, h_idx.data(), M * sizeof(uint32_t), cudaMemcpyHostToDevice));
      const std::string params = "T=" + std::to_string(T) + ";M=" + std::to_string(M) + ";dist=" + dists[di];
      const IndexStats st = index_stats(h_idx, T, d.l2_bytes);
      stats_json += std::string(stats_json.size() > 1 ? ", " : "") + jstr(params) + ": {\"distinct_sectors\": " +
                    std::to_string(st.distinct_sectors) + ", \"hot_sectors_k\": " + std::to_string(st.k_sectors) +
                    ", \"hot_sector_share\": " + std::to_string(st.hot_sector_share) +
                    ", \"max_element_count\": " + std::to_string(st.max_element_count) + "}";

      // ---------------- gather (+ fetch-granularity probes) ----------------
      struct GV { const char* name; uint32_t flip; };
      for (const GV& gv : {GV{"gather", 0u}, GV{"gather_pair64", 8u}, GV{"gather_pair128", 16u}}) {
        if (!want(gv.name)) continue;
        const uint32_t flip = gv.flip;
        auto call = [&] {
          if (flip == 0) gather_kernel<<<grid, block, 0, stream>>>(table, idx, out.p, M);
          else gather_pair_kernel<<<grid, block, 0, stream>>>(table, idx, out.p, M, flip);
        };
        // Poison with 0, then (below) with 0xFFFFFFFF: either could legitimately be a table
        // value, but an unwritten slot cannot match the expected value under both poisons.
        out.arm(stream);
        fill_u32(out.p, M, 0u, stream);
        call();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::string id = std::string(gv.name) + "@" + params;
        auto expect = [&](size_t i) -> uint32_t {
          const uint32_t j = h_idx[i];
          return flip == 0 ? gen_value(j, seed) : gen_value(j, seed) + gen_value(j ^ flip, seed);
        };
        size_t bad = out.verify(id.c_str(), M, expect, host);
        if (!bad) {  // second pass with a different poison catches unwritten slots holding 0
          fill_u32(out.p, M, 0xFFFFFFFFu, stream);
          call();
          CUDA_CHECK(cudaGetLastError());
          CUDA_CHECK(cudaStreamSynchronize(stream));
          bad = out.verify(id.c_str(), M, expect, host);
        }
        correctness.push_back({id, bad});
        if (bad) {
          std::fprintf(stderr, "FAIL %s: %zu mismatching words; not timed\n", id.c_str(), bad);
          ++failures;
          continue;
        }
        Row row;
        row.variant = gv.name;
        row.params = params;
        row.n_elems = M;
        // algorithmic bytes: idx + needed table elements (1 or 2 per lookup) + out
        row.read_bytes = (flip == 0 ? 2 : 3) * M * sizeof(uint32_t);
        row.write_bytes = M * sizeof(uint32_t);
        row.timing_mode = "isolated";
        row.flush = flush;
        write_rows(csv, "gather", row, timer.isolated(call, warmup, reps, flush));
        row.timing_mode = "steady";
        row.iters = steady_iters;
        row.flush = false;
        write_rows(csv, "gather", row, timer.steady(call, warmup, reps, steady_iters));
        std::printf("%-14s %-40s ok\n", gv.name, params.c_str());
      }

      // ---------------- scatter_add ----------------
      if (want("scatter_add")) {
        auto call = [&] { scatter_add_kernel<<<grid, block, 0, stream>>>(val, idx, acc.p, M); };
        acc.arm(stream);
        fill_u32(acc.p, T, 0u, stream);
        call();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<uint32_t> expect_acc(T, 0u);
        for (size_t i = 0; i < M; ++i) expect_acc[h_idx[i]] += gen_value(i, seed + 1);
        const std::string id = "scatter_add@" + params;
        const size_t bad = acc.verify(id.c_str(), T, [&](size_t k) { return expect_acc[k]; }, host);
        correctness.push_back({id, bad});
        if (bad) {
          std::fprintf(stderr, "FAIL %s: %zu mismatching words; not timed\n", id.c_str(), bad);
          ++failures;
        } else {
          // Values keep accumulating (wrapping) across timed calls; timing does not depend on them.
          Row row;
          row.variant = "scatter_add";
          row.params = params;
          row.n_elems = M;
          row.read_bytes = 3 * M * sizeof(uint32_t);  // idx + val + acc element (read-modify-write)
          row.write_bytes = M * sizeof(uint32_t);
          row.timing_mode = "isolated";
          row.flush = flush;
          write_rows(csv, "gather", row, timer.isolated(call, warmup, reps, flush));
          row.timing_mode = "steady";
          row.iters = steady_iters;
          row.flush = false;
          write_rows(csv, "gather", row, timer.steady(call, warmup, reps, steady_iters));
          std::printf("scatter_add  %-40s ok\n", params.c_str());
        }
      }
    }
  }
  std::fclose(csv);

  FILE* mf = std::fopen(a.str("--meta", "gather.meta.json").c_str(), "w");
  if (!mf) { std::perror("open meta"); return 6; }
  meta_begin(mf, "gather", d, smoke);
  meta_kv(mf, "lookups", std::to_string(M));
  meta_kv(mf, "block", std::to_string(block));
  meta_kv(mf, "l2_fetch_granularity_requested", std::to_string(fetch_req));
  {
    std::string vj = "[";
    for (size_t i = 0; i < variants.size(); ++i) vj += (i ? ", " : "") + jstr(variants[i]);
    meta_kv(mf, "variants", vj + "]");
  }
  meta_kv(mf, "index_stats", stats_json + "}");
  meta_kv(mf, "flush_l2", flush ? "true" : "false");
  meta_kv(mf, "flush_buffer_bytes", std::to_string(timer.flush_elems * sizeof(uint32_t)));
  meta_kv(mf, "reps", std::to_string(reps));
  meta_kv(mf, "warmup", std::to_string(warmup));
  meta_kv(mf, "steady_iters", std::to_string(steady_iters));
  meta_kv(mf, "seed", std::to_string(seed));
  meta_end(mf, correctness, failures);
  std::fclose(mf);

  out.release();
  acc.release();
  CUDA_CHECK(cudaFree(table));
  CUDA_CHECK(cudaFree(idx));
  CUDA_CHECK(cudaFree(val));
  timer.destroy();
  CUDA_CHECK(cudaStreamDestroy(stream));
  std::printf("correctness_failures=%d\n", failures);
  return failures == 0 ? 0 : 1;
}

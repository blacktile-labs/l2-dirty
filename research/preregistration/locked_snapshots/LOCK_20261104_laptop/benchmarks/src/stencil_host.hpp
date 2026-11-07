// stencil_host.hpp -- host reference + coefficients for B5 (2D periodic integer stencil).
// Pure C++17 (tested by test_host.cpp without a GPU).
//
// One sweep:  out[r][c] = C*x[r][c] + N*x[r-1][c] + S*x[r+1][c] + Wc*x[r][c-1] + E*x[r][c+1]
// on a torus (periodic boundaries), uint32 arithmetic mod 2^32. Because the ring Z/2^32 is
// commutative and the boundaries are periodic, two sweeps equal ONE sweep of the 13-point
// diamond stencil whose coefficients are the self-convolution of the 5-point ones -- bit-exact.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dmk {

struct Coef5 {
  uint32_t c, n, s, w, e;  // center, north (r-1), south (r+1), west (c-1), east (c+1)
};

// Asymmetric on purpose, so that an index mix-up cannot cancel out.
constexpr Coef5 kStencil5 = {3u, 5u, 7u, 11u, 13u};

// Fixed order of the 13 diamond offsets (|dr| + |dc| <= 2), shared by host and device.
constexpr int kOff13[13][2] = {{-2, 0}, {-1, -1}, {-1, 0}, {-1, 1}, {0, -2}, {0, -1}, {0, 0},
                               {0, 1},  {0, 2},   {1, -1}, {1, 0},  {1, 1},  {2, 0}};

struct Coef13 {
  uint32_t v[13];
};

inline Coef13 compose_twice(const Coef5& k) {
  const int o5[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
  const uint32_t c5[5] = {k.c, k.n, k.s, k.w, k.e};
  Coef13 out{};
  for (int a = 0; a < 5; ++a)
    for (int b = 0; b < 5; ++b) {
      const int dr = o5[a][0] + o5[b][0], dc = o5[a][1] + o5[b][1];
      for (int j = 0; j < 13; ++j)
        if (kOff13[j][0] == dr && kOff13[j][1] == dc) out.v[j] += c5[a] * c5[b];
    }
  return out;
}

inline size_t wrap(long long x, long long m) { return static_cast<size_t>(((x % m) + m) % m); }

inline void sweep5_host(const std::vector<uint32_t>& in, std::vector<uint32_t>& out, size_t H, size_t W,
                        const Coef5& k = kStencil5) {
  out.resize(H * W);
  for (size_t r = 0; r < H; ++r) {
    const size_t rn = wrap((long long)r - 1, H), rs = wrap((long long)r + 1, H);
    for (size_t c = 0; c < W; ++c) {
      const size_t cw = wrap((long long)c - 1, W), ce = wrap((long long)c + 1, W);
      out[r * W + c] = k.c * in[r * W + c] + k.n * in[rn * W + c] + k.s * in[rs * W + c] +
                       k.w * in[r * W + cw] + k.e * in[r * W + ce];
    }
  }
}

inline void sweep13_host(const std::vector<uint32_t>& in, std::vector<uint32_t>& out, size_t H, size_t W,
                         const Coef13& k) {
  out.resize(H * W);
  for (size_t r = 0; r < H; ++r)
    for (size_t c = 0; c < W; ++c) {
      uint32_t acc = 0;
      for (int j = 0; j < 13; ++j)
        acc += k.v[j] * in[wrap((long long)r + kOff13[j][0], H) * W + wrap((long long)c + kOff13[j][1], W)];
      out[r * W + c] = acc;
    }
}

}  // namespace dmk

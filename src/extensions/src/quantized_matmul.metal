// Copyright © 2024 Apple Inc.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

template <typename T, bool TRANSPOSE_B>
inline void quantized_matmul_w4a16_g64_impl(
    device const T* scales,
    device const T* biases,
    device const T* a,
    device const uint32_t* b,
    device T* out,
    constant const int& M,
    constant const int& N,
    constant const int& K,
    uint2 gid) {
  const int i = static_cast<int>(gid.x);
  const int k = static_cast<int>(gid.y);
  if (i >= M || k >= K) {
    return;
  }

  constexpr int group_size = 64;
  constexpr int bits = 4;
  constexpr int packs_per_word = 8;  // 32 / bits
  constexpr uint32_t pack_mask = (1u << bits) - 1u;

  const int groups_per_row = N / group_size;
  const int words_per_group = group_size / packs_per_word;
  const int packed_k = N / packs_per_word;

  float sum = 0.0f;
  for (int group_idx = 0; group_idx < groups_per_row; ++group_idx) {
    const int scale_bias_idx = k * groups_per_row + group_idx;
    const float scale = static_cast<float>(scales[scale_bias_idx]);
    const float bias = static_cast<float>(biases[scale_bias_idx]);
    const int n_group_base = group_idx * group_size;

    for (int word_idx = 0; word_idx < words_per_group; ++word_idx) {
      const int packed_col = group_idx * words_per_group + word_idx;
      const int b_idx = TRANSPOSE_B ? (k * packed_k + packed_col)
                                    : (packed_col * K + k);
      const uint32_t packed = b[b_idx];
      const int a_base = i * N + n_group_base + word_idx * packs_per_word;

      // Manually unpack 8 int4 values from one uint32 and accumulate.
      const float w0 = static_cast<float>((packed >> 0) & pack_mask) * scale + bias;
      const float w1 = static_cast<float>((packed >> 4) & pack_mask) * scale + bias;
      const float w2 = static_cast<float>((packed >> 8) & pack_mask) * scale + bias;
      const float w3 = static_cast<float>((packed >> 12) & pack_mask) * scale + bias;
      const float w4 = static_cast<float>((packed >> 16) & pack_mask) * scale + bias;
      const float w5 = static_cast<float>((packed >> 20) & pack_mask) * scale + bias;
      const float w6 = static_cast<float>((packed >> 24) & pack_mask) * scale + bias;
      const float w7 = static_cast<float>((packed >> 28) & pack_mask) * scale + bias;

      sum += static_cast<float>(a[a_base + 0]) * w0;
      sum += static_cast<float>(a[a_base + 1]) * w1;
      sum += static_cast<float>(a[a_base + 2]) * w2;
      sum += static_cast<float>(a[a_base + 3]) * w3;
      sum += static_cast<float>(a[a_base + 4]) * w4;
      sum += static_cast<float>(a[a_base + 5]) * w5;
      sum += static_cast<float>(a[a_base + 6]) * w6;
      sum += static_cast<float>(a[a_base + 7]) * w7;
    }
  }

  out[i * K + k] = static_cast<T>(sum);
}

template <typename T>
[[kernel]] void quantized_matmul_w4a16_g64_t(
    device const T* scales [[buffer(0)]],
    device const T* biases [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const uint32_t* b [[buffer(3)]],
    device T* out [[buffer(4)]],
    constant const int& M [[buffer(5)]],
    constant const int& N [[buffer(6)]],
    constant const int& K [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
  quantized_matmul_w4a16_g64_impl<T, true>(
      scales, biases, a, b, out, M, N, K, gid);
}

template <typename T>
[[kernel]] void quantized_matmul_w4a16_g64_nt(
    device const T* scales [[buffer(0)]],
    device const T* biases [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const uint32_t* b [[buffer(3)]],
    device T* out [[buffer(4)]],
    constant const int& M [[buffer(5)]],
    constant const int& N [[buffer(6)]],
    constant const int& K [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
  quantized_matmul_w4a16_g64_impl<T, false>(
      scales, biases, a, b, out, M, N, K, gid);
}

instantiate_kernel("quantized_matmul_w4a16_g64_t_f16", quantized_matmul_w4a16_g64_t, float16_t);
instantiate_kernel("quantized_matmul_w4a16_g64_t_bf16", quantized_matmul_w4a16_g64_t, bfloat16_t);
instantiate_kernel("quantized_matmul_w4a16_g64_nt_f16", quantized_matmul_w4a16_g64_nt, float16_t);
instantiate_kernel("quantized_matmul_w4a16_g64_nt_bf16", quantized_matmul_w4a16_g64_nt, bfloat16_t);

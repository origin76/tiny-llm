// Copyright © 2024 Apple Inc.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

template <typename T>
[[kernel]] void quantized_matmul_w4a16_g64(
    device const T* scales [[buffer(0)]],
    device const T* biases [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const uint32_t* b [[buffer(3)]],
    device T* out [[buffer(4)]],
    constant const int& M [[buffer(5)]],
    constant const int& N [[buffer(6)]],
    constant const int& K [[buffer(7)]],
    constant const int& transpose_b [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int i = static_cast<int>(gid.x);
  const int k = static_cast<int>(gid.y);
  if (i >= M || k >= K) {
    return;
  }

  constexpr int group_size = 64;
  constexpr int bits = 4;
  constexpr int packs_per_word = 32 / bits;
  constexpr uint32_t pack_mask = (1u << bits) - 1u;

  const int groups_per_row = N / group_size;
  const int words_per_group = group_size / packs_per_word;
  const int packed_k = N / packs_per_word;

  float sum = 0.0f;
  for (int group_idx = 0; group_idx < groups_per_row; ++group_idx) {
    const int scale_bias_idx = k * groups_per_row + group_idx;
    const float scale = static_cast<float>(scales[scale_bias_idx]);
    const float bias = static_cast<float>(biases[scale_bias_idx]);

    for (int word_idx = 0; word_idx < words_per_group; ++word_idx) {
      const int packed_col = group_idx * words_per_group + word_idx;
      const int b_idx = transpose_b ? (k * packed_k + packed_col)
                                    : (packed_col * K + k);
      const uint32_t packed = b[b_idx];
      const int n_base = group_idx * group_size + word_idx * packs_per_word;

      for (int pack_idx = 0; pack_idx < packs_per_word; ++pack_idx) {
        const uint32_t q = (packed >> (pack_idx * bits)) & pack_mask;
        const float w = static_cast<float>(q) * scale + bias;
        const float a_val = static_cast<float>(a[i * N + n_base + pack_idx]);
        sum += a_val * w;
      }
    }
  }

  out[i * K + k] = static_cast<T>(sum);
}

instantiate_kernel(
    "quantized_matmul_w4a16_g64_f16", quantized_matmul_w4a16_g64, float16_t);
instantiate_kernel(
    "quantized_matmul_w4a16_g64_bf16", quantized_matmul_w4a16_g64, bfloat16_t);

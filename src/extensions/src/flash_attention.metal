#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

[[kernel]] void flash_attention_f32_e128(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* mask [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant const int* mask_shape [[buffer(5)]],
    constant const int64_t* mask_strides [[buffer(6)]],
    constant const int& is_causal [[buffer(7)]],
    constant const int& N [[buffer(8)]],
    constant const int& L [[buffer(9)]],
    constant const int& S [[buffer(10)]],
    constant const int& E [[buffer(11)]],
    constant const int& num_kv_heads [[buffer(12)]],
    constant const int& num_heads [[buffer(13)]],
    constant const float& scale [[buffer(14)]],
    constant const int& Br [[buffer(15)]],
    constant const int& Bc [[buffer(16)]],
    [[maybe_unused]] constant const int& Tr [[buffer(17)]],
    constant const int& Tc [[buffer(18)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int kMaxBr = 32;
  constexpr int kMaxE = 128;
  constexpr float kNegLarge = -1.0e9f;

  const int n = static_cast<int>(group_id.x);
  const int tile_i = static_cast<int>(group_id.y);
  const int row = static_cast<int>(simd_gid);
  const int col = static_cast<int>(simd_lid);

  if (n >= N || row >= Br || col >= Bc) {
    return;
  }

  const int q_kv_ratio = num_heads / num_kv_heads;
  const int kv_head = n / q_kv_ratio;
  const int row_idx = tile_i * Br + row;
  const bool row_valid = row_idx < L;
  const int causal_offset = S - L;

  threadgroup float q_local[kMaxBr][kMaxE];
  threadgroup float o_local[kMaxBr * kMaxE];

  if (col == 0) {
    for (int e = 0; e < E; ++e) {
      q_local[row][e] = row_valid ? q[n * L * E + row_idx * E + e] : 0.0f;
      o_local[row * E + e] = 0.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float m_i = kNegLarge;
  float l_i = 0.0f;

  for (int tile_j = 0; tile_j < Tc; ++tile_j) {
    const int row_end = min((tile_i + 1) * Br - 1, L - 1);
    const int col_start = tile_j * Bc;
    if (is_causal && col_start > row_end + causal_offset) {
      continue;
    }

    const int col_idx = col_start + col;
    const bool col_valid = col_idx < S;
    const device float* k_tile = k + kv_head * S * E + col_start * E;
    const device float* v_tile = v + kv_head * S * E + col_start * E;

    float score = 0.0f;
    if (row_valid && col_valid) {
      for (int e = 0; e < E; ++e) {
        score += q_local[row][e] * k_tile[col * E + e];
      }
      score *= scale;

      const int col_end = min((tile_j + 1) * Bc - 1, S - 1);
      const int row_start = tile_i * Br;
      const bool block_all_valid = is_causal && (col_end <= row_start + causal_offset);
      if (!block_all_valid) {
        const int64_t mask_linear_idx =
            static_cast<int64_t>(n) * L * S + static_cast<int64_t>(row_idx) * S + col_idx;
        const int64_t mask_idx = elem_to_loc(mask_linear_idx, mask_shape, mask_strides, 3);
        score += mask[mask_idx];
      }
    } else {
      score = kNegLarge;
    }

    const float row_max = simd_max(score);
    const float next_m = max(m_i, row_max);
    const float prev_scale = exp(m_i - next_m);
    m_i = next_m;

    float prob = 0.0f;
    if (row_valid && col_valid) {
      prob = exp(score - m_i);
    }

    const float row_sum = simd_sum(prob);
    l_i = prev_scale * l_i + row_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = 0; e < E; ++e) {
      float weighted_v = 0.0f;
      if (row_valid && col_valid) {
        weighted_v = prob * v_tile[col * E + e];
      }
      const float value_sum = simd_sum(weighted_v);
      if (col == 0 && row_valid) {
        o_local[row * E + e] = prev_scale * o_local[row * E + e] + value_sum;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (col == 0 && row_valid) {
    const float inv_l = 1.0f / l_i;
    for (int e = 0; e < E; ++e) {
      out[n * L * E + row_idx * E + e] = o_local[row * E + e] * inv_l;
    }
  }
}

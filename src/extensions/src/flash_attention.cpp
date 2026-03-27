// Copyright © 2024-2025 Apple Inc.

#include "tiny_llm_ext.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace tiny_llm_ext {

namespace {

void validate_flash_attention_inputs(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& mask,
    int num_kv_heads,
    int num_heads) {
    if (q.dtype() != mx::float32 || k.dtype() != mx::float32 || v.dtype() != mx::float32 ||
        mask.dtype() != mx::float32) {
        throw std::runtime_error("flash_attention: q, k, v, and mask must be float32");
    }
    if (q.ndim() != 3 || k.ndim() != 3 || v.ndim() != 3 || mask.ndim() != 3) {
        throw std::runtime_error("flash_attention: q, k, v, and mask must be 3D arrays");
    }
    if (num_heads <= 0 || num_kv_heads <= 0) {
        throw std::runtime_error("flash_attention: num_heads and num_kv_heads must be positive");
    }
    if (num_heads % num_kv_heads != 0) {
        throw std::runtime_error("flash_attention: num_heads must be divisible by num_kv_heads");
    }

    const auto& q_shape = q.shape();
    const auto& k_shape = k.shape();
    const auto& v_shape = v.shape();
    const auto& mask_shape = mask.shape();

    if (q_shape[2] != k_shape[2] || q_shape[2] != v_shape[2]) {
        throw std::runtime_error("flash_attention: q.shape[2] must match k.shape[2] and v.shape[2]");
    }
    if (k_shape[1] != v_shape[1]) {
        throw std::runtime_error("flash_attention: k.shape[1] must be equal to v.shape[1]");
    }
    if (q_shape[0] % static_cast<size_t>(num_heads) != 0) {
        throw std::runtime_error("flash_attention: q.shape[0] must be divisible by num_heads");
    }
    if (k_shape[0] % static_cast<size_t>(num_kv_heads) != 0 ||
        v_shape[0] % static_cast<size_t>(num_kv_heads) != 0) {
        throw std::runtime_error("flash_attention: k.shape[0] and v.shape[0] must be divisible by num_kv_heads");
    }

    const size_t q_batches = q_shape[0] / static_cast<size_t>(num_heads);
    const size_t k_batches = k_shape[0] / static_cast<size_t>(num_kv_heads);
    const size_t v_batches = v_shape[0] / static_cast<size_t>(num_kv_heads);
    if (q_batches != k_batches || q_batches != v_batches) {
        throw std::runtime_error("flash_attention: q and kv head batches must map consistently");
    }

    if (mask_shape[0] != q_shape[0] || mask_shape[1] != q_shape[1] || mask_shape[2] != k_shape[1]) {
        throw std::runtime_error("flash_attention: mask must have shape [q.shape[0], q.shape[1], k.shape[1]]");
    }
}

}  // namespace

mx::array flash_attention(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& mask,
    float scale,
    bool is_causal,
    int num_kv_heads,
    int num_heads,
    mx::StreamOrDevice s) {
    validate_flash_attention_inputs(q, k, v, mask, num_kv_heads, num_heads);

    return mx::array(
        q.shape(),
        mx::float32,
        std::make_shared<FlashAttention>(to_stream(s), scale, is_causal, num_kv_heads, num_heads),
        std::vector<mx::array>{q, k, v, mask});
}

void FlashAttention::eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) {
    auto& q = inputs[0];
    auto& k = inputs[1];
    auto& v = inputs[2];
    auto& mask = inputs[3];
    auto& out = outputs[0];

    if (out.dtype() != mx::float32) {
        throw std::runtime_error("flash_attention: output dtype must be float32");
    }
    if (!q.flags().row_contiguous) {
        throw std::runtime_error("flash_attention: q must be contiguous");
    }
    if (!k.flags().row_contiguous) {
        throw std::runtime_error("flash_attention: k must be contiguous");
    }
    if (!v.flags().row_contiguous) {
        throw std::runtime_error("flash_attention: v must be contiguous");
    }

    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto& encoder = mx::cpu::get_command_encoder(stream());
    encoder.set_input_array(q);
    encoder.set_input_array(k);
    encoder.set_input_array(v);
    encoder.set_input_array(mask);
    encoder.set_output_array(out);

    encoder.dispatch([out_ptr = out.data<float>(),
                      q = mx::array::unsafe_weak_copy(q),
                      k = mx::array::unsafe_weak_copy(k),
                      v = mx::array::unsafe_weak_copy(v),
                      mask = mx::array::unsafe_weak_copy(mask),
                      num_heads = num_heads_,
                      num_kv_heads = num_kv_heads_,
                      scale = scale_,
                      is_causal = is_causal_]() {
        constexpr int64_t Br = 32;
        constexpr int64_t Bc = 32;

        const auto& q_shape = q.shape();
        const auto& k_shape = k.shape();
        const auto& mask_shape = mask.shape();
        const auto& mask_strides = mask.strides();

        const int64_t N = static_cast<int64_t>(q_shape[0]);
        const int64_t L = static_cast<int64_t>(q_shape[1]);
        const int64_t E = static_cast<int64_t>(q_shape[2]);
        const int64_t S = static_cast<int64_t>(k_shape[1]);
        const int64_t Tr = (L + Br - 1) / Br;
        const int64_t Tc = (S + Bc - 1) / Bc;
        const int64_t q_head_stride = L * E;
        const int64_t kv_head_stride = S * E;
        const int64_t q_kv_heads_ratio = num_heads / num_kv_heads;
        const int64_t causal_offset = S - L;

        const float* q_ptr = q.data<float>();
        const float* k_ptr = k.data<float>();
        const float* v_ptr = v.data<float>();
        const float* mask_ptr = mask.data<float>();

        for (int64_t n = 0; n < N; ++n) {
            const float* q_batch = q_ptr + n * q_head_stride;
            const int64_t kv_head = n / q_kv_heads_ratio;
            const float* k_batch = k_ptr + kv_head * kv_head_stride;
            const float* v_batch = v_ptr + kv_head * kv_head_stride;

            for (int64_t i = 0; i < Tr; ++i) {
                const int64_t row_start = i * Br;
                const int64_t row_count = std::min<int64_t>(Br, L - row_start);
                const int64_t row_end = row_start + row_count - 1;

                std::vector<float> q_tile(row_count * E, 0.0f);
                std::vector<float> o_tile(row_count * E, 0.0f);
                std::vector<float> m_i(row_count, -std::numeric_limits<float>::infinity());
                std::vector<float> l_i(row_count, 0.0f);

                for (int64_t row = 0; row < row_count; ++row) {
                    const int64_t q_offset = (row_start + row) * E;
                    for (int64_t e = 0; e < E; ++e) {
                        q_tile[row * E + e] = q_batch[q_offset + e];
                    }
                }

                for (int64_t j = 0; j < Tc; ++j) {
                    const int64_t col_start = j * Bc;
                    if (is_causal && col_start > row_end + causal_offset) {
                        continue;
                    }

                    const int64_t col_count = std::min<int64_t>(Bc, S - col_start);
                    const int64_t col_end = col_start + col_count - 1;
                    const bool block_all_valid = is_causal && (col_end <= row_start + causal_offset);

                    std::vector<float> k_tile(col_count * E, 0.0f);
                    std::vector<float> v_tile(col_count * E, 0.0f);
                    for (int64_t col = 0; col < col_count; ++col) {
                        const int64_t kv_offset = (col_start + col) * E;
                        for (int64_t e = 0; e < E; ++e) {
                            k_tile[col * E + e] = k_batch[kv_offset + e];
                            v_tile[col * E + e] = v_batch[kv_offset + e];
                        }
                    }

                    std::vector<float> scores(row_count * col_count, 0.0f);
                    for (int64_t row = 0; row < row_count; ++row) {
                        for (int64_t col = 0; col < col_count; ++col) {
                            float score = 0.0f;
                            for (int64_t e = 0; e < E; ++e) {
                                score += q_tile[row * E + e] * k_tile[col * E + e];
                            }
                            score *= scale;
                            if (!block_all_valid) {
                                const int64_t mask_linear_idx =
                                    n * L * S + (row_start + row) * S + (col_start + col);
                                const int64_t mask_idx = mx::elem_to_loc(mask_linear_idx, mask_shape, mask_strides);
                                score += mask_ptr[mask_idx];
                            }
                            scores[row * col_count + col] = score;
                        }
                    }

                    std::vector<float> m_prev_scale(row_count, 0.0f);
                    for (int64_t row = 0; row < row_count; ++row) {
                        float tile_row_max = -std::numeric_limits<float>::infinity();
                        for (int64_t col = 0; col < col_count; ++col) {
                            tile_row_max = std::max(tile_row_max, scores[row * col_count + col]);
                        }
                        const float next_m = std::max(m_i[row], tile_row_max);
                        m_prev_scale[row] = std::exp(m_i[row] - next_m);
                        m_i[row] = next_m;
                    }

                    std::vector<float> probs(row_count * col_count, 0.0f);
                    for (int64_t row = 0; row < row_count; ++row) {
                        for (int64_t col = 0; col < col_count; ++col) {
                            probs[row * col_count + col] = std::exp(scores[row * col_count + col] - m_i[row]);
                        }
                    }

                    for (int64_t row = 0; row < row_count; ++row) {
                        float row_sum = 0.0f;
                        for (int64_t col = 0; col < col_count; ++col) {
                            row_sum += probs[row * col_count + col];
                        }
                        l_i[row] = m_prev_scale[row] * l_i[row] + row_sum;
                    }

                    for (int64_t row = 0; row < row_count; ++row) {
                        for (int64_t e = 0; e < E; ++e) {
                            float value_accum = 0.0f;
                            for (int64_t col = 0; col < col_count; ++col) {
                                value_accum += probs[row * col_count + col] * v_tile[col * E + e];
                            }
                            o_tile[row * E + e] = m_prev_scale[row] * o_tile[row * E + e] + value_accum;
                        }
                    }
                }

                for (int64_t row = 0; row < row_count; ++row) {
                    for (int64_t e = 0; e < E; ++e) {
                        out_ptr[n * q_head_stride + (row_start + row) * E + e] = o_tile[row * E + e] / l_i[row];
                    }
                }
            }
        }
    });
}

#ifdef _METAL_

void FlashAttention::eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) {
    auto& q = inputs[0];
    auto& k = inputs[1];
    auto& v = inputs[2];
    auto& mask = inputs[3];
    auto& out = outputs[0];

    if (out.dtype() != mx::float32) {
        throw std::runtime_error("flash_attention: output dtype must be float32");
    }
    if (!q.flags().row_contiguous) {
        throw std::runtime_error("flash_attention: q must be contiguous");
    }
    if (!k.flags().row_contiguous) {
        throw std::runtime_error("flash_attention: k must be contiguous");
    }
    if (!v.flags().row_contiguous) {
        throw std::runtime_error("flash_attention: v must be contiguous");
    }

    const int N = static_cast<int>(q.shape()[0]);
    const int L = static_cast<int>(q.shape()[1]);
    const int S = static_cast<int>(k.shape()[1]);
    const int E = static_cast<int>(q.shape()[2]);
    constexpr int Br = 32;
    constexpr int Bc = 32;

    if (E > 128) {
        throw std::runtime_error("flash_attention: GPU kernel currently requires E <= 128");
    }

    const int Tr = (L + Br - 1) / Br;
    const int Tc = (S + Bc - 1) / Bc;

    auto& s = stream();
    auto& d = mx::metal::device(s.device);
    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto library = d.get_library("tiny_llm_ext");
    auto kernel = d.get_kernel("flash_attention_f32_e128", library);

    auto& compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(k, 1);
    compute_encoder.set_input_array(v, 2);
    compute_encoder.set_input_array(mask, 3);
    compute_encoder.set_output_array(out, 4);
    compute_encoder.set_vector_bytes(mask.shape(), 5);
    compute_encoder.set_vector_bytes(mask.strides(), 6);

    compute_encoder.set_bytes(static_cast<int>(is_causal_), 7);
    compute_encoder.set_bytes(N, 8);
    compute_encoder.set_bytes(L, 9);
    compute_encoder.set_bytes(S, 10);
    compute_encoder.set_bytes(E, 11);
    compute_encoder.set_bytes(num_kv_heads_, 12);
    compute_encoder.set_bytes(num_heads_, 13);
    compute_encoder.set_bytes(scale_, 14);
    compute_encoder.set_bytes(Br, 15);
    compute_encoder.set_bytes(Bc, 16);
    compute_encoder.set_bytes(Tr, 17);
    compute_encoder.set_bytes(Tc, 18);

    const size_t tgp_size = kernel->maxTotalThreadsPerThreadgroup();
    const size_t simd_width = kernel->threadExecutionWidth();
    if (simd_width != 32) {
        throw std::runtime_error("flash_attention: expected threadExecutionWidth == 32");
    }
    if (static_cast<size_t>(Br * Bc) > tgp_size) {
        throw std::runtime_error("flash_attention: Br * Bc exceeds max threads per threadgroup");
    }

    MTL::Size grid_dims = MTL::Size(static_cast<size_t>(N), static_cast<size_t>(Tr), 1);
    MTL::Size group_dims = MTL::Size(static_cast<size_t>(Bc), static_cast<size_t>(Br), 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

#else

void FlashAttention::eval_gpu(const std::vector<mx::array>&, std::vector<mx::array>&) {
    throw std::runtime_error("flash_attention: GPU implementation not available");
}

#endif

void FlashAttention::print(std::ostream& os) {
    os << name() << "(scale=" << scale_ << ", is_causal=" << is_causal_
       << ", num_kv_heads=" << num_kv_heads_ << ", num_heads=" << num_heads_ << ")";
}

std::vector<mx::array> FlashAttention::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
    throw std::runtime_error("FlashAttention: JVP not supported");
}

std::vector<mx::array> FlashAttention::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
    throw std::runtime_error("FlashAttention: VJP not supported");
}

std::pair<std::vector<mx::array>, std::vector<int>> FlashAttention::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
    throw std::runtime_error("FlashAttention: vmap not supported");
}

bool FlashAttention::is_equivalent(const mx::Primitive& other) const {
    const auto& rhs = static_cast<const FlashAttention&>(other);
    return scale_ == rhs.scale_ && is_causal_ == rhs.is_causal_ &&
           num_kv_heads_ == rhs.num_kv_heads_ && num_heads_ == rhs.num_heads_;
}

}  // namespace tiny_llm_ext

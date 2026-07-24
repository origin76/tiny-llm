#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace tiny_llm_ext {

void load_library(const char *path);

///////////////////////////////////////////////////////////////////////////////
// Operation
///////////////////////////////////////////////////////////////////////////////

/**
 * Quantized matrix multiplication
 * Computes: a @ dequant(b, scales, biases)
 *
 * @param scales Quantization scales, shape [n, k/group_size]
 * @param biases Quantization biases, shape [n, k/group_size]
 * @param group_size Size of each quantization group
 * @param bits Number of bits per packed value (currently only 4 is supported)
 * @param a Input array, shape [..., m, k]
 * @param b Quantized weight array. If transpose_b is true, shape [n, k*bits/32];
 *          otherwise shape [k*bits/32, n]
 * @param transpose_b Whether b is provided in transposed matmul layout
 * @param s Stream on which to schedule the operation
 **/
mx::array quantized_matmul(
    const mx::array& scales,      // Quantization scales
    const mx::array& biases,      // Quantization biases
    int group_size,               // Quantization group size
    int bits,                     // Bits per packed value (only 4 is supported)
    const mx::array& a,           // Input activation
    const mx::array& b,           // Quantized weight
    bool transpose_b = false,     // Whether to transpose b
    mx::StreamOrDevice s = {}     // Stream
);

///////////////////////////////////////////////////////////////////////////////
// Primitive
///////////////////////////////////////////////////////////////////////////////

class QuantizedMatmul : public mx::Primitive {
public:
    explicit QuantizedMatmul(
        mx::Stream stream,
        int group_size,
        int bits,
        bool transpose_b
    ) : mx::Primitive(stream),
        group_size_(group_size),
        bits_(bits),
        transpose_b_(transpose_b) {}

    /**
     * A primitive must know how to evaluate itself on the CPU/GPU
     * for the given inputs and populate the output array.
     */
    void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
    void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;

    /** The Jacobian-vector product. */
    std::vector<mx::array> jvp(const std::vector<mx::array>& primals,
                               const std::vector<mx::array>& tangents,
                               const std::vector<int>& argnums) override;

    /** The vector-Jacobian product. */
    std::vector<mx::array> vjp(const std::vector<mx::array>& primals,
                               const std::vector<mx::array>& cotangents,
                               const std::vector<int>& argnums,
                               const std::vector<mx::array>& outputs) override;

    /** Vectorized mapping. */
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(
        const std::vector<mx::array>& inputs,
        const std::vector<int>& axes) override;

    /** Print the primitive. */
    void print(std::ostream& os);

    /** Name of the primitive. */
    const char* name() const override { return "QuantizedMatmul"; }

    /** Equivalence check. */
    bool is_equivalent(const mx::Primitive& other) const override;

private:
    int group_size_;
    int bits_;
    bool transpose_b_;
};

/**
 * Flash attention
 * Computes: softmax(q @ k.T * scale + mask) @ v
 *
 * All inputs are flattened to 3D head-batch tensors:
 * q:    [N, L, E]
 * k/v:  [N_kv, S, E]
 * mask: [N, L, S]
 *
 * @param q Query tensor
 * @param k Key tensor
 * @param v Value tensor
 * @param mask Attention mask tensor
 * @param scale Attention scaling factor
 * @param is_causal Whether to enable causal-mask fast path
 * @param num_kv_heads Number of KV heads before flattening batch dims
 * @param num_heads Number of query heads before flattening batch dims
 * @param s Stream on which to schedule the operation
 **/
mx::array flash_attention(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& mask,
    float scale,
    bool is_causal,
    int num_kv_heads,
    int num_heads,
    mx::StreamOrDevice s = {});

class FlashAttention : public mx::Primitive {
public:
    explicit FlashAttention(
        mx::Stream stream,
        float scale,
        bool is_causal,
        int num_kv_heads,
        int num_heads)
        : mx::Primitive(stream),
          scale_(scale),
          is_causal_(is_causal),
          num_kv_heads_(num_kv_heads),
          num_heads_(num_heads) {}

    void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
    void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;

    std::vector<mx::array> jvp(const std::vector<mx::array>& primals,
                               const std::vector<mx::array>& tangents,
                               const std::vector<int>& argnums) override;

    std::vector<mx::array> vjp(const std::vector<mx::array>& primals,
                               const std::vector<mx::array>& cotangents,
                               const std::vector<int>& argnums,
                               const std::vector<mx::array>& outputs) override;

    std::pair<std::vector<mx::array>, std::vector<int>> vmap(
        const std::vector<mx::array>& inputs,
        const std::vector<int>& axes) override;

    void print(std::ostream& os);

    const char* name() const override { return "FlashAttention"; }

    bool is_equivalent(const mx::Primitive& other) const override;

private:
    float scale_;
    bool is_causal_;
    int num_kv_heads_;
    int num_heads_;
};

}  // namespace tiny_llm_ext

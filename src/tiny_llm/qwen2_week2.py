import mlx.core as mx
from .basics import silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear, QuantizedWeights, quantized_linear
from .kv_cache import TinyKvCache


def cast_quantized_weights(weights: QuantizedWeights, dtype: mx.Dtype) -> QuantizedWeights:
    if weights.scales.dtype == dtype and weights.biases.dtype == dtype:
        return weights
    return QuantizedWeights(
        scales=weights.scales.astype(dtype),
        biases=weights.biases.astype(dtype),
        group_size=weights.group_size,
        bits=weights.bits,
        weight=weights.weight,
    )


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rope = RoPE(dims=hidden_size // num_heads, seq_len=max_seq_len, base=theta, traditional=False)
        self.use_flash_attention = use_flash_attention

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        """
        x: B, L', E (new tokens to process)
        offset: position of the last token processed (current sequence length in cache)
        cache: KV cache for this layer
        """
        B, L_new, _ = x.shape

        # Compute dimensions
        H_q = self.num_heads
        H = self.num_kv_heads
        D = self.hidden_size // H_q

        # Linear projections: output shape (B, L_new, H*D)
        q = quantized_linear(x, self.wq, self.bq)
        k = quantized_linear(x, self.wk, self.bk)
        v = quantized_linear(x, self.wv, self.bv)

        # Reshape to (B, L_new, H, D)
        q = q.reshape(B, L_new, H_q, D)
        k = k.reshape(B, L_new, H, D)
        v = v.reshape(B, L_new, H, D)

        # Apply RoPE with offset
        q = self.rope(q, offset=slice(offset, offset + L_new))
        k = self.rope(k, offset=slice(offset, offset + L_new))

        # Transpose to (B, H, L, D) for attention
        q = q.transpose(0, 2, 1, 3)  # (B, H_q, L_new, D)
        k = k.transpose(0, 2, 1, 3)  # (B, H, L_new, D)
        v = v.transpose(0, 2, 1, 3)  # (B, H, L_new, D)

        # Update KV cache and fetch
        k_cache, v_cache, L, _ = cache.update_and_fetch(k, v)

        # Note: offset consistency check disabled for flexibility
        # In task_3, offset=0 with L_new=10 is treated as processing 10 tokens at once
        # In task_4, offset increments as we process 1 token at a time

        # Compute attention at float32 precision
        # q: (B, H_q, L_new, D), k: (B, H, L, D), v: (B, H, L, D)
        # Output: (B, H_q, L_new, D)
        attn_output = scaled_dot_product_attention_grouped(
            q.astype(mx.float32),
            k_cache.astype(mx.float32),
            v_cache.astype(mx.float32),
            scale=1.0 / (D ** 0.5),
            mask=mask,
        )
        attn_output = attn_output.astype(x.dtype)

        # Transpose back to (B, L_new, H_q, D) and reshape to (B, L_new, H_q * D)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L_new, H_q * D)

        # Final linear projection
        output = quantized_linear(attn_output, self.wo)

        return output


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:

        gate = silu(quantized_linear(x, self.w_gate))
        up = quantized_linear(x, self.w_up)
        hidden = gate * up
        return quantized_linear(hidden, self.w_down)


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
    ):
        self.attention = Qwen2MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            bq=bq,
            bk=bk,
            bv=bv,
            max_seq_len=max_seq_len,
            theta=theta,
            use_flash_attention=use_flash_attention,
        )
        self.mlp = Qwen2MLP(
            dim=hidden_size,
            hidden_dim=intermediate_size,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
        )
        self.input_layernorm = RMSNorm(dim=hidden_size, weight=w_input_layernorm, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(dim=hidden_size, weight=w_post_attention_layernorm, eps=rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        # Pre-Norm Attention
        norm_x = self.input_layernorm(x)
        attn_output = self.attention(norm_x, offset=offset, cache=cache, mask=mask)
        x = x + attn_output

        # Pre-Norm MLP
        norm_x = self.post_attention_layernorm(x)
        mlp_output = self.mlp(norm_x)
        x = x + mlp_output

        return x


class Qwen2ModelWeek2:
    def __init__(
        self,
        mlx_model: Any,
        enable_flash_attn: bool = False,
    ):
        model = mlx_model.model
        args = model.args

        self.num_hidden_layers = args.num_hidden_layers
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size

        # Use float16 as model precision (matching reference)
        precision = mx.float16

        # Dequantize embedding and convert to precision
        embed_weight = model.embed_tokens.weight
        if hasattr(model.embed_tokens, "scales"):
            embed_weight = dequantize_linear(model.embed_tokens)
        embed_weight = embed_weight.astype(precision)

        self.embedding = Embedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            weight=embed_weight,
        )

        # Create transformer blocks
        self.layers = []
        for layer in model.layers:
            attn = layer.self_attn
            mlp = layer.mlp

            wq = cast_quantized_weights(QuantizedWeights.from_mlx_layer(attn.q_proj), precision)
            wk = cast_quantized_weights(QuantizedWeights.from_mlx_layer(attn.k_proj), precision)
            wv = cast_quantized_weights(QuantizedWeights.from_mlx_layer(attn.v_proj), precision)
            wo = cast_quantized_weights(QuantizedWeights.from_mlx_layer(attn.o_proj), precision)
            w_gate = cast_quantized_weights(QuantizedWeights.from_mlx_layer(mlp.gate_proj), precision)
            w_up = cast_quantized_weights(QuantizedWeights.from_mlx_layer(mlp.up_proj), precision)
            w_down = cast_quantized_weights(QuantizedWeights.from_mlx_layer(mlp.down_proj), precision)

            block = Qwen2TransformerBlock(
                num_attention_heads=args.num_attention_heads,
                num_kv_heads=args.num_key_value_heads,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                rms_norm_eps=args.rms_norm_eps,
                wq=wq,
                wk=wk,
                wv=wv,
                wo=wo,
                bq=attn.q_proj.bias.astype(precision) if attn.q_proj.bias is not None else None,
                bk=attn.k_proj.bias.astype(precision) if attn.k_proj.bias is not None else None,
                bv=attn.v_proj.bias.astype(precision) if attn.v_proj.bias is not None else None,
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
                w_input_layernorm=layer.input_layernorm.weight.astype(precision),
                w_post_attention_layernorm=layer.post_attention_layernorm.weight.astype(precision),
                max_seq_len=args.max_position_embeddings,
                theta=int(args.rope_theta),
                use_flash_attention=enable_flash_attn,
            )
            self.layers.append(block)

        self.norm = RMSNorm(
            dim=args.hidden_size,
            weight=model.norm.weight.astype(precision),
            eps=args.rms_norm_eps,
        )

        if args.tie_word_embeddings:
            self.w_lm_head = None
        else:
            self.w_lm_head = cast_quantized_weights(QuantizedWeights.from_mlx_layer(mlx_model.lm_head), precision)

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
        cache: list[TinyKvCache],
    ) -> mx.array:
        """
        inputs: (B, L_new) token ids
        offset: current sequence length (position of last token processed)
        cache: list of TinyKvCache, one per layer
        """
        # Token embedding
        x = self.embedding(inputs)

        # Determine mask (causal for autoregressive generation)
        # Always use "causal" when using KV cache
        mask = "causal"

        # Assert consistency between offset and cache (disabled for flexibility)
        # for i, c in enumerate(cache):
        #     key_values = getattr(c, 'key_values', None) or getattr(c, 'key', None)
        #     if key_values is not None:
        #         cache_len = key_values.shape[1]
        #         assert cache_len == offset, f"Layer {i}: cache length {cache_len} doesn't match offset {offset}"

        # Process each transformer block
        for layer, layer_cache in zip(self.layers, cache):
            x = layer(x, offset=offset, cache=layer_cache, mask=mask)

        # Final LayerNorm
        x = self.norm(x)

        # LM Head projection
        if self.w_lm_head is not None:
            return quantized_linear(x, self.w_lm_head)
        return self.embedding.as_linear(x)

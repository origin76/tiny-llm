import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
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

    # x: B, L, E
    # q = linear(x, wq, bq) -> B, L, H_q, D
    # k = linear(x, wk, bk) -> B, L, H, D
    # v = linear(x, wv, bv) -> B, L, H, D
    # q = rope(q, offset=slice(0, L))
    # k = rope(k, offset=slice(0, L))
    # (transpose as needed)
    # x = scaled_dot_product_attention_grouped(q, k, v, scale, mask) -> B, L, H_q, D ; Do this at float32 precision
    # (transpose as needed)
    # x = linear(x, wo) -> B, L, E
    def __call__(
    self,
    x: mx.array,
    mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, E = x.shape
        
        # 修正维度定义
        H_q = self.num_heads           # query heads 总数（例如 8）
        H = self.num_kv_heads          # kv heads 总数（例如 2 或 8）
        D = self.hidden_size // H_q    # 每个 head 的维度
        
        # 线性投影：输出形状 (B, L, H*D)
        q = linear(x, self.wq, self.bq)  # (B, L, H_q * D)
        k = linear(x, self.wk, self.bk)  # (B, L, H * D)
        v = linear(x, self.wv, self.bv)  # (B, L, H * D)

        # Reshape 并 Transpose 到 attention 函数期望的格式：
        # 从 (B, L, H, D) -> (B, H, L, D)
        q = q.reshape(B, L, H_q, D)
        k = k.reshape(B, L, H, D)
        v = v.reshape(B, L, H, D)

        rope = RoPE(dims=D, seq_len=self.max_seq_len, base=self.theta, traditional=False)
        
        # 假设 RoPE 作用于最后两维或需要特定形状，这里假设 RoPE 接受 (B, H, L, D)
        q = rope(q, offset=slice(0, L))
        k = rope(k, offset=slice(0, L))
        
        q = q.transpose(0, 2, 1, 3)  # (B, H_q, L, D)
        k = k.transpose(0, 2, 1, 3)  # (B, H, L, D)
        v = v.transpose(0, 2, 1, 3)  # (B, H, L, D)

        # 调用 GQA attention
        # 输入: q=(B, H_q, L, D), k=(B, H, L, D), v=(B, H, L, D)
        # 输出: (B, H_q, L, D)
        x = scaled_dot_product_attention_grouped(
            q, k, v, 
            scale=1.0 / (D ** 0.5), 
            mask=mask
        )
        
        # 转置回 (B, L, H_q, D) 然后 reshape 为 (B, L, H_q*D)
        x = x.transpose(0, 2, 1, 3).reshape(B, L, H_q * D)
        
        # 最终线性投影
        x = linear(x, self.wo)  # (B, L, E)
        
        return x


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        gate = silu(linear(x, self.w_gate))  # (B,T,I)
        up = linear(x, self.w_up)            # (B,T,I)
        hidden = gate * up                   # (B,T,I)
        return linear(hidden, self.w_down)   # (B,T,H)


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
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

    def __call__(self, x: mx.array, mask=None):

        # Pre-Norm Attention
        norm_x = self.input_layernorm(x)
        attn_output = self.attention(norm_x, mask=mask)
        x = x + attn_output

        # Pre-Norm MLP
        norm_x = self.post_attention_layernorm(x)
        mlp_output = self.mlp(norm_x)
        x = x + mlp_output

        return x
    
def _deq(layer) -> mx.array:
    """
    Dequantize a layer if quantized, otherwise return its weight directly.

    dequantize_linear(layer) internally calls:
        mx.dequantize(layer.weight, layer.scales, layer.biases,
                      layer.group_size, layer.bits)

    Quantized MLX layers have a `scales` attribute; plain layers don't.
    """
    if hasattr(layer, "scales"):
        return dequantize_linear(layer)   # pass the full layer object
    return layer.weight

# class ModelArgs(BaseModelArgs):
#     model_type: str
#     hidden_size: int
#     num_hidden_layers: int
#     intermediate_size: int
#     num_attention_heads: int
#     rms_norm_eps: float
#     vocab_size: int
#     num_key_value_heads: int
#     max_position_embeddings: int = 32768
#     rope_theta: float = 1000000
#     rope_traditional: bool = False
#     rope_scaling: Optional[Dict[str, Union[float, str]]] = None
#     tie_word_embeddings: bool = True
class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        model = mlx_model.model
        args = model.args
        
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        
        embed_weight = _deq(model.embed_tokens)
        # Ensure the embedding weight is float32 for downstream computation
        embed_weight = embed_weight.astype(mx.float32)

        self.embedding = Embedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            weight=embed_weight,
        )
        
        self.layers = []
        for layer in model.layers:
            attn = layer.self_attn
            mlp = layer.mlp
            
            block = Qwen2TransformerBlock(
                num_attention_heads=args.num_attention_heads,
                num_kv_heads=args.num_key_value_heads,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                rms_norm_eps=args.rms_norm_eps,
                wq=_deq(attn.q_proj),
                wk=_deq(attn.k_proj),
                wv=_deq(attn.v_proj),
                wo=_deq(attn.o_proj),
                bq=attn.q_proj.bias,
                bk=attn.k_proj.bias,
                bv=attn.v_proj.bias,
                w_gate=_deq(mlp.gate_proj),
                w_up=_deq(mlp.up_proj),
                w_down=_deq(mlp.down_proj),
                w_input_layernorm=layer.input_layernorm.weight,
                w_post_attention_layernorm=layer.post_attention_layernorm.weight,
                max_seq_len=args.max_position_embeddings,
                theta=int(args.rope_theta),
            )
            self.layers.append(block)
            
        self.norm = RMSNorm(
            dim=args.hidden_size,
            weight=model.norm.weight,
            eps=args.rms_norm_eps,
        )
        
        if args.tie_word_embeddings:
            self.lm_head_weight = embed_weight
        else:
            self.lm_head_weight = _deq(mlx_model.lm_head)

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        # inputs: (B, L) token ids

        # 1. Token embedding -> (B, L, E)
        x = self.embedding(inputs)

        # 2. 构建 causal mask
        B, L = inputs.shape
        mask = "causal" if L > 1 else None

        # 3. 逐层 Transformer Block
        for layer in self.layers:
            x = layer(x, mask=mask)

        # 4. 最终 LayerNorm
        x = self.norm(x)

        # 5. LM Head 投影 -> (B, L, vocab_size)
        logits = linear(x, self.lm_head_weight)

        return logits

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
        pass

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass

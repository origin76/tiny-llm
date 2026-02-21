import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    key = key.swapaxes(-1,-2)
    softmax_input = mx.matmul(query, key)
    if scale is None:
        scale = 1.0 / (query.shape[-1] ** 0.5)
    softmax_input = softmax_input * scale
    if mask is not None:
        softmax_input = softmax_input + mask
    attention_weights = softmax(softmax_input, axis=-1)
    output = mx.matmul(attention_weights, value)
    return output


class SimpleMultiHeadAttention:
    ### w_q/w_k/w_v: (H x D) x E
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    ### input shapes: N * L * E
    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        N, L, _ = query.shape
        H = self.num_heads
        D = self.head_dim

        # --------------------------------
        # 1️⃣ Linear projections
        # --------------------------------
        Q = query @ self.wq.T # (N, L, E) @ (E, H*D) → (N, L, H*D)
        K = key   @ self.wk.T
        V = value @ self.wv.T

        # --------------------------------
        # 2️⃣ Split heads
        # (N, L, H*D) → (N, L, H, D)
        # --------------------------------
        Q = Q.reshape(N, L, H, D)
        K = K.reshape(N, L, H, D)
        V = V.reshape(N, L, H, D)

        # --------------------------------
        # 3️⃣ Move head dimension forward
        # (N, L, H, D) → (N, H, L, D)
        # --------------------------------
        Q = Q.swapaxes(1, 2)
        K = K.swapaxes(1, 2)
        V = V.swapaxes(1, 2)

        # --------------------------------
        # 5️⃣ Scaled dot-product attention
        # --------------------------------
        attn_output = scaled_dot_product_attention_simple(
            Q, K, V, mask=mask
        )  # (N, H, L, D)

        # (N, H, L, D) → (N, L, H, D)
        attn_output = attn_output.swapaxes(1, 2)

        # (N, L, H, D) → (N, L, E)
        attn_output = attn_output.reshape(N, L, self.hidden_size)

        # --------------------------------
        # 7️⃣ Final linear projection
        # --------------------------------
        out = attn_output @ self.wo.T  # (N, L, E)

        return out


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    i = mx.arange(L)[:, None]           # (L,1)
    j = mx.arange(S)[None, :]           # (1,S)

    shift = S - L

    mask = mx.where(j <= i + shift, 0.0, float("-inf"))
    return mask.astype(dtype)

# N.. is zero or more dimensions for batches
# H_q is the number of query heads
# H is the number of key/value heads (H_q must be divisible by H)
# L is the query sequence length
# S is the key/value sequence length
# D is the head dimension
# query: N.. x H_q x L x D
# key: N.. x H x S x D
# value: N.. x H x S x D
# mask: N.. x H_q x L x S
# output: N.. x H_q x L x D
def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    H_q = query.shape[-3]
    H_kv = key.shape[-3]
    assert H_q % H_kv == 0, "Number of query heads must be divisible by number of key/value heads"

    n_repeats = H_q // H_kv
    if n_repeats > 1:
        key = mx.repeat(key, repeats=n_repeats, axis=-3)
        value = mx.repeat(value, repeats=n_repeats, axis=-3)
    
    if isinstance(mask, str) and mask == "causal":
        L = query.shape[-2]
        S = key.shape[-2]
        mask = causal_mask(L, S, dtype=query.dtype)
        while mask.ndim < query.ndim:
            mask = mask[None]

    return scaled_dot_product_attention_simple(
        query, key, value, scale=scale, mask=mask
    )


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass

import mlx.core as mx


class FastRMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        pass

    def __call__(self, x: mx.array) -> mx.array:
        pass


class FastRoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        pass

    def __call__(self, x: mx.array, offset: int | list[int] | mx.array = 0) -> mx.array:
        pass


def swiglu(gate: mx.array, up: mx.array) -> mx.array:
    pass

def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    return mx.triu(
            mx.full((L, S), -mx.inf),
            k= S - L + 1,
        ).astype(dtype)
    
def scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
    mask: mx.array | str | None = None,
) -> mx.array:
    s = query @ key.swapaxes(-1, -2)
    if scale is None:
        scale = 1.0 / mx.sqrt(key.shape[-1])
    s = s * scale
    if mask is not None:
        if isinstance(mask, str):
            if mask != "causal":
                raise ValueError(f"Unknown mask type: {mask!r}")

            # [L, S]，自动共享给所有 batch/head/group
            mask = causal_mask(query.shape[-2], key.shape[-2], query.dtype)
    if mask is not None:
        s = s + mask
    a = mx.softmax(s, axis=-1)
    return a @ value


def decode_attention_custom(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass

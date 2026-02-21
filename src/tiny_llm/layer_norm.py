import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype

        # 1️⃣ 转 float32 做计算
        x_float = x.astype(mx.float32)

        # 2️⃣ float32 accumulate
        rms = mx.sqrt(
            mx.mean(x_float * x_float, axis=-1, keepdims=True) + self.eps
        )

        # 3️⃣ 归一化
        out = x_float / rms

        # 4️⃣ cast 回原 dtype
        out = out.astype(orig_dtype)

        return out * self.weight

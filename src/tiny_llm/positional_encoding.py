import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        assert dims % 2 == 0, "D must be even for simplicity."

        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional

        # 计算频率
        # shape: (D//2,)
        inv_freq = 1.0 / (
            base ** (mx.arange(0, dims, 2) / dims)
        )

        # 位置索引
        # shape: (seq_len,)
        positions = mx.arange(seq_len)

        # 外积 -> (seq_len, D//2)
        freqs = positions[:, None] * inv_freq[None, :]

        # 预计算 cos / sin
        self.cos_freqs = mx.cos(freqs)  # (seq_len, D//2)
        self.sin_freqs = mx.sin(freqs)  # (seq_len, D//2)

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:

        N, L, H, D = x.shape
        assert D == self.dims

        if offset is None:
            start = 0
        else:
            start = offset.start

        end = start + L

        # 取对应位置的 cos/sin
        cos = self.cos_freqs[start:end]  # (L, D//2)
        sin = self.sin_freqs[start:end]  # (L, D//2)

        # 扩展到 (1, L, 1, D//2)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

        if self.traditional:
            x = x.reshape(N, L, H, D // 2, 2)

            x_even = x[..., 0]
            x_odd = x[..., 1]

            out_even = x_even * cos - x_odd * sin
            out_odd = x_even * sin + x_odd * cos

            out = mx.stack([out_even, out_odd], axis=-1)
            out = out.reshape(N, L, H, D)

            return out

        else:
            HALF = D // 2

            x1 = x[..., :HALF]
            x2 = x[..., HALF:]

            y1 = x1 * cos - x2 * sin
            y2 = x1 * sin + x2 * cos

            out = mx.concatenate([y1, y2], axis=-1)

            return out
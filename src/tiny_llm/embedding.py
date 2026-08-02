import mlx.core as mx

from .quantize import QuantizedWeights, dequantize_weights, quantized_linear


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight[x]
    
    # Embedding::as_linear
    # weight: vocab_size x embedding_dim
    # Input: N.. x embedding_dim
    # Output: N.. x vocab_size
    def as_linear(self, x: mx.array) -> mx.array:
        # 将嵌入层视为线性层
        # weight: (vocab_size, embedding_dim)
        # x: (batch_size, seq_len)
        # output: (batch_size, seq_len, embedding_dim)
        return x @ self.weight.T


class QuantizedEmbedding:
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        weight: QuantizedWeights,
        use_custom_kernel: bool = False,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight
        self.use_custom_kernel = use_custom_kernel

    def __call__(self, x: mx.array) -> mx.array:
        return dequantize_weights(
            self.weight.weight[x],
            self.weight.scales[x],
            None if self.weight.biases is None else self.weight.biases[x],
            self.weight.group_size,
            self.weight.bits,
        )

    def as_linear(self, x: mx.array) -> mx.array:
        return quantized_linear(x, self.weight)

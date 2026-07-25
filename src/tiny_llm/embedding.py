import mlx.core as mx
from .quantize import QuantizedWeights


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
        pass

    def __call__(self, x: mx.array) -> mx.array:
        pass

    def as_linear(self, x: mx.array) -> mx.array:
        pass
